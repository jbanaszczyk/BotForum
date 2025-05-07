import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Protocol, Optional, Dict, List, Self

import requests
import yaml


class ConfigurationError(Exception):
    pass


class LogLevel(Enum):
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass
class AppConfig:
    base_url: str = ""
    model: str = ""
    judge_model: str = ""
    models_to_compare: List[str] = field(default_factory=list)
    system_prompts: Dict[str, str] = field(default_factory=dict)
    judge_system_prompt: str = ""
    log_level: LogLevel = LogLevel.INFO


@dataclass
class ChatBotConfig:
    exit_commands: List[str]
    reset_command: str


class ConfigRepository(Protocol):
    def load_config(self) -> AppConfig: ...


class YamlConfigRepository(ConfigRepository):
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> AppConfig:
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)

            if not config:
                raise ConfigurationError("Configuration file is empty or invalid")

            models_to_compare = self._get_models_to_compare(config)
            judge_model = self._get_judge_model(config)
            return AppConfig(
                base_url=self._get_base_url(config),
                judge_model=judge_model,
                models_to_compare=models_to_compare,
                system_prompts=self._get_system_prompts(config, judge_model, models_to_compare),
                judge_system_prompt=self._get_judge_system_prompt(config),
                log_level=self._get_log_level(config)
            )
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file '{self.config_file}' not found")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Configuration loading failed: {e}")

    def _get_base_url(self, config: Dict) -> str:
        return config.get("ollama_url", "http://localhost:11434")

    def _get_judge_model(self, config: Dict) -> str:
        for model_entry in config.get("models", []):
            if isinstance(model_entry, dict):
                model_name = next(iter(model_entry.keys()))
                model_config = model_entry[model_name]
                if model_config and model_config.get("judge") is True:
                    return model_name
        
        if config.get("judge") and config["judge"].get("model"):
            return config["judge"]["model"]
            
        raise ConfigurationError("Judge model not found. Add 'judge: true' attribute to one of the models.")

    def _get_models_to_compare(self, config: Dict) -> List[str]:
        if not config.get("models"):
            raise ConfigurationError("Missing models in configuration")

        models = []
        for model_entry in config["models"]:
            if isinstance(model_entry, dict):
                models.append(next(iter(model_entry.keys())))
            elif isinstance(model_entry, str):
                models.append(model_entry)

        if not models:
            raise ConfigurationError("No models defined in configuration")

        return models

    def _get_system_prompts(self, config: Dict, judge_model: str, models_to_compare: List[str]) -> Dict[str, str]:
        prompts = {}
        default_prompt = config.get("default_system_prompt", "You are a helpful assistant.")

        # Set judge model prompt
        if config.get("judge") and config["judge"].get("system_prompt"):
            prompts[judge_model] = config["judge"]["system_prompt"]
        else:
            prompts[judge_model] = default_prompt

        # Set model prompts
        for model_entry in config.get("models", []):
            if isinstance(model_entry, dict):
                model_name = next(iter(model_entry.keys()))
                model_config = model_entry[model_name]
                if model_config and model_config.get("system_prompt"):
                    prompts[model_name] = model_config["system_prompt"]
                else:
                    prompts[model_name] = default_prompt
            elif isinstance(model_entry, str):
                prompts[model_entry] = default_prompt

        return prompts

    def _get_judge_system_prompt(self, config: Dict) -> str:
        if not config.get("judge") or not config["judge"].get("system_prompt"):
            return "Provide detailed and comprehensive responses"
        return config["judge"]["system_prompt"]

    def _get_log_level(self, config: Dict) -> LogLevel:
        log_level_str = config.get("log_level", "INFO").upper()
        try:
            return LogLevel[log_level_str]
        except KeyError:
            valid_levels = ", ".join([level.name for level in LogLevel])
            raise ConfigurationError(
                f"Invalid log_level '{log_level_str}'. Must be one of: {valid_levels}"
            )


class ResponseResult:
    def __init__(self, content: str = "", thinking: Optional[str] = None, error: Optional[str] = None):
        self.content = content
        self.thinking = thinking
        self.error = error

    @property
    def is_successful(self) -> bool:
        return self.error is None

    @classmethod
    def parse_model_response(cls, raw_response: str) -> Self:
        try:
            thinking, content = None, raw_response
            if "<think>" in raw_response and "</think>" in raw_response:
                think_start = raw_response.find("<think>")
                think_end = raw_response.find("</think>")
                thinking = raw_response[think_start + 7:think_end].strip()
                content = (raw_response[:think_start] + raw_response[think_end + 8:]).strip()
            return cls(content=content, thinking=thinking)
        except Exception as e:
            return cls(error=f"Response parsing error: {e}")


@dataclass
class ModelResponse:
    model_name: str
    response: ResponseResult


class LLMClient(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def send_message(self, message: str) -> ResponseResult:
        pass

    @abstractmethod
    def reset_history(self) -> None:
        pass


class OllamaLLMClient(LLMClient):
    def __init__(self, config: AppConfig, logger: logging.Logger):
        super().__init__()
        self.config = config
        self.logger = logger
        self.session = requests.Session()
        self.base_url = config.base_url
        self.model = config.model

        if not self._is_model_available():
            error_msg = (f"Model '{self.model}' is not available locally. "
                         f"Use 'ollama pull {self.model}' command to download the model.")
            raise ConfigurationError(error_msg)

        self.system_prompt = config.system_prompts[self.model]
        self.history: List[Message] = []
        self.logger.info(f"OllamaLLMClient initialized. Using model: {self.model}")

    def _is_model_available(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            available_models = [model["name"] for model in response.json()["models"]]
            return self.model in available_models
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error checking model availability (network/HTTP error): {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error checking model availability: {e}")
            return False

    def send_message(self, message_text: str) -> ResponseResult:
        try:
            self.logger.debug(f"Sending message to model {self.model}: {message_text[:50]}...")

            user_message = Message(role="user", content=message_text)
            self.history.append(user_message)

            messages_payload = [Message(role="system", content=self.system_prompt)] + self.history

            payload = {
                "model": self.model,
                "messages": [asdict(msg) for msg in messages_payload],
                "stream": False
            }

            response = self.session.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()

            result = response.json()
            self.logger.debug(f"Received response from model {self.model}")

            assistant_response_content = result["message"]["content"]
            parsed_response = ResponseResult.parse_model_response(assistant_response_content)

            if parsed_response.is_successful:
                assistant_message = Message(role="assistant", content=parsed_response.content)
                self.history.append(assistant_message)
                return parsed_response
            else:
                return parsed_response

        except Exception as e:
            self.logger.error(f"LLM request to model {self.model} failed: {e}")
            if self.history and self.history[-1].role == "user":
                self.history.pop()
            return ResponseResult(error=f"LLM request failed: {e}")

    def reset_history(self) -> None:
        self.history = []
        self.logger.info("Conversation history has been reset.")


class JudgeBot:
    def __init__(self, config: AppConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.judge_client = OllamaLLMClient(
            AppConfig(
                base_url=config.base_url,
                model=config.judge_model,
                system_prompts=config.system_prompts,
                judge_system_prompt=config.judge_system_prompt,
                log_level=config.log_level
            ),
            logger
        )
        self.judge_system_prompt = config.judge_system_prompt

        # Load judge format configuration
        try:
            with open('config.yaml', 'r') as file:
                yaml_config = yaml.safe_load(file)

            judge_config = yaml_config.get('judge', {})
            self.user_question_prefix = judge_config.get('user_question_prefix', 'User question:')
            self.model_responses_prefix = judge_config.get('model_responses_prefix', 'Model responses:')
            self.model_response_format = judge_config.get('model_response_format',
                                                          'Model {model_name} response:\n{response_content}')

            response_format = judge_config.get('response_format', {})
            self.response_format_header = response_format.get('header', 'Response format:')

            points = response_format.get('points', [])
            if points:
                self.response_format_points = "\n".join([f"{i + 1}. {point}" for i, point in enumerate(points)])
            else:
                self.response_format_points = "1. Brief evaluation of each response\n2. If needed, own improved response"

        except Exception as e:
            logger.warning(f"Failed to load judge format from YAML, using defaults: {e}")
            self.user_question_prefix = 'User question:'
            self.model_responses_prefix = 'Model responses:'
            self.model_response_format = 'Model {model_name} response:\n{response_content}'
            self.response_format_header = 'Response format:'
            self.response_format_points = "1. Brief evaluation of each response\n2. If needed, own improved response"

    def evaluate_responses(self, user_input: str, model_responses: List[ModelResponse]) -> ResponseResult:
        evaluation_prompt = self._create_evaluation_prompt(user_input, model_responses)
        return self.judge_client.send_message(evaluation_prompt)

    def _create_evaluation_prompt(self, user_input: str, model_responses: List[ModelResponse]) -> str:
        responses_text = "\n\n".join([
            self.model_response_format.format(
                model_name=resp.model_name,
                response_content=resp.response.content
            )
            for resp in model_responses
        ])

        return (
            f"{self.user_question_prefix} {user_input}\n\n"
            f"{self.model_responses_prefix}\n{responses_text}\n\n"
            f"{self.judge_system_prompt}\n"
            f"{self.response_format_header}\n"
            f"{self.response_format_points}"
        )


class ChatBot:
    def __init__(self,
                 models_to_compare: List[LLMClient],
                 judge_bot: JudgeBot,
                 logger: logging.Logger,
                 config: ChatBotConfig):
        self.models_to_compare = models_to_compare
        self.judge_bot = judge_bot
        self.logger = logger
        self.config = config

    def start(self) -> None:
        commands_info = (
            f"Type '{', '.join(self.config.exit_commands)}' to end. "
            f"Type '{self.config.reset_command}' to clear history."
        )
        self.logger.info(f"Application started. {commands_info}")

        while True:
            try:
                user_input = input("You: ")
                if self._should_exit(user_input):
                    break
                if user_input.lower() == self.config.reset_command:
                    for client in self.models_to_compare:
                        client.reset_history()
                    print("AI: Conversation history has been reset.")
                    continue
                self._process_user_input(user_input)
            except KeyboardInterrupt:
                self.logger.info("Application terminated by user (KeyboardInterrupt).")
                break
            except Exception as e:
                self.logger.error(f"An unexpected error occurred: {e}")
                print(f"Error: An unexpected error occurred. Please check logs.")

    def _should_exit(self, user_input: str) -> bool:
        return user_input.lower() in self.config.exit_commands

    def _process_user_input(self, user_input: str) -> None:
        self.logger.debug(f"Processing user input: {user_input[:50]}...")

        model_responses = []
        for client in self.models_to_compare:
            response = client.send_message(user_input)
            if response.is_successful:
                model_responses.append(ModelResponse(client.model, response))
                print(f"\n[{client.model}]: {response.content}")
                self.logger.debug(f"Response from {client.model}: {response.content[:50]}...")
            else:
                self.logger.error(f"Error processing for {client.model}: {response.error}")
                print(f"Error [{client.model}]: {response.error}")

        if model_responses:
            judge_response = self.judge_bot.evaluate_responses(user_input, model_responses)
            if judge_response.is_successful:
                print(f"\n[JUDGE]: {judge_response.content}")
                self.logger.debug(f"Judge evaluation: {judge_response.content[:50]}...")
            else:
                self.logger.error(f"Error in judge evaluation: {judge_response.error}")
                print(f"Error [JUDGE]: {judge_response.error}")


def configure_logging(log_level: LogLevel = LogLevel.INFO) -> logging.Logger:
    logging.basicConfig(
        level=log_level.value,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def main():
    try:
        config_repository = YamlConfigRepository('config.yaml')
        app_config = config_repository.load_config()
        logger = configure_logging(app_config.log_level)

        with open('config.yaml', 'r') as file:
            yaml_config = yaml.safe_load(file)

        commands_config = yaml_config.get('commands', [])
        exit_commands = []
        reset_command = 'reset'

        for command in commands_config:
            if isinstance(command, dict):
                if 'exit' in command:
                    exit_commands.extend(command['exit'])
                if 'reset' in command:
                    reset_command = command['reset'][0] if command['reset'] else 'reset'

        if not exit_commands:
            exit_commands = ['exit', 'quit', 'bye']

        chat_bot_config = ChatBotConfig(
            exit_commands=exit_commands,
            reset_command=reset_command
        )

        models_to_compare = []
        for model in app_config.models_to_compare:
            client_config = AppConfig(
                base_url=app_config.base_url,
                model=model,
                judge_model=app_config.judge_model,
                system_prompts=app_config.system_prompts,
                judge_system_prompt=app_config.judge_system_prompt,
                log_level=app_config.log_level
            )
            models_to_compare.append(OllamaLLMClient(client_config, logger))

        judge_bot = JudgeBot(app_config, logger)
        chat_bot = ChatBot(
            models_to_compare=models_to_compare,
            judge_bot=judge_bot,
            logger=logger,
            config=chat_bot_config
        )
        chat_bot.start()

    except ConfigurationError as e:
        print(f"Configuration error: {e}")
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Configuration error: {e}", exc_info=False)
    except Exception as e:
        print(f"Unexpected application error: {e}")
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Unexpected application error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
