import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Protocol, Optional, Dict, List, Self, override

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
class JudgeFormatConfig:
    user_question_prefix: str = "User question:"
    model_responses_prefix: str = "Model responses:"
    model_response_format: str = "Model {model_name} response:\n{response_content}"
    response_format_header: str = "Response format:"
    response_format_points: str = """
        1. Conduct an evaluation of the responses.
        2. If necessary, provide an improved version of the response.
        When improving responses, take into account all provided correct answers.
        3. Each evaluation should include the following:
            - A rating of the answer's correctness (on a scale from 0 to 100), where 100 indicates a high degree of accuracy.
            - A rating of the answer's overall quality (on a scale from 0 to 100), where 100 indicates exceptional quality.
            - A concise written assessment of the response.
            Quality rating should not exceed correctness rating.
    """


@dataclass
class AppConfig:
    base_url: str = ""
    model: str = ""
    models: List[str] = field(default_factory=list)
    judge_models: List[str] = field(default_factory=list)
    system_prompts: Dict[str, str] = field(default_factory=dict)
    judge_system_prompt: str = ""
    log_level: LogLevel = LogLevel.INFO
    judge_format: JudgeFormatConfig = field(default_factory=JudgeFormatConfig)
    http_timeout: float = 30.0


@dataclass
class ChatBotConfig:
    exit_commands: List[str]
    reset_commands: List[str]


class ConfigRepository(Protocol):
    def load_config(self) -> tuple[AppConfig, Dict]: ...

    def get_chat_bot_config(self, config: Dict) -> ChatBotConfig: ...


class YamlConfigRepository(ConfigRepository):
    def __init__(self, config_file="config.yaml"):
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)

    def load_config(self) -> tuple[AppConfig, Dict]:
        try:
            with open(self.config_file, 'r') as file:
                config = yaml.safe_load(file)

            if not config:
                raise ConfigurationError("Configuration file is empty or invalid")

            models = self._get_models(config)
            judge_models = self._get_judge_models(config)
            app_config = AppConfig(
                base_url=self._get_base_url(config),
                models=models,
                judge_models=judge_models,
                system_prompts=self._get_system_prompts(config, judge_models),
                judge_system_prompt=self._get_judge_system_prompt(config),
                log_level=self._get_log_level(config),
                judge_format=self._get_judge_format_config(config),
                http_timeout=self._get_http_timeout(config)
            )

            return app_config, config
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file '{self.config_file}' not found")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Configuration loading failed: {e}")

    @staticmethod
    def _get_base_url(config: Dict) -> str:
        return config.get("ollama_url", "http://localhost:11434")

    @staticmethod
    def _get_http_timeout(config: Dict) -> float:
        return float(config.get("http_timeout", 30.0))

    @staticmethod
    def _get_judge_models(config: Dict) -> List[str]:
        judge_models = []

        for model_entry in config.get("models", []):
            if isinstance(model_entry, dict):
                model_name = next(iter(model_entry.keys()))
                model_config = model_entry[model_name]
                if model_config and model_config.get("judge") is True:
                    judge_models.append(model_name)

        if config.get("judge") and config["judge"].get("model"):
            judge_models.append(config["judge"]["model"])

        if not judge_models:
            raise ConfigurationError("Judge model not found. Add 'judge: true' attribute to at least one of the models.")

        return judge_models

    @staticmethod
    def _get_models(config: Dict) -> List[str]:
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

    @staticmethod
    def _get_system_prompts(config: Dict, judge_models: List[str]) -> Dict[str, str]:
        prompts = {}
        default_prompt = config.get("default_system_prompt", "You are a helpful assistant.")

        if config.get("judge") and config["judge"].get("system_prompt"):
            judge_prompt = config["judge"]["system_prompt"]
            for judge_model in judge_models:
                prompts[judge_model] = judge_prompt
        else:
            for judge_model in judge_models:
                prompts[judge_model] = default_prompt

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

    @staticmethod
    def _get_judge_system_prompt(config: Dict) -> str:
        if not config.get("judge") or not config["judge"].get("system_prompt"):
            return "Provide detailed and comprehensive responses"
        return config["judge"]["system_prompt"]

    @staticmethod
    def _get_log_level(config: Dict) -> LogLevel:
        log_level_str = config.get("log_level", "INFO").upper()
        try:
            return LogLevel[log_level_str]
        except KeyError:
            valid_levels = ", ".join([level.name for level in LogLevel])
            raise ConfigurationError(f"Invalid log_level '{log_level_str}'. Must be one of: {valid_levels}")

    @staticmethod
    def _validate_config_values(validations: List[tuple]) -> None:
        for value, error_message in validations:
            if not value:
                raise ConfigurationError(error_message)

    def _get_judge_format_config(self, config: Dict) -> JudgeFormatConfig:
        if 'judge' not in config:
            raise ConfigurationError("Missing 'judge' section in configuration")

        judge_config = config['judge']

        user_question_prefix = judge_config.get('user_question_prefix', '')
        model_responses_prefix = judge_config.get('model_responses_prefix', '')
        model_response_format = judge_config.get('model_response_format', '')
        response_format = judge_config.get('response_format', {})
        response_format_header = response_format.get('header', '') if response_format else ''
        points = response_format.get('points', []) if response_format else []

        self._validate_config_values([
            (user_question_prefix, "Missing 'user_question_prefix' in judge configuration"),
            (model_responses_prefix, "Missing 'model_responses_prefix' in judge configuration"),
            (model_response_format, "Missing 'model_response_format' in judge configuration"),
            (response_format, "Missing 'response_format' in judge configuration"),
            (response_format_header, "Missing 'header' in judge response_format configuration"),
            (points, "Missing 'points' in judge response_format configuration")
        ])

        response_format_points = "\n".join([f"{index + 1}. {point}" for index, point in enumerate(points)])

        return JudgeFormatConfig(
            user_question_prefix=user_question_prefix,
            model_responses_prefix=model_responses_prefix,
            model_response_format=model_response_format,
            response_format_header=response_format_header,
            response_format_points=response_format_points
        )

    def get_chat_bot_config(self, config: Dict) -> ChatBotConfig:
        if 'commands' not in config:
            raise ConfigurationError("Missing 'commands' section in configuration")

        commands_config = config['commands']
        exit_commands = []
        reset_commands = []

        for command in commands_config:
            if isinstance(command, dict):
                if 'exit' in command:
                    exit_commands.extend([cmd.lower() for cmd in command['exit']])
                if 'reset' in command:
                    reset_commands.extend([cmd.lower() for cmd in command['reset']])

        if not exit_commands:
            raise ConfigurationError("No exit commands defined in configuration. Add at least one exit command.")

        if not reset_commands:
            raise ConfigurationError("No reset commands defined in configuration. Add at least one reset command.")

        return ChatBotConfig(
            exit_commands=exit_commands,
            reset_commands=reset_commands
        )


@dataclass
class ResponseResult:
    content: str = ""
    thinking: Optional[str] = None
    error: Optional[str] = None

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
        
    @abstractmethod
    def close(self) -> None:
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
            error_msg = f"Model '{self.model}' is not available locally. Use 'ollama pull {self.model}' command to download the model."
            raise ConfigurationError(error_msg)

        self.system_prompt = config.system_prompts[self.model]
        self.history: List[Message] = []
        self.logger.info(f"OllamaLLMClient initialized. Using model: {self.model}")

    def _is_model_available(self) -> bool:
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.config.http_timeout)
            response.raise_for_status()
            available_models = [model["name"] for model in response.json()["models"]]
            return self.model in available_models
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error checking model availability (network/HTTP error): {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error checking model availability: {e}")
            return False

    @override
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

            response = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.config.http_timeout)
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

    @override
    def reset_history(self) -> None:
        self.history = []
        self.logger.info(f"Conversation history has been reset for model {self.model}")

    @override
    def close(self) -> None:
        if hasattr(self, 'session') and self.session:
            self.session.close()
            self.logger.debug(f"Closed HTTP session for model {self.model}")

    def __del__(self) -> None:
        # noinspection PyBroadException
        try:
            self.close()
        except Exception:
            pass


class JudgeBot:
    def __init__(self, config: AppConfig, logger: logging.Logger, llm_client: OllamaLLMClient):
        self.config = config
        self.logger = logger
        self.model = llm_client.model
        self.judge_client = llm_client
        self.judge_system_prompt = config.judge_system_prompt

        judge_format = config.judge_format
        self.user_question_prefix = judge_format.user_question_prefix
        self.model_responses_prefix = judge_format.model_responses_prefix
        self.model_response_format = judge_format.model_response_format
        self.response_format_header = judge_format.response_format_header
        self.response_format_points = judge_format.response_format_points

    def evaluate_responses(self, user_input: str, model_responses: List[ModelResponse]) -> ResponseResult:
        evaluation_prompt = self._create_evaluation_prompt(user_input, model_responses)
        return self.judge_client.send_message(evaluation_prompt)

    def _create_evaluation_prompt(self, user_input: str, model_responses: List[ModelResponse]) -> str:
        responses_text = "\n\n".join([self.model_response_format.format(model_name=resp.model_name, response_content=resp.response.content) for resp in model_responses])
        return f"{self.user_question_prefix} {user_input}\n\n{self.model_responses_prefix}\n{responses_text}\n\n{self.judge_system_prompt}\n{self.response_format_header}\n{self.response_format_points}"


class ChatBot:
    def __init__(self,
                 models: Dict[str, LLMClient],
                 judges: List[JudgeBot],
                 logger: logging.Logger,
                 config: ChatBotConfig):
        self.models = models
        self.judges = judges
        self.logger = logger
        self.config = config

    def start(self) -> None:
        commands_info = f"Type '{', '.join(self.config.exit_commands)}' to end. Type '{', '.join(self.config.reset_commands)}' to clear history."
        self.logger.info(f"Application started. {commands_info}")

        try:
            while True:
                try:
                    user_input = input("You: ")
                    if self._should_exit(user_input):
                        break
                    if self._should_reset(user_input):
                        for _, client in self.models.items():
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
        finally:
            self.logger.info("Closing all client sessions...")
            for client_name, client in self.models.items():
                try:
                    if hasattr(client, 'close'):
                        client.close()
                        self.logger.debug(f"Closed session for client {client_name}")
                except Exception as e:
                    self.logger.error(f"Error closing client {client_name}: {e}")

    def _should_exit(self, user_input: str) -> bool:
        return user_input.lower() in self.config.exit_commands

    def _should_reset(self, user_input: str) -> bool:
        return user_input.lower() in self.config.reset_commands

    def _process_user_input(self, user_input: str) -> None:
        self.logger.debug(f"Processing user input: {user_input[:50]}...")

        model_responses = []
        for model_name, client in self.models.items():
            response = client.send_message(user_input)
            if response.is_successful:
                model_responses.append(ModelResponse(model_name, response))
                print(f"\n[{model_name}]: {response.content}")
                self.logger.debug(f"Response from {model_name}: {response.content[:50]}...")
            else:
                self.logger.error(f"Error processing for {model_name}: {response.error}")
                print(f"Error [{model_name}]: {response.error}")

        if model_responses:
            for judge_bot in self.judges:
                judge_response = judge_bot.evaluate_responses(user_input, model_responses)
                if judge_response.is_successful:
                    print(f"\n[JUDGE {judge_bot.model}]: {judge_response.content}")
                    self.logger.debug(f"Judge {judge_bot.model} evaluation: {judge_response.content[:50]}...")
                else:
                    self.logger.error(f"Error in judge {judge_bot.model} evaluation: {judge_response.error}")
                    print(f"Error [JUDGE {judge_bot.model}]: {judge_response.error}")


def configure_logging(log_level: LogLevel = LogLevel.INFO) -> logging.Logger:
    logging.basicConfig(
        level=log_level.value,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def main():
    try:
        configure_logging()

        config_repository = YamlConfigRepository("config.yaml")
        app_config, yaml_config = config_repository.load_config()
        logger = configure_logging(app_config.log_level)

        chat_bot_config = config_repository.get_chat_bot_config(yaml_config)

        client_models = {}
        for model in app_config.models:
            client_config = AppConfig(
                base_url=app_config.base_url,
                model=model,
                system_prompts=app_config.system_prompts,
                judge_system_prompt=app_config.judge_system_prompt,
                log_level=app_config.log_level,
                http_timeout=app_config.http_timeout
            )
            client_models[model] = OllamaLLMClient(client_config, logger)

        judge_bots = []
        for judge_model in app_config.judge_models:
            judge_bots.append(JudgeBot(app_config, logger, client_models[judge_model]))

        chat_bot = ChatBot(
            models=client_models,
            judges=judge_bots,
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