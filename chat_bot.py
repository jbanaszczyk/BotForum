import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Protocol, Optional, Dict, List, Self

import requests
from dotenv import load_dotenv


class ConfigurationError(Exception):
    pass


class MissingEnvVariableError(ConfigurationError):
    def __init__(self, variable_name: str):
        super().__init__(f"{variable_name} must be set in environment variables")
        self.variable_name = variable_name


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


class EnvConfigRepository(ConfigRepository):
    def __init__(self):
        pass

    @staticmethod
    def _get_required_env_var(var_name: str) -> str:
        value = os.getenv(var_name)
        if not value:
            raise MissingEnvVariableError(var_name)
        return value

    def load_config(self) -> AppConfig:
        try:
            load_dotenv()
            judge_model = self._get_judge_model()
            models_to_compare = self._get_models_to_compare()
            return AppConfig(
                base_url=self._get_base_url(),
                judge_model=judge_model,
                models_to_compare=models_to_compare,
                system_prompts=self._get_system_prompts(judge_model, models_to_compare),
                judge_system_prompt=self._get_judge_system_prompt(),
                log_level=self._get_log_level()
            )
        except ConfigurationError:
            raise
        except Exception as e:
            raise ConfigurationError(f"Configuration loading failed: {e}")

    @staticmethod
    def _get_base_url() -> str:
        return os.getenv("BASE_URL", "http://localhost:11434")

    def _get_judge_model(self) -> str:
        return self._get_required_env_var("JUDGE_MODEL")

    def _get_models_to_compare(self) -> List[str]:
        models_str = self._get_required_env_var("MODELS")
        return [m.strip() for m in models_str.split(",") if m.strip()]

    def _get_system_prompts(self, judge_model: str, models_to_compare: List[str]) -> Dict[str, str]:
        prompts = {}
        default_prompt = self._get_required_env_var("DEFAULT_SYSTEM_PROMPT")

        all_models = [judge_model] + models_to_compare
        for model in all_models:
            if model not in prompts:
                model_key = model.split(":")[0].upper()
                specific_model_prompt = os.getenv(f"{model_key}_SYSTEM_PROMPT")
                if not specific_model_prompt:
                    specific_model_prompt = default_prompt
                    logging.warning(f"system prompt: {model_key}_SYSTEM_PROMPT not found, using default")
                prompts[model] = specific_model_prompt
        return prompts

    def _get_judge_system_prompt(self) -> str:
        return self._get_required_env_var("JUDGE_SYSTEM_PROMPT")

    @staticmethod
    def _get_log_level() -> LogLevel:
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        try:
            return LogLevel[log_level_str]
        except KeyError:
            valid_levels = ", ".join([level.name for level in LogLevel])
            raise ConfigurationError(
                f"Invalid LOG_LEVEL '{log_level_str}'. Must be one of: {valid_levels}"
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
        self.user_question_prefix = os.getenv('JUDGE_USER_QUESTION_PREFIX')
        self.model_responses_prefix = os.getenv('JUDGE_MODEL_RESPONSES_PREFIX')
        self.model_response_format = os.getenv('JUDGE_MODEL_RESPONSE_FORMAT')
        self.response_format_header = os.getenv('JUDGE_RESPONSE_FORMAT_HEADER')
        self.response_format_points = os.getenv('JUDGE_RESPONSE_FORMAT_POINTS')

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
        config_repository = EnvConfigRepository()
        app_config = config_repository.load_config()
        logger = configure_logging(app_config.log_level)

        chat_bot_config = ChatBotConfig(
            exit_commands=os.getenv('EXIT_COMMANDS', 'exit,quit,bye').split(','),
            reset_command=os.getenv('RESET_COMMAND', 'reset')
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

    except MissingEnvVariableError as e:
        print(f"Configuration error: Missing environment variable - {e.variable_name}. Message: {e}")
        logging.basicConfig(level=logging.ERROR)
        logging.error(f"Configuration error: {e}", exc_info=False)
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
