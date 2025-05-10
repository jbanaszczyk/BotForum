import logging
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict, fields
from enum import Enum
from typing import Protocol, Optional, Dict, List, Self, override

import requests
import typeguard
import yaml

DEFAULT_OLLAMA_URL: typing.Final[str] = "http://localhost:11434"
DEFAULT_HTTP_TIMEOUT: typing.Final[float] = 30.0


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
class JudgeConfig:
    system_prompt: str
    user_question_prefix: str
    model_responses_prefix: str
    model_response_format: str
    response_format: str


@dataclass
class ChatBotConfig:
    help_commands: List[str]
    exit_commands: List[str]
    reset_commands: List[str]
    prompts_commands: List[str]


@dataclass
class AppConfig:
    base_url: str = ""
    http_timeout: float = DEFAULT_HTTP_TIMEOUT
    log_level: LogLevel = LogLevel.INFO
    model: str = ""
    models: List[str] = field(default_factory=list)
    judges: List[str] = field(default_factory=list)
    system_prompts: Dict[str, str] = field(default_factory=dict)
    judge_config: JudgeConfig = field(default_factory=JudgeConfig)
    chat_bot_config: ChatBotConfig = field(default_factory=ChatBotConfig)


class ConfigRepository(Protocol):
    def load_config(self) -> AppConfig: ...


class YamlConfigRepository(ConfigRepository):
    def __init__(self, config_filename="config.yaml"):
        self.config_filename = config_filename
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def _str_or_list_as_str(value: typing.Union[str, List[str], None], default_value="") -> str:
        if isinstance(value, list):
            return "\n".join(value)
        if not value:
            return default_value
        return str(value)

    def _get_system_prompt(self, config: Dict, default_system_prompt: str) -> str:
        if not config or not isinstance(config, dict):
            return default_system_prompt
        return self._str_or_list_as_str(config.get("system_prompt"), default_system_prompt)

    def _validate_section_type(self, config: Dict, section_name: str, expected_section_type: typing.Type) -> None:
        section = config.get(section_name)
        if not section:
            raise ConfigurationError(f"Missing '{section_name}' in configuration file {self.config_filename}")
        try:
            typeguard.check_type(section, expected_section_type, collection_check_strategy=typeguard.CollectionCheckStrategy.ALL_ITEMS)
        except typeguard.TypeCheckError:
            raise ConfigurationError(f"Malformed '{section_name}' in configuration file {self.config_filename}")

    def _get_default_system_prompt(self, config: Dict) -> str:
        DEFAULT_SYSTEM_PROMPT: typing.Final[str] = "default_system_prompt"
        default_system_prompt = self._str_or_list_as_str(config.get(DEFAULT_SYSTEM_PROMPT))
        if not default_system_prompt:
            raise ConfigurationError(f"Missing {DEFAULT_SYSTEM_PROMPT} in configuration file {self.config_filename}")
        return default_system_prompt

    @staticmethod
    def _get_base_url(config: Dict) -> str:
        return config.get("ollama_url", DEFAULT_OLLAMA_URL)

    @staticmethod
    def _get_http_timeout(config: Dict) -> float:
        return float(config.get("http_timeout", DEFAULT_HTTP_TIMEOUT))

    @staticmethod
    def _get_log_level(config: Dict) -> LogLevel:
        log_level_str = config.get("log_level", "INFO").upper()
        try:
            return LogLevel[log_level_str]
        except KeyError:
            valid_levels = ", ".join([level.name for level in LogLevel])
            raise ConfigurationError(f"Invalid log_level '{log_level_str}'. Must be one of: {valid_levels}")

    def _get_models(self, models_section: Dict, default_system_prompt: str) -> typing.Tuple[List[str], List[str], Dict[str, str]]:
        SYSTEM_PROMPT: typing.Final[str] = "system_prompt"
        JUDGE: typing.Final[str] = "judge"
        models = []
        judges = []
        system_prompts = {}
        for model_entry in models_section:
            model_name, model_params = next(iter(model_entry.items())) \
                if isinstance(model_entry, dict) \
                else (model_entry, {SYSTEM_PROMPT: default_system_prompt, JUDGE: False})

            models.append(model_name)

            system_prompts[model_name] = self._str_or_list_as_str(model_params.get(SYSTEM_PROMPT), default_system_prompt)
            if model_params.get(JUDGE) is True:
                judges.append(model_name)

        return models, judges, system_prompts

    def _get_judge_config(self, judge_section: Dict, default_system_prompt) -> JudgeConfig:
        data = {}
        for a_field in fields(JudgeConfig):
            data[a_field.name] = self._str_or_list_as_str(judge_section.get(a_field.name))
        if not data.get("system_prompt"):
            data["system_prompt"] = default_system_prompt

        for a_field in fields(JudgeConfig):
            if not data[a_field.name]:
                raise ConfigurationError(f"Missing '{a_field.name}' in judge configuration")

        return JudgeConfig(**data)

    @staticmethod
    def _get_chat_bot_config(commands_section: Dict) -> ChatBotConfig:

        COMMANDS_SUFFIX: typing.Final[str] = "_commands"

        data = {}
        for command in commands_section:
            command_name, command_values = next(iter(command.items()))
            data[command_name + COMMANDS_SUFFIX] = command_values

        for a_field in fields(ChatBotConfig):
            if not data.get(a_field.name):
                raise ConfigurationError(f"Missing '{a_field.name.removesuffix(COMMANDS_SUFFIX)}' in commands configuration")

        return ChatBotConfig(**data)

    @override
    def load_config(self) -> AppConfig:
        try:
            with open(self.config_filename, 'r') as file:
                config = yaml.safe_load(file)

            if not isinstance(config, dict):
                raise ConfigurationError(f"Configuration file {self.config_filename} is empty or invalid")

            expected_section_types = (
                ("commands", typing.List[typing.Dict[str, typing.List[str]]]),
                ("models", typing.List),
                ("judge", typing.Dict[str, str | typing.List[str]]),
                ("default_system_prompt", str | typing.List[str]),
            )

            for section_name, section_type in expected_section_types:
                self._validate_section_type(config, section_name, section_type)

            default_system_prompt = self._get_default_system_prompt(config)

            chat_bot_config = self._get_chat_bot_config(config.get("commands"))
            models, judges, system_prompts = self._get_models(config.get("models"), default_system_prompt)
            judge_config = self._get_judge_config(config.get("judge"), default_system_prompt)

            app_config = AppConfig(
                base_url=self._get_base_url(config),
                http_timeout=self._get_http_timeout(config),
                log_level=self._get_log_level(config),
                models=models,
                judges=judges,
                system_prompts=system_prompts,
                judge_config=judge_config,
                chat_bot_config=chat_bot_config,
            )

            return app_config

        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file '{self.config_filename}' not found")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML configuration: {e}")
        except ConfigurationError as e:
            self.logger.error(f"Configuration error: {e}", exc_info=True)
            raise
        except Exception as e:
            raise ConfigurationError(f"Configuration loading failed: {e}")


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
        self.system_prompt = None

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
        self.system_prompt = config.system_prompts[self.model]
        self.history: List[Message] = []

        if not self._is_model_available():
            error_msg = f"Model '{self.model}' is not available locally. Use 'ollama pull {self.model}' command to download the model."
            raise ConfigurationError(error_msg)

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
        self.judge_config = config.judge_config

    def evaluate_responses(self, user_input: str, model_responses: List[ModelResponse]) -> ResponseResult:
        evaluation_prompt = self._create_evaluation_prompt(user_input, model_responses)
        return self.judge_client.send_message(evaluation_prompt)

    def _create_evaluation_prompt(self, user_input: str, model_responses: List[ModelResponse]) -> str:
        judge_config = self.judge_config
        responses_text = "\n\n".join([judge_config.model_response_format.format(model_name=response.model_name, response_content=response.response.content) for response in model_responses])
        return (
            f"{judge_config.user_question_prefix}\n\n"
            f"{user_input}\n\n"
            f"{judge_config.model_responses_prefix}\n"
            f"{responses_text}\n\n"
            f"{judge_config.system_prompt}\n"
            f"{judge_config.response_format}\n"
        )


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
        self.logger.info(f"Application started.")
        print(f"Type '{', '.join(self.config.exit_commands)}' to end.")
        print(f"Type '{', '.join(self.config.reset_commands)}' to clear history.")
        print(f"Type '{', '.join(self.config.prompts_commands)}' to show system prompts.")

        # noinspection PyUnreachableCode
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

                    if self._should_show_prompts(user_input):
                        for model_name, client in self.models.items():
                            print(f"[{model_name} system prompt]: {client.system_prompt}")
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
        return user_input.strip() in self.config.exit_commands

    def _should_reset(self, user_input: str) -> bool:
        return user_input.strip() in self.config.reset_commands

    def _should_show_prompts(self, user_input: str) -> bool:
        return user_input.strip() in self.config.prompts_commands

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
    # noinspection SpellCheckingInspection
    logging.basicConfig(
        level=log_level.value,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def main():
    try:
        configure_logging()

        config_repository = YamlConfigRepository("config.yaml")
        app_config = config_repository.load_config()
        logger = configure_logging(app_config.log_level)

        client_models = {}
        for model in app_config.models:
            client_config = AppConfig(
                base_url=app_config.base_url,
                http_timeout=app_config.http_timeout,
                log_level=app_config.log_level,
                model=model,
                system_prompts=app_config.system_prompts,
                judge_config=app_config.judge_config,
                chat_bot_config=app_config.chat_bot_config,
            )
            client_models[model] = OllamaLLMClient(client_config, logger)
            logger.info(f"OllamaLLMClient initialized. Using model: {model}")

        judge_bots = []
        for judge_model in app_config.judges:
            judge_bots.append(JudgeBot(app_config, logger, client_models[judge_model]))

        chat_bot = ChatBot(
            models=client_models,
            judges=judge_bots,
            logger=logger,
            config=app_config.chat_bot_config
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
