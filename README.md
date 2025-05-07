A command-line application for comparing responses from multiple large language models simultaneously. This tool connects to Ollama to interact with various models, displays their responses side-by-side, and uses a designated "judge" model to evaluate response quality.
The application loads configuration from environment variables, checks for model availability, manages conversation history, and provides a simple interface for sending prompts and viewing results. It includes robust error handling for configuration issues, unavailable models, and API communication problems.
Key functionality includes:
- Simultaneous querying of multiple LLM models
- Side-by-side response comparison
- Response evaluation by a judge model
- Conversation history management
- Support for thinking/reasoning extraction
- Flexible environment-based configuration

This tool enables efficient model performance comparison and helps identify strengths and weaknesses across different language models running on Ollama.

## Features

- **Multiple Model Interaction**: Communicate with several language models at once
- **Response Comparison**: Compare responses from different models side-by-side
- **Response Evaluation**: A designated "judge" model evaluates each response
- **Conversation Management**: Reset conversation history or exit the application with simple commands
- **Flexible Configuration**: Configure models, prompts, and system behavior through YAML configuration

## Requirements

- Python 3.8+
- Ollama (running locally or on a remote server)
- Required Python packages:
    - requests
    - pyyaml

## Installation

1. Clone this repository
2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Create/edit a `.env` file with your configuration (see Configuration section)
4. Make sure Ollama is running and the required models are available

## Usage

1. Start the application:
```
python chat_bot.py
```

2. Interact with the chatbot by typing messages
3. Use `reset` to clear conversation history
4. Use `exit`, `quit`, or `bye` to end the session

## Configuration

The application is configurable through the `config.yaml` file:

### Core Configuration
```yaml
ollama_url: http://localhost:11434
log_level: INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL

## How It Works

1. The application initializes multiple LLM clients based on the configured models
2. When a user sends a message, it's forwarded to all configured models
3. Each model processes the message and provides a response
4. All responses are displayed to the user
5. The judge model evaluates the responses and provides its assessment
6. The conversation history is maintained for each model unless reset

## Key Components

- **OllamaLLMClient**: Handles communication with Ollama API for each model
- **JudgeBot**: Specializes in evaluating responses from other models
- **ChatBot**: Coordinates the conversation flow and user interaction
- **ConfigRepository**: Manages application configuration from YAML file

## Error Handling

The application provides clear error messages for common issues:
- Missing configuration variables
- Models not available locally (with instructions to pull them)
- Network/API communication errors
