# LLM Response Comparator

A command-line application for comparing responses from multiple large language models simultaneously. This tool connects to Ollama to interact with various models, displays their responses side-by-side, and uses a designated "judge" model to evaluate response quality.

The application loads configuration from environment variables, checks model availability, manages conversation history, and provides a simple interface for sending prompts and viewing results. It includes robust error handling for configuration issues, unavailable models, and API communication problems.

## Key Functionality

* Simultaneous querying of multiple LLM models
* Side-by-side response comparison
* Response evaluation by a judge model
* Conversation history management
* Support for thinking/reasoning extraction
* Flexible environment-based configuration

This tool enables efficient model performance comparison and helps identify strengths and weaknesses across different language models running on Ollama.

## Features

* **Multiple Model Interaction**: Communicate with several language models simultaneously.
* **Response Comparison**: View responses from different models side-by-side.
* **Response Evaluation**: Utilize a designated "judge" model to evaluate each response.
* **Conversation Management**: Reset conversation history or exit the application with simple commands.
* **Flexible Configuration**: Configure models, prompts, and system behavior via YAML.

## Requirements

* Python 3.8+
* Ollama (running locally or on a remote server)
* Required Python packages:

  * `requests`
  * `pyyaml`

## Installation

1. Clone this repository.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create or edit the `config.yaml` file (see Configuration section below).

4. Ensure Ollama is running and required models are available.

## Usage

Start the application:

```bash
python chat_bot.py
```

Interact by typing messages:

* Use `/reset` to clear conversation history.
* Use `/exit`, `/quit`, or `/bye` to end the session.

## Configuration

Configure the application through the `config.yaml` file:

### Core Settings

```yaml
ollama_url: http://localhost:11434    # URL to Ollama API
http_timeout: 60.0                    # HTTP request timeout in seconds
log_level: INFO                       # Logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL)
```

All settings are optional with sensible defaults.

### Model Configuration

Important:

* Use a colon `:` after the model name only if parameters follow.

Example:

```yaml
models:
  - model-name:latest                 # Basic model
  - another-model:14b:                # Model with parameters
      judge: true                     # Mark as judge (optional)
      system_prompt: "Custom prompt"  # Custom prompt (optional)
  - just-another-model:
      system_prompt:
        - "This is a custom system prompt"
        - "It can be a list of strings"
```

Each model supports:

* Model name with optional version
* `judge`: Boolean flag marking model as evaluator
* `system_prompt`: Custom system prompt (single string or list)

### Default System Prompt

Used unless overridden per model:

```yaml
default_system_prompt:
  - "Respond clearly, concisely, and professionally."
  - "Provide accurate, relevant, logically structured information."
  - "Maintain neutrality, avoid speculation, and clarify assumptions explicitly when needed."
```

`default_system_prompt` is optional and can be single or multiple strings.

### Commands Configuration

Define application commands and aliases:

```yaml
commands:
  - help:
      - /?
      - /help
  - exit:
      - /exit
      - /quit
      - /bye
  - reset:
      - /reset
  - prompts:
      - /prompt
      - /prompts
```

Commands are optional with default aliases provided.

### Judge Configuration

Configure evaluation of model responses:

```yaml
judge:
  system_prompt:
    - "You are an AI assistant responsible for evaluating responses."
    - "Your evaluation must be objective, precise, and consistent."

  user_question_prefix: "**User Question:**"
  model_responses_prefix: "**Collected Model Responses:**"

  model_response_format:
    - "### Model name: {model_name}"
    - "### Model response: {response_content}"

  response_format:
    - "Your evaluation must include:"
    - "- **Correctness**: Rating (0-100)"
    - "- **Quality**: Rating (0-100)"
    - "- **Assessment**: Brief justification"
```

All fields are optional and accept single or multiple strings, automatically joined with newlines.

File paths are relative to the application directory.

## How It Works

1. Initializes multiple LLM clients based on configuration.
2. Forwards user messages to all configured models.
3. Receives and displays responses side-by-side.
4. Judge model evaluates responses, providing an assessment.
5. Maintains conversation history per model unless reset.

## Key Components

* **OllamaLLMClient**: Handles API communication with Ollama.
* **JudgeBot**: Evaluates responses from other models.
* **ChatBot**: Manages conversation flow and user interactions.
* **ConfigRepository**: Loads configuration from YAML.

## Error Handling

Provides clear messages for common issues:

* Missing configuration
* Unavailable local models (with pull instructions)
* Network/API communication errors
