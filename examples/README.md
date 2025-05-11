# Experiments

## Environment

- I'm using an RTX3060
    - VRAM = 12GB
- My goal was to use only models that fit entirely in VRAM.
    - Larger models are not suitable because they perform too slowly.

## Content

- Each directory contains:
    - `config.yaml`
    - Conversation logs

## What is inside

### examples_1

- prompts:
    - Requesting a simple answer rather than a comprehensive explanation:
        - "Respond naturally and professionally"
- queries:
    - What is the capital of Poland?
    - What is the capital of San Escobar?
- models:
    - Three small models (approximately 1GB each)
    - Two larger models (~8GB)
- judges:
    - Two judges
    - Larger models were selected as judges
- results:
    - Smaller models often produce hallucinations

### examples_2

- prompts:
    - Requesting a comprehensive explanation:
        - "Provide a detailed and comprehensive response"
        - This required increasing the `http_timeout`
- queries:
    - What is the capital of Poland?
    - What is the capital of San Escobar?
- models:
    - Five larger models (~8GB)
- judges:
    - Five judges
    - Each model was selected as a judge
- results:
    - Please refer to [Capital_of_SanEscobar](`config_2.Capital_of_SanEscobar.txt`)

### examples_3

- same as `config_2`
- but with a new **updated, prompts**

### examples_4

- In reference to the article https://www.elektroda.pl/rtvforum/viewtopic.php?p=21546193#21546193
  - Title: "czy elektroda.pl to najlepszy portal dla elektroników?"
  - Prompt: Is elektroda.pl the best Polish portal for electronics enthusiasts?
    - Repeated 2 times
    - but LLMs are aware of previous answers.
  - As usual – the Bielik LLM has trouble understanding the concept of judging.
- config
  - with a new **updated, prompts**
  - same as `config_2`
  - models:
    - Five larger models (~8GB)
  - judges:
    - Five judges
    - Each model was selected as a judge
  - timeout: 5 minutes
