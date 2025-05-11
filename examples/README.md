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

### config_1

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

### config_2

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

### config_3

- same as `config_2`
- but with a new updated, prompts
