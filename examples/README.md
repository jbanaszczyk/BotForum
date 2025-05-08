# Experiments

## Environment

- I'm using RTX3060
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
    - what is the capital of Poland?
    - what is the capital of SanEscobar?
- models
    - Three tiny models (approximately 1GB each)
    - Two larger models (~8GB)
- judges
    - Two judges
    - Larger models were selected as judges
- results
    - Smaller models often produce hallucinations

### config_2

- prompts:
    - Requesting comprehensive explanation:
        - "Provide detailed and comprehensive response"
        - it forced to increase `http_timeut`
- queries:
    - what is the capital of Poland?
    - what is the capital of SanEscobar?
- models
    - Five larger models (~8GB)
- judges
    - Five judges
    - Each model is selected as judge
- results
    - please note [Capital_of_SanEscobar](`config_2.Capital_of_SanEscobar.txt`)
