# bedrock-model-map

Scrapes AWS Bedrock model cards and generates a capability map (regions, APIs, per-endpoint model IDs, inference IDs) to `bedrock_models.json` and `bedrock_models.yaml`.

## Usage

```sh
uv run generate_model_map.py
```

Source: https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
