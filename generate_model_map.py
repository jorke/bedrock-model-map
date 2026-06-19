#!/usr/bin/env python3
"""
Generate a mapping of all Bedrock foundation models with capabilities,
sourced from AWS documentation model cards.

Scrapes individual model card pages for regions, APIs, endpoints, and inference IDs.
Output: bedrock_models.json and bedrock_models.yaml
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from urllib.request import urlopen, Request

import yaml

BASE_URL = "https://docs.aws.amazon.com/bedrock/latest/userguide"

# All model card slugs from https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html
MODEL_CARD_SLUGS = [
    # AI21 Labs
    "model-card-ai21-labs-jamba-1-5-large",
    "model-card-ai21-labs-jamba-1-5-mini",
    # Amazon
    "model-card-amazon-amazon-nova-multimodal-embeddings",
    "model-card-amazon-nova-2-lite",
    "model-card-amazon-nova-2-sonic",
    "model-card-amazon-nova-canvas",
    "model-card-amazon-nova-lite",
    "model-card-amazon-nova-micro",
    "model-card-amazon-nova-premier",
    "model-card-amazon-nova-pro",
    "model-card-amazon-nova-reel",
    "model-card-amazon-nova-sonic",
    "model-card-amazon-titan-embeddings-g1---text",
    "model-card-amazon-titan-image-generator-g1-v2",
    "model-card-amazon-titan-multimodal-embeddings-g1",
    "model-card-amazon-titan-text-embeddings-v2",
    "model-card-amazon-titan-text-embeddings-v2-2",
    # Anthropic
    "model-card-anthropic-claude-mythos-5",
    "model-card-anthropic-claude-fable-5",
    "model-card-anthropic-claude-opus-4-8",
    "model-card-anthropic-claude-opus-4-7",
    "model-card-anthropic-claude-opus-4-6",
    "model-card-anthropic-claude-sonnet-4-6",
    "model-card-anthropic-claude-haiku-4-5",
    "model-card-anthropic-claude-opus-4-5",
    "model-card-anthropic-claude-sonnet-4-5",
    "model-card-anthropic-claude-sonnet-4",
    "model-card-anthropic-claude-opus-4-1",
    "model-card-anthropic-claude-3-5-haiku",
    "model-card-anthropic-claude-3-haiku",
    "model-card-anthropic-claude-mythos-preview",
    # Cohere
    "model-card-cohere-rerank-3-5",
    "model-card-cohere-command-r",
    "model-card-cohere-command-r-plus",
    "model-card-cohere-embed-english",
    "model-card-cohere-embed-multilingual",
    "model-card-cohere-embed-v4",
    # DeepSeek
    "model-card-deepseek-deepseek-v3-2",
    "model-card-deepseek-deepseek-v3-1",
    "model-card-deepseek-deepseek-r1",
    # Google
    "model-card-google-gemma-4-31b",
    "model-card-google-gemma-4-26b-a4b",
    "model-card-google-gemma-4-e2b",
    "model-card-google-gemma-3-12b-it",
    "model-card-google-gemma-3-27b-pt",
    "model-card-google-gemma-3-4b-it",
    # Meta
    "model-card-meta-llama-3-3-70b-instruct",
    "model-card-meta-llama-3-2-11b-instruct",
    "model-card-meta-llama-3-2-1b-instruct",
    "model-card-meta-llama-3-2-3b-instruct",
    "model-card-meta-llama-3-2-90b-instruct",
    "model-card-meta-llama-3-1-405b-instruct",
    "model-card-meta-llama-3-1-70b-instruct",
    "model-card-meta-llama-3-1-8b-instruct",
    "model-card-meta-llama-3-70b-instruct",
    "model-card-meta-llama-3-8b-instruct",
    "model-card-meta-llama-4-maverick-17b-instruct",
    "model-card-meta-llama-4-scout-17b-instruct",
    # MiniMax
    "model-card-minimax-minimax-m2-5",
    "model-card-minimax-minimax-m2-1",
    "model-card-minimax-minimax-m2",
    # Mistral AI
    "model-card-mistral-ai-ministral-14b-3-0",
    "model-card-mistral-ai-devstral-2-123b",
    "model-card-mistral-ai-magistral-small-2509",
    "model-card-mistral-ai-ministral-3-8b",
    "model-card-mistral-ai-ministral-3b",
    "model-card-mistral-ai-mistral-7b-instruct",
    "model-card-mistral-ai-mistral-large",
    "model-card-mistral-ai-mistral-large-3",
    "model-card-mistral-ai-mistral-small",
    "model-card-mistral-ai-mixtral-8x7b-instruct",
    "model-card-mistral-ai-pixtral-large",
    "model-card-mistral-ai-voxtral-mini-3b-2507",
    "model-card-mistral-ai-voxtral-small-24b-2507",
    # Moonshot AI
    "model-card-moonshot-ai-kimi-k2-5",
    "model-card-moonshot-ai-kimi-k2-thinking",
    # NVIDIA
    "model-card-nvidia-nvidia-nemotron-nano-12b-v2-vl-bf16",
    "model-card-nvidia-nvidia-nemotron-nano-9b-v2",
    "model-card-nvidia-nemotron-nano-3-30b",
    "model-card-nvidia-nemotron-super-3-120b",
    # OpenAI
    "model-card-openai-gpt-55",
    "model-card-openai-gpt-54",
    "model-card-openai-gpt-oss-safeguard-120b",
    "model-card-openai-gpt-oss-safeguard-20b",
    "model-card-openai-gpt-oss-120b",
    "model-card-openai-gpt-oss-20b",
    # Qwen
    "model-card-qwen-qwen3-235b-a22b-2507",
    "model-card-qwen-qwen3-32b",
    "model-card-qwen-qwen3-coder-480b-a35b-instruct",
    "model-card-qwen-qwen3-coder-next",
    "model-card-qwen-qwen3-next-80b-a3b",
    "model-card-qwen-qwen3-vl-235b-a22b",
    "model-card-qwen-qwen3-coder-30b-a3b-instruct",
    # Stability AI
    "model-card-stability-ai-stable-image-conservative-upscale",
    "model-card-stability-ai-stable-image-control-sketch",
    "model-card-stability-ai-stable-image-control-structure",
    "model-card-stability-ai-stable-image-creative-upscale",
    "model-card-stability-ai-stable-image-erase-object",
    "model-card-stability-ai-stable-image-fast-upscale",
    "model-card-stability-ai-stable-image-inpaint",
    "model-card-stability-ai-stable-image-outpaint",
    "model-card-stability-ai-stable-image-remove-background",
    "model-card-stability-ai-stable-image-search-and-recolor",
    "model-card-stability-ai-stable-image-search-and-replace",
    "model-card-stability-ai-stable-image-style-guide",
    "model-card-stability-ai-stable-image-style-transfer",
    # TwelveLabs
    "model-card-twelvelabs-marengo-embed-3-0",
    "model-card-twelvelabs-marengo-embed-v2-7",
    "model-card-twelvelabs-pegasus-v1-2",
    # Writer
    "model-card-writer-palmyra-x4",
    "model-card-writer-palmyra-x5",
    "model-card-writer-palmyra-vision-7b",
    # xAI
    "model-card-xai-grok-4-3",
    # Z.AI
    "model-card-zai-glm-4-7",
    "model-card-zai-glm-4-7-flash",
    "model-card-zai-glm-5",
]


def fetch_page(url):
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (bedrock-model-map)"})
    with urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def strip_tags(s):
    return re.sub(r"<[^>]+>", "", s).strip()


def split_sections(html):
    """Split HTML into sections by <h2> headings. Returns dict of title -> content."""
    h2_pattern = re.compile(r"<h2[^>]*>(.*?)</h2>", re.DOTALL)
    matches = list(h2_pattern.finditer(html))
    sections = {}
    for i, m in enumerate(matches):
        title = strip_tags(m.group(1)).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(html)
        sections[title] = html[start:end]
    return sections


def parse_model_card(html):
    """Parse a model card HTML page into structured data."""
    model = {}
    sections = split_sections(html)

    # Provider and model name from first h2 (e.g. "Anthropic — Claude Sonnet 4.6")
    first_h2 = re.search(r"<h2[^>]*>(.*?)</h2>", html, re.DOTALL)
    if first_h2:
        header_text = strip_tags(first_h2.group(1))
        if "—" in header_text:
            parts = header_text.split("—", 1)
            model["provider"] = parts[0].strip()
            model["modelName"] = parts[1].strip()
        elif "—" in header_text:
            parts = header_text.split("—", 1)
            model["provider"] = parts[0].strip()
            model["modelName"] = parts[1].strip()
        else:
            model["modelName"] = header_text

    # Model details from <li> items
    details_html = sections.get("Model Details", "")
    li_items = re.findall(r"<li[^>]*>(.*?)</li>", details_html, re.DOTALL)
    for li in li_items:
        text = strip_tags(li)
        for field, key in [
            ("Model launch date:", "launchDate"),
            ("Model lifecycle:", "modelLifecycle"),
            ("Context window:", "contextWindow"),
            ("Max output tokens:", "maxOutputTokens"),
            ("Knowledge cutoff:", "knowledgeCutoff"),
            ("Reasoning:", "reasoning"),
        ]:
            if text.startswith(field):
                val = text[len(field) :].strip()
                if key == "reasoning":
                    model[key] = "supported" in val.lower()
                else:
                    model[key] = val
                break

    # Modalities / APIs / Endpoints from first table in Model Details
    tables_in_details = re.findall(
        r"<table[^>]*>(.*?)</table>", details_html, re.DOTALL
    )
    if tables_in_details:
        _parse_modalities_table(tables_in_details[0], model)

    # Programmatic Access — model IDs and inference IDs
    prog_html = sections.get("Programmatic Access", "")
    if prog_html:
        _parse_programmatic_access(prog_html, model)

    # Service Tiers
    tiers_html = sections.get("Service Tiers", "")
    if tiers_html:
        _parse_service_tiers(tiers_html, model)

    # Regional Availability
    regional_html = sections.get("Regional Availability", "")
    if regional_html:
        _parse_regional_availability(regional_html, model)

    return {k: v for k, v in model.items() if v is not None and v != [] and v != {}}


def _parse_modalities_table(table_html, model):
    """Parse the modalities/APIs/endpoints table."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.DOTALL)
    input_mods = []
    output_mods = []
    apis = []
    endpoints = []

    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(cells) >= 4:
            in_text = strip_tags(cells[0])
            out_text = strip_tags(cells[1])
            api_text = strip_tags(cells[2])
            ep_text = strip_tags(cells[3])

            if in_text and "icon-yes" in cells[0]:
                input_mods.append(in_text.upper())
            if out_text and "icon-yes" in cells[1]:
                output_mods.append(out_text.upper())
            if api_text and "icon-yes" in cells[2]:
                apis.append(api_text)
            if ep_text and "icon-yes" in cells[3]:
                endpoints.append(ep_text)
        elif len(cells) == 3:
            in_text = strip_tags(cells[0])
            out_text = strip_tags(cells[1])
            api_text = strip_tags(cells[2])

            if in_text and "icon-yes" in cells[0]:
                input_mods.append(in_text.upper())
            if out_text and "icon-yes" in cells[1]:
                output_mods.append(out_text.upper())
            if api_text and "icon-yes" in cells[2]:
                apis.append(api_text)

    model["inputModalities"] = input_mods
    model["outputModalities"] = output_mods
    model["apisSupported"] = apis
    model["endpointsSupported"] = endpoints


def _cell_codes(cell):
    """Return list of <code> contents in a cell, or the plain text if none.

    Filters out empty values and "N/A" placeholders.
    """
    code_blocks = re.findall(r"<code[^>]*>(.*?)</code>", cell, re.DOTALL)
    values = (
        [strip_tags(cb) for cb in code_blocks] if code_blocks else [strip_tags(cell)]
    )
    cleaned = []
    for v in values:
        v = v.strip()
        if not v:
            continue
        low = v.lower()
        if low in ("n/a", "none", "-", "—") or low.startswith("not supported"):
            continue
        cleaned.append(v)
    return cleaned


def _map_columns(header_row):
    """Map programmatic-access column headers to logical field names by position."""
    headers = re.findall(r"<th[^>]*>(.*?)</th>", header_row, re.DOTALL)
    col_map = {}
    for idx, h in enumerate(headers):
        title = strip_tags(h).lower()
        if "global" in title:
            col_map["global"] = idx
        elif "geo" in title:
            col_map["geo"] = idx
        elif "url" in title:
            col_map["url"] = idx
        elif "model id" in title:
            col_map["modelId"] = idx
        elif "endpoint" in title:
            col_map["endpoint"] = idx
    return col_map


def _parse_programmatic_access(html, model):
    """Extract per-endpoint model IDs, URLs and inference IDs.

    The programmatic access table has one row per endpoint, so each model ID,
    endpoint URL and inference ID is associated with the endpoint in its row.
    """
    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
    if not tables:
        return

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tables[0], re.DOTALL)

    # Locate the header row and map columns to logical fields.
    col_map = {}
    for row in rows:
        if "<th" in row:
            col_map = _map_columns(row)
            break

    def cell(cells, key):
        idx = col_map.get(key)
        if idx is None or idx >= len(cells):
            return []
        return _cell_codes(cells[idx])

    access = []
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if not cells:
            continue

        endpoint = cell(cells, "endpoint")
        model_id = cell(cells, "modelId")
        url = cell(cells, "url")
        geo_ids = cell(cells, "geo")
        global_ids = cell(cells, "global")

        if not endpoint and not model_id:
            continue

        entry = {}
        if endpoint:
            entry["endpoint"] = endpoint[0]
        if model_id:
            entry["modelId"] = model_id[0]
        if url:
            entry["endpointUrl"] = url[0]
        if geo_ids:
            entry["geoInferenceIds"] = geo_ids
        if global_ids:
            entry["globalInferenceIds"] = global_ids
        if entry:
            access.append(entry)

    if access:
        model["programmaticAccess"] = access


def _parse_service_tiers(html, model):
    """Parse service tiers table."""
    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
    if not tables:
        return

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tables[0], re.DOTALL)
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(cells) >= 4:
            model["serviceTiers"] = {
                "standard": "icon-yes" in cells[0],
                "priority": "icon-yes" in cells[1],
                "flex": "icon-yes" in cells[2],
                "reserved": "icon-yes" in cells[3],
            }
            break


def _parse_regional_availability(html, model):
    """Parse regional availability table."""
    tables = re.findall(r"<table[^>]*>(.*?)</table>", html, re.DOTALL)
    if not tables:
        return

    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", tables[0], re.DOTALL)
    regions = {}
    for row in rows:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.DOTALL)
        if len(cells) >= 4:
            region_text = strip_tags(cells[0])
            region_match = re.search(r"([a-z]{2}-[a-z]+-\d)", region_text)
            if region_match:
                region_code = region_match.group(1)
                regions[region_code] = {
                    "inRegion": "icon-yes" in cells[1],
                    "geo": "icon-yes" in cells[2],
                    "global": "icon-yes" in cells[3],
                }

    if regions:
        model["regions"] = regions


def scrape_model_card(slug):
    url = f"{BASE_URL}/{slug}.html"
    try:
        html = fetch_page(url)
        return parse_model_card(html)
    except Exception as e:
        print(f"  Error: {slug} - {e}")
        return None


def main():
    print(f"Scraping {len(MODEL_CARD_SLUGS)} model cards from AWS docs...")

    models = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(scrape_model_card, slug): slug for slug in MODEL_CARD_SLUGS
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result:
                models.append(result)
            if i % 20 == 0:
                print(f"  {i}/{len(MODEL_CARD_SLUGS)} done...")

    models.sort(key=lambda x: (x.get("provider", ""), x.get("modelName", "")))

    data = {
        "source": "https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modelCount": len(models),
        "models": models,
    }

    with open("bedrock_models.json", "w") as f:
        json.dump(data, f, indent=2)

    with open("bedrock_models.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(
        f"\nWrote {len(models)} models to bedrock_models.json and bedrock_models.yaml"
    )


if __name__ == "__main__":
    main()
