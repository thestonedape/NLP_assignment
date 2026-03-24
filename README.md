# NLP Assignment — Supply Chain Network

This repository contains a Python script that builds a simple **supply chain relationship network** from recent news articles.

## Project files
- `supply_chain_network.py` — Collects recent news via Google News RSS, extracts entities/relationships with spaCy, and builds a NetworkX graph.
- `output/` — Output directory (generated) for CSV files and the rendered graph image.

## What the script does
1. Collects recent news articles related to **Tesla** and supply-chain keywords using RSS.
2. Cleans article text and extracts named entities (ORG/GPE/LOC/FAC/PRODUCT).
3. Heuristically detects relationship types based on keywords.
4. Builds a NetworkX graph and saves outputs to `output/`.

## Setup
### Requirements
Install dependencies (example):
```bash
pip install feedparser matplotlib networkx pandas spacy
python -m spacy download en_core_web_sm
```

## Run
```bash
python supply_chain_network.py
```

## Outputs
After running, the following files are created in `output/`:
- `news_articles.csv`
- `publisher_audit.csv`
- `extracted_entities.csv`
- `entity_mentions.csv`
- `extracted_relationships.csv`
- `supply_chain_network.png`

## Notes
- The company keyword is currently set to `Tesla` in `supply_chain_network.py` (`COMPANY` constant).
- If no articles are found, try adjusting the RSS queries or increasing `RECENT_DAYS`.