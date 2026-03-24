# NLP Assignment: Supply Chain Network from News

## Overview
This project builds a supply chain network for a selected company using information extracted from recent online news.

The current implementation is configured for Tesla and follows three required stages:
1. News collection from web sources.
2. Information extraction using Named Entity Recognition (NER) and relationship extraction.
3. Network construction and visualization.

## Assignment Objective
Construct a supply chain network from news data and produce:
1. A list of collected articles.
2. Extracted entities and relationships.
3. A graph that visualizes supply chain connections.

## Repository Structure
- supply_chain_network.py: End-to-end pipeline (collection, extraction, graph creation).
- output/: Generated outputs after running the pipeline.

## Methodology
### 1) News Collection
The script collects recent Tesla supply-chain-related articles from Google News RSS search feeds using multiple query variations (supply chain, suppliers, battery deals, manufacturing, logistics, and materials).

### 2) Information Extraction
The script uses spaCy (en_core_web_sm) to extract named entities from article text.

Entity types retained:
- ORG (organizations)
- GPE (countries, cities, states)
- LOC (locations)
- FAC (facilities)
- PRODUCT (products/components)

Then it applies rule-based relationship extraction using keyword patterns to label edges such as:
- supplies
- partners_with
- manufactures_in
- materials_dependency
- logistics_support

### 3) Network Construction
Extracted entities become nodes, and extracted relationships become edges in a NetworkX graph.
The graph is rendered with matplotlib and saved as an image.

## Setup
### Requirements
- Python 3.10+
- pip

### Install Dependencies
```bash
pip install feedparser matplotlib networkx pandas spacy
python -m spacy download en_core_web_sm
```

## Run
From the project root:
```bash
python supply_chain_network.py
```

## Output Files
After execution, these files are generated inside output/:
- news_articles.csv: Full list of collected news articles.
- publisher_audit.csv: Publisher-level article counts.
- extracted_entities.csv: Unique entities with mention counts.
- entity_mentions.csv: Entity mentions at row level by article.
- extracted_relationships.csv: Extracted source-target relationships and evidence text.
- supply_chain_network.png: Final network graph visualization.

## How to Interpret the Graph
- Node: Entity found in news (company, location, facility, product, etc.).
- Edge: Extracted relationship between two entities.
- Central nodes typically indicate frequently mentioned organizations or dependencies.

## Customization
You can modify these constants in supply_chain_network.py:
- COMPANY: Target company name.
- RSS_URLS: Query list for article collection.
- MAX_ARTICLES: Number of articles to collect.
- RECENT_DAYS: Recency window.

You can also refine:
- NOISY_ENTITY_TERMS and NOISY_ENTITY_SUBSTRINGS for filtering.
- RELATION_KEYWORDS for relationship categories.

## Reproducibility Notes
- Results may change over time because RSS feeds are live and continuously updated.
- Relationship extraction is heuristic and pattern-based; it is intended for assignment-scale analysis.

## Troubleshooting
- If no articles are collected: increase RECENT_DAYS or broaden RSS queries.
- If spaCy model error appears: run python -m spacy download en_core_web_sm.
- If graph appears cluttered: tighten entity filters in NOISY_ENTITY_TERMS/NOISY_ENTITY_SUBSTRINGS.