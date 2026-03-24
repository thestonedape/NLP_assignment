from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import feedparser
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import spacy


COMPANY = "Tesla"
RSS_URLS = [
    "https://news.google.com/rss/search?q=Tesla+supply+chain&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Tesla+suppliers&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Tesla+battery+deal&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Tesla+manufacturing+plant&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Tesla+semiconductor+supply+chain&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Tesla+lithium+nickel+supplier&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Tesla+logistics+partner&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Tesla+gigafactory+supply+chain&hl=en-US&gl=US&ceid=US:en",
]
MAX_ARTICLES = 80
RECENT_DAYS = 365
OUTPUT_DIR = Path("output")

NOISY_ENTITY_TERMS = {
    "supply chain",
    "supply-chain",
    "supply-chain disruptions strike",
    "report",
    "insight",
    "news",
    "fleet",
    "picks suppliers",
    "tesla picks suppliers",
    "disruptions",
    "first time",
    "model y",
    "ai",
}

NOISY_ENTITY_SUBSTRINGS = {
    "picks suppliers",
    "disruptions strike",
    "supply chain crisis",
    "shipments rise",
    "talk supply chain",
    "open india showrooms",
    "deal faces",
    "showrooms",
    "opportunities",
    "tesla supply chain",
}

RELATION_KEYWORDS = {
    "supplies": ["supply", "supplier", "supplies", "sourcing", "source", "procure"],
    "partners_with": ["partner", "partnership", "collaborat", "joint venture", "deal", "agreement"],
    "manufactures_in": ["factory", "plant", "manufactur", "production", "assembly", "gigafactory", "build"],
    "materials_dependency": ["lithium", "nickel", "battery", "chip", "component", "rare earth", "semiconductor"],
    "logistics_support": ["logistics", "delivery", "freight", "transport", "fleet"],
}


@dataclass
class Article:
    title: str
    link: str
    published: str
    source: str
    summary: str


def strip_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_source(title: str) -> str:
    if " - " in title:
        return title.rsplit(" - ", 1)[-1].strip()
    return "Unknown"


def clean_title_for_nlp(title: str) -> str:
    if " - " in title:
        title = title.rsplit(" - ", 1)[0]
    return title.strip()


def normalize_entity(text: str) -> str:
    t = text.strip().replace("’", "'")
    t = re.sub(r"^the\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t)
    return t.strip(" .,:;|-_")


def to_iso_date(struct_time) -> str:
    if not struct_time:
        return ""
    dt = datetime(*struct_time[:6], tzinfo=timezone.utc)
    return dt.date().isoformat()


def collect_news() -> list[Article]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)

    collected: list[Article] = []
    seen_titles: set[str] = set()
    seen_links: set[str] = set()

    for url in RSS_URLS:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = (entry.get("title") or "").strip()
            link = entry.get("link", "")
            if not title or title in seen_titles or (link and link in seen_links):
                continue

            published_parsed = entry.get("published_parsed")
            if published_parsed:
                published_dt = datetime(*published_parsed[:6], tzinfo=timezone.utc)
                if published_dt < cutoff:
                    continue

            article = Article(
                title=title,
                link=link,
                published=to_iso_date(published_parsed),
                source=parse_source(title),
                summary=strip_html(entry.get("summary", "")),
            )
            collected.append(article)
            seen_titles.add(title)
            if link:
                seen_links.add(link)

            if len(collected) >= MAX_ARTICLES:
                break

        if len(collected) >= MAX_ARTICLES:
            break

    return collected


def good_entity_text(text: str, lower_source_names: set[str]) -> bool:
    t = normalize_entity(text)
    if not t or len(t) < 2:
        return False
    lowered = t.lower()
    if lowered in NOISY_ENTITY_TERMS:
        return False
    if any(piece in lowered for piece in NOISY_ENTITY_SUBSTRINGS):
        return False
    if lowered in lower_source_names:
        return False
    if lowered.startswith("tesla ") and lowered != "tesla":
        return False
    if any(ch in t for ch in ",|/"):
        return False
    if len(t.split()) > 4:
        return False
    if not re.search(r"[A-Za-z]", t):
        return False
    return True


def extract_entities(nlp, articles: Iterable[Article]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_names = {a.source for a in articles if a.source and a.source != "Unknown"}
    lower_source_names = {s.lower() for s in source_names}
    records = []
    for article in articles:
        title_text = clean_title_for_nlp(article.title)
        summary_text = article.summary if article.summary and article.summary != article.title else ""
        text = f"{title_text}. {summary_text}".strip()
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in {"ORG", "GPE", "LOC", "FAC", "PRODUCT"} and good_entity_text(
                ent.text, lower_source_names
            ):
                clean_ent = normalize_entity(ent.text)
                records.append(
                    {
                        "article_title": article.title,
                        "entity": clean_ent,
                        "entity_type": ent.label_,
                    }
                )

    df = pd.DataFrame(records)
    if df.empty:
        return df, df

    grouped = (
        df.groupby(["entity", "entity_type"], as_index=False)
        .size()
        .rename(columns={"size": "mention_count"})
        .sort_values("mention_count", ascending=False)
    )
    return grouped, df.sort_values(["entity_type", "entity"])


def relation_type(sentence: str) -> str | None:
    s = sentence.lower()
    for rel, keys in RELATION_KEYWORDS.items():
        if any(k in s for k in keys):
            return rel
    return None


def extract_relationships(nlp, articles: Iterable[Article]) -> pd.DataFrame:
    source_names = {a.source for a in articles if a.source and a.source != "Unknown"}
    lower_source_names = {s.lower() for s in source_names}
    edges = []

    for article in articles:
        title_text = clean_title_for_nlp(article.title)
        summary_text = article.summary if article.summary and article.summary != article.title else ""
        text = f"{title_text}. {summary_text}".strip()
        doc = nlp(text)

        for sent in doc.sents:
            rel = relation_type(sent.text)
            if not rel:
                continue

            ents = [
                e
                for e in sent.ents
                if e.label_ in {"ORG", "GPE", "LOC", "FAC", "PRODUCT"}
                and good_entity_text(e.text, lower_source_names)
            ]
            if len(ents) < 2:
                continue

            company_ent = next((e for e in ents if e.text.strip().lower() == COMPANY.lower()), None)
            if company_ent:
                subject = company_ent
                obj = next((e for e in ents if e.text != subject.text), None)
            else:
                subject = next((e for e in ents if e.label_ == "ORG"), ents[0])
                obj = next((e for e in ents if e.text != subject.text), None)
            if obj is None:
                continue

            edges.append(
                {
                    "source": normalize_entity(subject.text),
                    "target": normalize_entity(obj.text),
                    "relationship": rel,
                    "evidence": sent.text.strip(),
                    "article_title": article.title,
                }
            )

    for article in articles:
        title_text = clean_title_for_nlp(article.title)
        summary_text = article.summary if article.summary and article.summary != article.title else ""
        text = f"{title_text}. {summary_text}".strip()
        doc = nlp(text)
        orgs = {
            normalize_entity(e.text)
            for e in doc.ents
            if e.label_ == "ORG"
            and e.text.strip().lower() != COMPANY.lower()
            and good_entity_text(e.text, lower_source_names)
        }
        for org in orgs:
            edges.append(
                {
                    "source": COMPANY,
                    "target": org,
                    "relationship": "mentioned_with",
                    "evidence": article.title,
                    "article_title": article.title,
                }
            )

    rel_df = pd.DataFrame(edges)
    if rel_df.empty:
        return rel_df

    rel_df = rel_df.drop_duplicates(subset=["source", "target", "relationship", "article_title"])
    return rel_df


def build_graph(entity_df: pd.DataFrame, rel_df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()

    type_lookup = {}
    if not entity_df.empty:
        for _, row in entity_df.iterrows():
            type_lookup[row["entity"]] = row["entity_type"]

    for node, ntype in type_lookup.items():
        g.add_node(node, entity_type=ntype)

    if COMPANY not in g:
        g.add_node(COMPANY, entity_type="ORG")

    if not rel_df.empty:
        grouped = rel_df.groupby(["source", "target"], as_index=False).agg(
            relation_count=("relationship", "size"),
            relation_labels=("relationship", lambda x: ", ".join(sorted(set(x)))),
        )
        for _, row in grouped.iterrows():
            s, t = row["source"], row["target"]
            if s not in g:
                g.add_node(s, entity_type=type_lookup.get(s, "OTHER"))
            if t not in g:
                g.add_node(t, entity_type=type_lookup.get(t, "OTHER"))
            g.add_edge(s, t, weight=int(row["relation_count"]), label=row["relation_labels"])

    return g


def draw_graph(g: nx.Graph, path: Path) -> None:
    if g.number_of_nodes() == 0:
        return

    color_map = {
        "ORG": "#1f77b4",
        "GPE": "#2ca02c",
        "LOC": "#17becf",
        "FAC": "#ff7f0e",
        "PRODUCT": "#d62728",
        "OTHER": "#7f7f7f",
    }

    node_colors = [color_map.get(g.nodes[n].get("entity_type", "OTHER"), "#7f7f7f") for n in g.nodes]
    node_sizes = [800 if n == COMPANY else 500 for n in g.nodes]

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(g, seed=42, k=0.8)
    nx.draw_networkx_nodes(g, pos, node_color=node_colors, node_size=node_sizes, alpha=0.9)
    nx.draw_networkx_edges(g, pos, width=1.5, alpha=0.5)
    nx.draw_networkx_labels(g, pos, font_size=8)

    plt.title(f"Supply Chain Network from Recent {COMPANY} News", fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Collecting news...")
    articles = collect_news()
    if not articles:
        raise RuntimeError("No articles found. Try another query or check internet access.")

    articles_df = pd.DataFrame([a.__dict__ for a in articles])
    articles_path = OUTPUT_DIR / "news_articles.csv"
    articles_df.to_csv(articles_path, index=False)

    sources_path = OUTPUT_DIR / "publisher_audit.csv"
    (
        articles_df.groupby("source", as_index=False)
        .size()
        .rename(columns={"size": "article_count"})
        .sort_values("article_count", ascending=False)
        .to_csv(sources_path, index=False)
    )

    print("Loading NLP model and extracting entities/relationships...")
    nlp = spacy.load("en_core_web_sm")

    entity_df, mention_df = extract_entities(nlp, articles)
    entity_path = OUTPUT_DIR / "extracted_entities.csv"
    entity_df.to_csv(entity_path, index=False)
    mention_path = OUTPUT_DIR / "entity_mentions.csv"
    mention_df.to_csv(mention_path, index=False)

    rel_df = extract_relationships(nlp, articles)
    rel_path = OUTPUT_DIR / "extracted_relationships.csv"
    rel_df.to_csv(rel_path, index=False)

    print("Building and drawing network graph...")
    g = build_graph(entity_df, rel_df)
    graph_path = OUTPUT_DIR / "supply_chain_network.png"
    draw_graph(g, graph_path)

    print("Done.")
    print(f"Articles: {articles_path}")
    print(f"Publisher audit: {sources_path}")
    print(f"Entities: {entity_path}")
    print(f"Entity mentions: {mention_path}")
    print(f"Relationships: {rel_path}")
    print(f"Graph: {graph_path}")


if __name__ == "__main__":
    main()
