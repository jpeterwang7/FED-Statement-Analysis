"""
Fit BERTopic on the unique, non-admin FOMC sentences.

Reads fomc_sentences.csv and writes (in topics/):
  unique_sentences.csv          the deduped documents the model is fitted on
  embeddings.npy                cached sentence embeddings (gitignored; delete to re-encode)
  unique_sentence_topics.csv    text, topic_raw (HDBSCAN, -1 = outlier), topic (outliers reassigned)
  topic_info.csv                one row per topic: size, keywords, representative sentences
  topic_themes.csv              blank mapping template for the hand step (never overwritten)
  hand_check_sample.csv         blind sample to label by hand (never overwritten)
  hand_check_key.csv            model topics for the sample (never overwritten)
  model/                        saved BERTopic model (gitignored)

Run:  .venv/bin/python scripts/fit_topics.py      (install requirements-topics.txt first)
Theme definitions for the hand steps: topics/theme_codebook.md
"""
import random
from pathlib import Path

import numpy as np
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from umap import UMAP

ROOT = Path(__file__).resolve().parent.parent
SENTENCES = ROOT / "fomc_sentences.csv"
OUT = ROOT / "topics"

SEED = 42
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
HAND_CHECK_N = 50
# "committee" opens most sentences and says nothing about the topic
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | {"committee"})


def load_docs():
    s = pd.read_csv(SENTENCES)
    s = s[~s["is_admin"]]
    # first occurrence of each text; its sentence_id is used for the hand-check sample
    u = s.drop_duplicates("text", keep="first").reset_index(drop=True)
    u["n_occurrences"] = u["text"].map(s["text"].value_counts())
    return u[["sentence_id", "release_date", "text", "n_occurrences"]]


def embed(docs):
    cache = OUT / "embeddings.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(docs):
            return emb
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    emb = model.encode(docs, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    np.save(cache, emb)
    return emb


def build_model():
    return BERTopic(
        embedding_model=SentenceTransformer(EMBEDDING_MODEL, device="cpu"),
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine",
                        random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=10, min_samples=5, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), min_df=2),
        representation_model={"KeyBERT": KeyBERTInspired(random_state=SEED)},
        calculate_probabilities=False,
    )


def write_if_absent(df, path):
    """Hand-edited files are created once and never overwritten by a refit."""
    if path.exists():
        print(f"  kept existing {path.relative_to(ROOT)} (delete it to regenerate)")
        return
    df.to_csv(path, index=False)


def hand_check_sample(u, topics):
    """About HAND_CHECK_N sentences, at least one per topic, the rest proportional to topic size."""
    rng = random.Random(SEED)
    by_topic = {}
    for i, t in enumerate(topics):
        by_topic.setdefault(t, []).append(i)
    total = len(topics)
    picks = []
    for t, idx in sorted(by_topic.items()):
        k = max(1, round(HAND_CHECK_N * len(idx) / total))
        picks += rng.sample(idx, min(k, len(idx)))
    rng.shuffle(picks)
    sample = pd.DataFrame({
        "sentence_id": u.loc[picks, "sentence_id"].values,
        "text": u.loc[picks, "text"].values,
        "my_theme": "",
        "confidence": "",
        "notes": "",
    })
    key = pd.DataFrame({
        "sentence_id": u.loc[picks, "sentence_id"].values,
        "topic": [topics[i] for i in picks],
    })
    return sample, key


def main():
    np.random.seed(SEED)
    random.seed(SEED)
    OUT.mkdir(exist_ok=True)

    u = load_docs()
    docs = u["text"].tolist()
    u.to_csv(OUT / "unique_sentences.csv", index=False)
    print(f"{len(docs)} unique non-admin sentences")

    emb = embed(docs)
    model = build_model()
    topics_raw, _ = model.fit_transform(docs, embeddings=emb)
    n_out = sum(t == -1 for t in topics_raw)

    topics = model.reduce_outliers(docs, topics_raw, strategy="embeddings", embeddings=emb)
    model.update_topics(docs, topics=topics, vectorizer_model=model.vectorizer_model,
                        representation_model=model.representation_model)

    pd.DataFrame({"text": docs, "topic_raw": topics_raw, "topic": topics}).to_csv(
        OUT / "unique_sentence_topics.csv", index=False)

    info = model.get_topic_info()
    info = pd.DataFrame({
        "topic_id": info["Topic"],
        "size": info["Count"],
        "keywords": info["Representation"].map(lambda w: ", ".join(w)),
        "keybert_keywords": info["KeyBERT"].map(lambda w: ", ".join(w)),
        "representative_docs": info["Representative_Docs"].map(
            lambda d: " || ".join(d) if isinstance(d, list) else ""),
    })
    info.to_csv(OUT / "topic_info.csv", index=False)

    template = info[["topic_id", "size", "keywords"]].assign(theme="", notes="")
    write_if_absent(template, OUT / "topic_themes.csv")

    sample, key = hand_check_sample(u, topics)
    write_if_absent(sample, OUT / "hand_check_sample.csv")
    write_if_absent(key, OUT / "hand_check_key.csv")

    model.save(OUT / "model", serialization="safetensors", save_ctfidf=True,
               save_embedding_model=EMBEDDING_MODEL)

    print(f"{info['topic_id'].ge(0).sum()} topics; "
          f"{n_out} outliers before reassignment ({n_out / len(docs):.0%}), "
          f"{sum(t == -1 for t in topics)} after")


if __name__ == "__main__":
    main()
