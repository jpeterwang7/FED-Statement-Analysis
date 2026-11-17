"""
Map hand-named themes back to every sentence and compute each statement's theme mix.

Reads fomc_sentences.csv, topics/unique_sentence_topics.csv, topics/topic_themes.csv and
fomc_statements_metadata.csv. Writes:
  fomc_sentence_topics.csv        one row per sentence (2,980): sentence_id, statement_id, topic, theme
                                  (admin sentences get theme "admin" and no topic)
  fomc_statement_themes.csv       long: one row per statement x theme, sentence and word shares
  fomc_statement_themes_wide.csv  wide: one row per statement, one sentence-share column per theme

Shares leave out admin sentences, so each statement's shares sum to 1.
Run:  .venv/bin/python scripts/theme_shares.py   (after fit_topics.py and filling topic_themes.csv)
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TOPICS = ROOT / "topics"

# Fixed theme order: stacking order in the streamgraph and column order in the wide file
THEMES = ["policy", "inflation", "labor", "growth", "pandemic", "balance_sheet",
          "financial_stability", "energy_geopolitics", "other"]


def norm(theme):
    return "_".join(str(theme).strip().lower().replace("-", " ").split())


def main():
    sent = pd.read_csv(ROOT / "fomc_sentences.csv")
    topics = pd.read_csv(TOPICS / "unique_sentence_topics.csv")[["text", "topic"]]
    mapping = pd.read_csv(TOPICS / "topic_themes.csv")[["topic_id", "theme"]]
    meta = pd.read_csv(ROOT / "fomc_statements_metadata.csv")

    mapping["theme"] = mapping["theme"].map(norm)
    unknown = set(mapping["theme"]) - set(THEMES)
    if mapping["theme"].isna().any() or unknown:
        sys.exit(f"topic_themes.csv has blank or unknown themes: {sorted(unknown)}")

    df = sent.merge(topics, on="text", how="left", validate="m:1")
    df = df.merge(mapping.rename(columns={"topic_id": "topic"}), on="topic", how="left")
    df.loc[df["is_admin"], ["topic", "theme"]] = [pd.NA, "admin"]
    missing = df["theme"].isna().sum()
    if missing:
        sys.exit(f"{missing} non-admin sentences have no theme; rerun fit_topics.py?")
    df["topic"] = df["topic"].astype("Int64")
    assert len(df) == len(sent)
    df[["sentence_id", "statement_id", "topic", "theme"]].to_csv(
        ROOT / "fomc_sentence_topics.csv", index=False)

    body = df[~df["is_admin"]]
    counts = body.groupby(["statement_id", "theme"]).agg(
        n_sentences=("sentence_id", "size"), n_words=("word_count", "sum"))
    full = pd.MultiIndex.from_product([body["statement_id"].unique(), THEMES],
                                      names=["statement_id", "theme"])
    counts = counts.reindex(full, fill_value=0).reset_index()
    tot = counts.groupby("statement_id")[["n_sentences", "n_words"]].transform("sum")
    counts["share"] = counts["n_sentences"] / tot["n_sentences"]
    counts["word_share"] = counts["n_words"] / tot["n_words"]

    keep = ["statement_id", "release_date", "chair", "meeting_type", "statement_kind"]
    long = meta[keep].merge(counts, on="statement_id", how="inner")
    long["theme"] = pd.Categorical(long["theme"], THEMES, ordered=True)
    long = long.sort_values(["release_date", "statement_id", "theme"])
    long.to_csv(ROOT / "fomc_statement_themes.csv", index=False)

    wide = long.pivot(index="statement_id", columns="theme", values="share")[THEMES].reset_index()
    wide = meta[keep].merge(wide, on="statement_id", how="inner")
    wide.to_csv(ROOT / "fomc_statement_themes_wide.csv", index=False)

    sums = long.groupby("statement_id", observed=True)["share"].sum()
    assert (sums - 1).abs().max() < 1e-9
    dropped = len(meta) - wide.shape[0]
    print(f"{len(df)} sentences tagged; {wide.shape[0]} statements with shares"
          + (f" ({dropped} statements had only admin sentences)" if dropped else ""))
    print(body["theme"].value_counts(normalize=True).reindex(THEMES).fillna(0)
          .map("{:.1%}".format).to_string())


if __name__ == "__main__":
    main()
