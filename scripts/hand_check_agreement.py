"""
Compare the hand labels in topics/hand_check_sample.csv with the model's theme
(the sentence's BERTopic topic, mapped through topics/topic_themes.csv).

Writes topics/hand_check_disagreements.csv and prints overall agreement, Cohen's kappa
and agreement per theme.

Run:  .venv/bin/python scripts/hand_check_agreement.py [mapping.csv]
      (defaults to topics/topic_themes.csv; pass topics/topic_themes_first_pass.csv
       to reproduce the first-pass number)
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
TOPICS = ROOT / "topics"


def norm(theme):
    """'balance sheet' / 'Balance_Sheet' -> 'balance_sheet'."""
    return "_".join(str(theme).strip().lower().replace("-", " ").split())


def main():
    mapping_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else TOPICS / "topic_themes.csv"
    sample = pd.read_csv(TOPICS / "hand_check_sample.csv")
    key = pd.read_csv(TOPICS / "hand_check_key.csv")
    mapping = pd.read_csv(mapping_path)

    df = sample.merge(key, on="sentence_id", validate="1:1").merge(
        mapping[["topic_id", "theme"]].rename(columns={"topic_id": "topic", "theme": "model_theme"}),
        on="topic", how="left", validate="m:1")
    if df["my_theme"].isna().any() or df["model_theme"].isna().any():
        sys.exit("Blank labels or unmapped topics; fill them in first.")
    df["my_theme"] = df["my_theme"].map(norm)
    df["model_theme"] = df["model_theme"].map(norm)
    df["agree"] = df["my_theme"] == df["model_theme"]

    print(f"Mapping: {mapping_path.relative_to(ROOT)}")
    print(f"Agreement: {df['agree'].sum()}/{len(df)} = {df['agree'].mean():.0%}")
    print(f"Cohen's kappa: {cohen_kappa_score(df['my_theme'], df['model_theme']):.2f}\n")

    per = df.groupby("my_theme")["agree"].agg(n="size", agreed="sum")
    per["rate"] = (per["agreed"] / per["n"]).map("{:.0%}".format)
    print("By hand label:")
    print(per.to_string(), "\n")

    print("Confusion (rows = hand, columns = model):")
    print(pd.crosstab(df["my_theme"], df["model_theme"]).to_string(), "\n")

    out = df.loc[~df["agree"], ["sentence_id", "topic", "my_theme", "model_theme", "text"]]
    out.sort_values(["topic", "sentence_id"]).to_csv(TOPICS / "hand_check_disagreements.csv", index=False)
    print(f"{len(out)} disagreements written to topics/hand_check_disagreements.csv")


if __name__ == "__main__":
    main()
