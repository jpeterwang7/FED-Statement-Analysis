#!/usr/bin/env python3
"""Linear SVM that predicts whether an FOMC sentence is forward-looking (Yes / No).

Features (one block each, stacked side by side):
  1. TF-IDF of the target sentence, word 1-3 grams        weight 1.0
  2. TF-IDF of the PREVIOUS sentence (same passage)        weight W_CTX  (smaller)
  3. TF-IDF of the NEXT sentence (same passage)            weight W_CTX  (smaller)
  4. Numeric / flag features, scaled                       weight W_NUM  (modest)
     is_decision, is_admin, is_first, is_last (statement), paragraph, sentence position, word_count.
  5. TONE block (from the predicted_tone / confidence columns)   weight W_TONE
     tone_hawkish / tone_neutral / tone_dovish one-hot indicators, tone_score (+1 hawkish, -1 dovish,
     0 neutral, times the classifier's confidence) and tone_change (tone_score minus the previous
     sentence's tone_score inside the same statement; 0 for the first sentence).

The model is run once for each text column (clean_text, clean_text_phrases) so you can compare them.

Evaluation design
  * The hand-labelled 300 sentences are the TEST set. They never enter training or tuning, and any
    training sentence whose text is identical to a test sentence is dropped too (boilerplate leakage).
  * Tuning uses expanding-window, time-ordered folds with a gap of a few statements before each
    validation block; metrics are pooled over the folds.
  * The final model is refit on all remaining rows with the best settings and scored once on the test set.

If the scored CSV has no clean_text / clean_text_phrases columns they are built here (see add_clean_columns).

Usage
  python svm_forward_guidance.py --scored fomc_sentences_forward_guidance_scored.csv \
      --hand Future_Guidance_Statement_Scores_handlabel_consistent.csv \
      --tone fomc_sentence_scores.csv [--tone-key sentence_id] [--scope statement|paragraph] [--out svm_output]
"""
import argparse
import itertools
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# ------------------------------------------------------------------ settings you may want to change
TEXT_COLUMNS = ["clean_text", "clean_text_phrases"]
TONE_LABEL_COL, TONE_CONF_COL = "predicted_tone", "confidence"   # columns in the tone CSV
TONE_SIGN = {"hawkish": 1.0, "neutral": 0.0, "dovish": -1.0}
NGRAM_RANGE = (1, 3)
MIN_DF = 2
GRID_C = [0.1, 0.3, 1.0, 3.0]
GRID_W_CTX = [0.0, 0.3, 0.5]       # 0.0 switches the previous/next-sentence blocks off (ablation)
GRID_W_NUM = [0.0, 0.2, 0.4]       # 0.0 switches the numeric block off (ablation)
GRID_W_TONE = [0.0, 0.2, 0.4]      # 0.0 switches the tone block off (ablation); only searched if --tone is given
# (train through year, validate first year, validate last year); a few statements before each block are dropped
FOLDS = [(2010, 2011, 2014), (2014, 2015, 2018), (2018, 2019, 2022), (2022, 2023, 2026)]
GAP_STATEMENTS = 2
N_BOOT = 1000
SEED = 42


# ------------------------------------------------------------------ data
PHRASES = ["federal funds rate", "target range", "for some time", "extended period", "considerable time",
           "maximum employment", "price stability", "longer run", "balance sheet", "forward guidance",
           "monetary policy", "economic activity", "labor market", "inflation expectations",
           "policy firming", "policy accommodation", "additional firming", "incoming data",
           "appropriate to", "remain patient", "data dependent", "stance of monetary policy",
           "securities holdings", "mortgage backed securities", "agency debt"]


def add_clean_columns(df):
    """Builds clean_text (lower-cased, numbers/dates/percents replaced by tokens) and clean_text_phrases
    (same, plus key policy phrases joined with underscores so they act as single words)."""
    def clean(t):
        t = str(t).lower()
        t = re.sub(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b"
                   r"(\s+\d{1,2})?(,\s*\d{4})?", " DATE ", t)
        t = re.sub(r"\b\d+(?:[-/.]\d+)*\s*(?:basis points?|percent|percentage points?)\b", " NUM_UNIT ", t)
        t = re.sub(r"\b\d+(?:[-/.,]\d+)*\b", " NUM ", t)
        t = re.sub(r"[^\w\s]", " ", t)
        return re.sub(r"\s+", " ", t).strip()

    def phrases(t):
        for p in PHRASES:
            t = re.sub(r"\b" + re.escape(p.replace("-", " ")) + r"\b", p.replace(" ", "_"), t)
        return t

    df["clean_text"] = df["text"].map(clean)
    df["clean_text_phrases"] = df["clean_text"].map(phrases)
    return df


def add_tone_features(df):
    """Turns predicted_tone + confidence into one-hot indicators, a signed confidence-weighted score,
    and the change in that score from the previous sentence of the same statement."""
    df["tone_label"] = df["tone_label"].fillna("neutral").str.strip().str.lower()
    df["tone_conf"] = pd.to_numeric(df["tone_conf"], errors="coerce").fillna(0.0)
    for lab in ("hawkish", "neutral", "dovish"):
        df[f"tone_{lab}"] = (df["tone_label"] == lab).astype(int)
    df["tone_score"] = df["tone_label"].map(TONE_SIGN).fillna(0.0) * df["tone_conf"]
    df["tone_change"] = df.groupby("statement_id", sort=False)["tone_score"].diff().fillna(0.0)
    return df


def load_data(scored_csv, tone_csv, tone_key, scope):
    df = pd.read_csv(scored_csv, keep_default_na=False, na_values=[""])
    if "clean_text" not in df.columns or "clean_text_phrases" not in df.columns:
        print("NOTE: clean_text / clean_text_phrases not in the scored file; building them from 'text'.")
        df = add_clean_columns(df)
    df["release_date"] = pd.to_datetime(df["release_date"])
    df["year"] = df["release_date"].dt.year
    df["y"] = (df["forward_looking"] == "Yes").astype(int)

    tone_cont, tone_bin = [], []
    if tone_csv:
        tone = pd.read_csv(tone_csv)
        for c in (tone_key, TONE_LABEL_COL, TONE_CONF_COL):
            if c not in tone.columns:
                raise SystemExit(f"column {c!r} not found in {tone_csv}")
        tone = tone[[tone_key, TONE_LABEL_COL, TONE_CONF_COL]].rename(
            columns={TONE_LABEL_COL: "tone_label", TONE_CONF_COL: "tone_conf"})
        df = df.merge(tone.drop_duplicates(tone_key), on=tone_key, how="left")
        n_missing = int(df["tone_label"].isna().sum())
        if n_missing:
            print(f"NOTE: {n_missing} sentences had no tone row; treated as neutral / confidence 0.")
    else:
        print("NOTE: no --tone file given, so the tone block is not used.")

    df = df.sort_values(["release_date", "statement_id", "sentence"]).reset_index(drop=True)
    if tone_csv:
        df = add_tone_features(df)
        tone_cont = ["tone_score", "tone_change"]
        tone_bin = ["tone_hawkish", "tone_neutral", "tone_dovish"]

    # previous / next sentence text, only when it sits in the same statement (or the same paragraph)
    keys = ["statement_id"] + (["paragraph"] if scope == "paragraph" else [])
    grp = df.groupby(keys, sort=False)
    for col in TEXT_COLUMNS:
        df[f"prev_{col}"] = grp[col].shift(1).fillna("")
        df[f"next_{col}"] = grp[col].shift(-1).fillna("")

    by_stmt = df.groupby("statement_id", sort=False)
    df["is_first"] = (by_stmt.cumcount() == 0).astype(int)
    df["is_last"] = (by_stmt.cumcount(ascending=False) == 0).astype(int)
    for c in ("is_decision", "is_admin"):
        df[c] = df[c].astype(int)
    return df, tone_cont, tone_bin


def split_pool_and_test(df, hand_csv, drop_duplicates=True):
    hand = pd.read_csv(hand_csv)
    hand = hand[["sentence_id", "hand_forward_looking"] + (["period"] if "period" in hand.columns else [])]
    hand["y_hand"] = (hand["hand_forward_looking"].astype(str).str.strip().str.lower() == "yes").astype(int)
    test = df.merge(hand.drop(columns=["hand_forward_looking"]), on="sentence_id", how="inner")
    if len(test) != len(hand):
        raise SystemExit("some hand-labelled sentence_ids are missing from the scored file")
    test_texts = set(test["clean_text"])
    in_test = df["sentence_id"].isin(test["sentence_id"])
    dup_of_test = (df["clean_text"].isin(test_texts) & ~in_test) if drop_duplicates else (in_test & False)
    pool = df[~in_test & ~dup_of_test].reset_index(drop=True)
    print(f"pool (train + tuning): {len(pool)} rows | test: {len(test)} rows | "
          f"dropped as exact text duplicates of test sentences: {int(dup_of_test.sum())}")
    return pool, test.reset_index(drop=True)


# ------------------------------------------------------------------ features
class Features:
    """Fits the vocabulary and scalers on training rows only; produces five blocks per dataset."""

    def __init__(self, text_col, cont_cols, bin_cols, tone_cont=(), tone_bin=()):
        self.text_col, self.cont_cols, self.bin_cols = text_col, list(cont_cols), list(bin_cols)
        self.tone_cont, self.tone_bin = list(tone_cont), list(tone_bin)

    def fit(self, d):
        # one shared vocabulary (fit on target sentences) so previous/next text uses the same columns
        self.vec = TfidfVectorizer(ngram_range=NGRAM_RANGE, token_pattern=r"(?u)\b\w+\b", lowercase=False,
                                   sublinear_tf=True, min_df=MIN_DF)
        self.vec.fit(d[self.text_col])
        self.scaler = StandardScaler().fit(d[self.cont_cols])
        self.tone_scaler = StandardScaler().fit(d[self.tone_cont]) if self.tone_cont else None
        return self

    def blocks(self, d):
        t = self.vec.transform(d[self.text_col])
        p = self.vec.transform(d["prev_" + self.text_col])     # empty string -> all-zero row
        n = self.vec.transform(d["next_" + self.text_col])
        num = np.hstack([self.scaler.transform(d[self.cont_cols]), d[self.bin_cols].to_numpy(float)])
        if self.tone_cont:
            tone = np.hstack([self.tone_scaler.transform(d[self.tone_cont]), d[self.tone_bin].to_numpy(float)])
        else:
            tone = np.zeros((len(d), 0))
        return t, p, n, sparse.csr_matrix(num), sparse.csr_matrix(tone)

    @staticmethod
    def assemble(b, w_ctx, w_num, w_tone=0.0):
        t, p, n, num, tone = b
        return sparse.hstack([t, w_ctx * p, w_ctx * n, w_num * num, w_tone * tone]).tocsr()


# ------------------------------------------------------------------ evaluation helpers
def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], zero_division=0)
    return dict(accuracy=accuracy_score(y_true, y_pred),
                macro_f1=f1_score(y_true, y_pred, average="macro", zero_division=0),
                f1_yes=f[0], precision_yes=p[0], recall_yes=r[0])


def bootstrap_ci(y_true, y_pred, n=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    vals = [f1_score(y_true[s], y_pred[s], average="macro", zero_division=0)
            for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def make_folds(pool):
    stm = pool.drop_duplicates("statement_id")[["statement_id", "year"]]   # already in date order
    out = []
    for train_end, v0, v1 in FOLDS:
        tr_ids = stm[stm["year"] <= train_end]["statement_id"].tolist()
        if GAP_STATEMENTS:
            tr_ids = tr_ids[:-GAP_STATEMENTS]
        va_ids = stm[(stm["year"] >= v0) & (stm["year"] <= v1)]["statement_id"]
        tr = np.where(pool["statement_id"].isin(tr_ids))[0]
        va = np.where(pool["statement_id"].isin(va_ids))[0]
        if len(tr) and len(va):
            out.append((tr, va))
    return out


def new_model(C):
    return LinearSVC(C=C, class_weight="balanced", max_iter=20000, random_state=SEED)


# ------------------------------------------------------------------ main experiment
def run(args):
    warnings.filterwarnings("ignore")
    os.makedirs(args.out, exist_ok=True)
    df, tone_cont, tone_bin = load_data(args.scored, args.tone, args.tone_key, args.scope)
    pool, test = split_pool_and_test(df, args.hand, not args.keep_test_duplicates)
    cont_cols = ["paragraph", "sentence", "word_count"]
    bin_cols = ["is_decision", "is_admin", "is_first", "is_last"]
    print("numeric features:", cont_cols + bin_cols)
    print("tone features   :", (tone_cont + tone_bin) or "none")

    folds = make_folds(pool)
    grid_tone = GRID_W_TONE if tone_cont else [0.0]
    grid = list(itertools.product(GRID_C, GRID_W_CTX, GRID_W_NUM, grid_tone))
    summary, test_rows = [], []

    for text_col in TEXT_COLUMNS:
        print(f"\n=================== text column: {text_col} ===================")
        # --- cross-validated tuning (time-ordered folds, pooled predictions)
        preds = {g: [] for g in grid}
        truth, last_fold = [], []
        for k, (tr, va) in enumerate(folds):
            ft = Features(text_col, cont_cols, bin_cols, tone_cont, tone_bin).fit(pool.iloc[tr])
            btr, bva = ft.blocks(pool.iloc[tr]), ft.blocks(pool.iloc[va])
            ytr, yva = pool["y"].to_numpy()[tr], pool["y"].to_numpy()[va]
            truth.append(yva)
            for (C, wc, wn, wt) in grid:
                clf = new_model(C).fit(Features.assemble(btr, wc, wn, wt), ytr)
                preds[(C, wc, wn, wt)].append(clf.predict(Features.assemble(bva, wc, wn, wt)))
        y_all = np.concatenate(truth)
        rows = []
        for g in grid:
            p_all = np.concatenate(preds[g])
            m = metrics(y_all, p_all)
            m_last = metrics(truth[-1], preds[g][-1])
            rows.append(dict(text_column=text_col, C=g[0], w_context=g[1], w_numeric=g[2], w_tone=g[3],
                             cv_macro_f1=m["macro_f1"], cv_f1_yes=m["f1_yes"], cv_accuracy=m["accuracy"],
                             last_fold_macro_f1=m_last["macro_f1"]))
        cv = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False).reset_index(drop=True)
        cv.to_csv(os.path.join(args.out, f"cv_results_{text_col}.csv"), index=False)
        best = cv.iloc[0]
        print("top CV settings:\n", cv.head(5).round(3).to_string(index=False))
        # ablation: best score with each optional block switched off
        for label, mask in (("no context", cv.w_context == 0), ("no numeric", cv.w_numeric == 0),
                            ("no tone", cv.w_tone == 0),
                            ("text only", (cv.w_context == 0) & (cv.w_numeric == 0) & (cv.w_tone == 0))):
            print(f"  best CV macro-F1 with {label:11s}: {cv[mask].cv_macro_f1.max():.3f}")

        # --- final fit on the whole pool, single look at the hand-labelled test set
        ft = Features(text_col, cont_cols, bin_cols, tone_cont, tone_bin).fit(pool)
        btr_all, bte_all = ft.blocks(pool), ft.blocks(test)
        Xtr = Features.assemble(btr_all, best.w_context, best.w_numeric, best.w_tone)
        Xte = Features.assemble(bte_all, best.w_context, best.w_numeric, best.w_tone)
        clf = new_model(best.C).fit(Xtr, pool["y"].to_numpy())
        score = clf.decision_function(Xte)
        pred = (score > 0).astype(int)

        y_hand, y_llm = test["y_hand"].to_numpy(), test["y"].to_numpy()

        # reference only (not used for selection): best CV config with the tone block switched off
        ref = {}
        if tone_cont:
            b0 = cv[cv.w_tone == 0].iloc[0]
            c0 = new_model(b0.C).fit(Features.assemble(btr_all, b0.w_context, b0.w_numeric, 0.0), pool["y"].to_numpy())
            p0 = (c0.decision_function(Features.assemble(bte_all, b0.w_context, b0.w_numeric, 0.0)) > 0).astype(int)
            ref = metrics(y_hand, p0)
            print(f"TEST vs HAND, same pipeline WITHOUT tone: macro-F1 {ref['macro_f1']:.3f} | "
                  f"F1(Yes) {ref['f1_yes']:.3f} | acc {ref['accuracy']:.3f}")
        m_hand, m_llm = metrics(y_hand, pred), metrics(y_llm, pred)
        lo, hi = bootstrap_ci(y_hand, pred)
        print(f"\nTEST vs HAND labels : macro-F1 {m_hand['macro_f1']:.3f} (95% CI {lo:.3f}-{hi:.3f}) | "
              f"F1(Yes) {m_hand['f1_yes']:.3f} | acc {m_hand['accuracy']:.3f}")
        print(f"TEST vs LLM  labels : macro-F1 {m_llm['macro_f1']:.3f} | F1(Yes) {m_llm['f1_yes']:.3f}")
        print("confusion (rows=hand No/Yes, cols=pred No/Yes):\n", confusion_matrix(y_hand, pred, labels=[0, 1]))
        if "period" in test.columns:
            for per, idx in test.groupby("period").groups.items():
                i = np.asarray(list(idx))
                print(f"   {per}: macro-F1 {f1_score(y_hand[i], pred[i], average='macro', zero_division=0):.3f}  (n={len(i)})")

        # interpretability: strongest target-sentence n-grams
        names = ft.vec.get_feature_names_out()
        w = clf.coef_[0][:len(names)]
        order = np.argsort(w)
        print("  strongest FORWARD n-grams :", ", ".join(names[order[::-1][:12]]))
        print("  strongest NOT-forward     :", ", ".join(names[order[:12]]))

        if tone_cont:
            tw = clf.coef_[0][-(len(tone_cont) + len(tone_bin)):] if best.w_tone > 0 else np.zeros(len(tone_cont) + len(tone_bin))
            print("  tone-block weights (scaled units):", ", ".join(f"{n}={v:+.3f}" for n, v in zip(tone_cont + tone_bin, tw)))

        out = test[["sentence_id", "release_date", "text"] + ([])].copy()
        if tone_cont:
            out["tone_label"] = test["tone_label"].to_numpy()
        out["hand_forward_looking"] = np.where(y_hand == 1, "Yes", "No")
        out["llm_forward_looking"] = np.where(y_llm == 1, "Yes", "No")
        out["svm_forward_looking"] = np.where(pred == 1, "Yes", "No")
        out["svm_score"] = score
        out.to_csv(os.path.join(args.out, f"test_predictions_{text_col}.csv"), index=False)
        summary.append(dict(text_column=text_col, C=best.C, w_context=best.w_context, w_numeric=best.w_numeric,
                            w_tone=best.w_tone, cv_macro_f1=best.cv_macro_f1, test_macro_f1_vs_hand=m_hand["macro_f1"],
                            test_macro_f1_vs_hand_no_tone=ref.get("macro_f1", np.nan),
                            test_f1_yes_vs_hand=m_hand["f1_yes"], test_macro_f1_vs_llm=m_llm["macro_f1"]))
    pd.DataFrame(summary).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print("\n", pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scored", required=True, help="scored CSV that contains clean_text and clean_text_phrases")
    ap.add_argument("--hand", required=True, help="hand-labelled CSV (sentence_id, hand_forward_looking, period)")
    ap.add_argument("--tone", default=None, help="CSV with sentence_id, predicted_tone, confidence (enables the tone block)")
    ap.add_argument("--tone-key", default="sentence_id", choices=["sentence_id", "statement_id"],
                    help="column used to join the tone file")
    ap.add_argument("--scope", default="statement", choices=["statement", "paragraph"],
                    help="neighbours count as 'the same passage' within the statement or only within the paragraph")
    ap.add_argument("--keep-test-duplicates", action="store_true",
                    help="do NOT drop training rows whose text equals a test sentence (leaks boilerplate; for comparison only)")
    ap.add_argument("--out", default="svm_output")
    run(ap.parse_args())
