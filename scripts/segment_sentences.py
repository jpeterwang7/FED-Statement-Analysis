"""
Split the FOMC statements into sentences for sentence-level analysis.

Reads statements/*.txt (listed in fomc_statements_metadata.csv) and writes
  fomc_sentences.csv   one row per sentence, voting records and headings left out

Run from anywhere:  python3 scripts/segment_sentences.py   (after build_corpus.py)
"""
import csv
import re
from pathlib import Path

from build_corpus import is_vote_paragraph

ROOT = Path(__file__).resolve().parent.parent
META_PATH = ROOT / "fomc_statements_metadata.csv"
CSV_PATH = ROOT / "fomc_sentences.csv"

# A sentence ends at . ! or ? (optionally followed by a closing quote) when the next one starts
# with a capital letter, quote, $ or parenthesis.
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“$(])|(?<=[.!?][\"”])\s+(?=[A-Z\"“$(])")
# Periods that don't end a sentence: U.S., St. Louis, Jr., 2 p.m. EST, initials
ABBREV_END_RE = re.compile(r"\b(?:U\.S|St|Jr|Mr|Ms|Dr|No|[ap]\.m|[A-Z])\.$")
# Footnote marker glued to a sentence end (2010-08-10: "securities.1 The Committee")
FOOTNOTE_RE = re.compile(r"(?<=[a-z])\.\d+ (?=[A-Z])")

# Sentences that announce the federal funds rate setting. Some of them also carry guidance
# ("decided to raise the target range ... and anticipates that ongoing increases ...").
DECISION_RE = re.compile(
    r"\b(?:decided|voted|agreed)(?: today| unanimously)? to\b[^.]*\b(?:federal funds rate|target range|funds rate target)"
    r"|\b(?:has|have) (?:lowered|raised|reduced|cut|increased|decreased)\b[^.]*\bfederal funds rate"
    r"|\bwill (?:maintain|keep) the target range for the federal funds rate at"
    r"|\btoday reaffirmed its view that\b"
)
# Procedural text: discount rate actions, Board actions on reserve balances, open market directives,
# changes to how the FOMC releases votes and minutes
ADMIN_RE = re.compile(
    r"\bdiscount rate\b|\bBoards? of Directors\b|\bprimary credit\b|\bdiscount window\b"
    r"|^In a related (?:action|set of actions)\b|\bBoard (?:also )?established interest rates\b"
    r"|^\"Effective\b|\bdirects the Desk\b|\bto authorize and direct the Federal Reserve Bank of New York\b"
    r"|^More information about these plans\b|\broll call of the vote\b|\brelease of its minutes\b"
)


def split_sentences(paragraph):
    out, buf = [], ""
    for piece in SENTENCE_END_RE.split(paragraph):
        buf = f"{buf} {piece}" if buf else piece
        if ABBREV_END_RE.search(buf):
            continue
        out.append(buf)
        buf = ""
    if buf:
        out.append(buf)
    return out


def is_heading(paragraph):
    """'Federal Reserve Actions', 'Joint Statement by Central Banks'."""
    return len(paragraph.split()) < 8 and not re.search(r"[.!?:\"”]$", paragraph)


def main():
    with META_PATH.open(encoding="utf-8") as f:
        meta = list(csv.DictReader(f))

    rows = []
    for m in meta:
        text = (ROOT / m["file_name"]).read_text(encoding="utf-8")
        paras = []  # (paragraph number in the txt file, text)
        for para_no, p in enumerate((p for p in text.split("\n\n") if p.strip()), start=1):
            p = " ".join(p.split())
            if paras and p[0].islower():  # 2004-12-14 has a sentence broken across two paragraphs
                paras[-1] = (paras[-1][0], f"{paras[-1][1]} {p}")
            else:
                paras.append((para_no, p))

        n, prev_vote = 0, False
        for para_no, p in paras:
            prev_vote = is_vote_paragraph(p, prev_vote)
            if prev_vote or is_heading(p):
                continue
            sents = split_sentences(FOOTNOTE_RE.sub(". ", p))
            # an admin opening sentence makes the whole paragraph admin
            # ("In taking the discount rate action, ..." / the quoted Desk directive)
            admin_para = bool(ADMIN_RE.search(sents[0]))
            for s in sents:
                n += 1
                is_admin = admin_para or bool(ADMIN_RE.search(s))
                rows.append({
                    "sentence_id": f"{m['statement_id']}_s{n:02d}",
                    "statement_id": m["statement_id"],
                    "release_date": m["release_date"],
                    "paragraph": para_no,
                    "sentence": n,
                    "text": s,
                    "word_count": len(s.split()),
                    "is_decision": not is_admin and bool(DECISION_RE.search(s)),
                    "is_admin": is_admin,
                })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} sentences from {len(meta)} statements to {CSV_PATH}")


if __name__ == "__main__":
    main()
