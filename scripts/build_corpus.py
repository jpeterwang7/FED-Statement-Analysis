"""
Build the FOMC statement corpus from raw/fomc_statements_raw.json.

Outputs (relative to the project folder):
  statements/<YYYY-MM-DD>_statement.txt   one plain-text file per statement
  fomc_statements_metadata.csv             one row per statement

Run from anywhere:  python3 scripts/build_corpus.py
"""
import csv
import json
import re
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw" / "fomc_statements_raw.json"
OUT_DIR = ROOT / "statements"
CSV_PATH = ROOT / "fomc_statements_metadata.csv"

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# Chair at the time of each statement (by release date).
CHAIRS = [
    (date(1987, 8, 11), "Alan Greenspan"),
    (date(2006, 2, 1), "Ben Bernanke"),
    (date(2014, 2, 3), "Janet Yellen"),
    (date(2018, 2, 5), "Jerome Powell"),
    (date(2026, 5, 22), "Kevin Warsh"),
]


def chair_on(d):
    name = None
    for start, who in CHAIRS:
        if d >= start:
            name = who
    return name


def parse_release_date(raw):
    raw = raw.strip().replace("  ", " ")
    return datetime.strptime(raw, "%B %d, %Y").date()


def parse_release_time(raw):
    """'For release at 2:00 p.m. EDT' -> ('14:00', 'EDT'). Immediate releases have no stated time."""
    m = re.search(r"(\d{1,2}):(\d{2})\s*([ap])\.m\.\s*(E[SD]T)", raw)
    if not m:
        return "", ""
    h, mi, ap, tz = int(m.group(1)), m.group(2), m.group(3), m.group(4)
    if ap == "p" and h != 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return f"{h:02d}:{mi}", tz


def parse_meeting_dates(heading, year):
    """Handles 'January 30-31 Meeting - 2001', 'July 31-August 1 Meeting - 2012',
    'March 2 (unscheduled) Meeting - 2020', 'August 16 Conference Call - 2007',
    'Apr/May 30-1', 'June 16-17*'."""
    m = re.match(r"\s*([A-Za-z]+)(?:/([A-Za-z]+))?\s+(\d{1,2})(?:-(?:([A-Za-z]+)\s+)?(\d{1,2}))?", heading)
    if not m:
        return None, None
    m1 = MONTHS[m.group(1)[:3].lower()]
    second_month = m.group(2) or m.group(4)
    m2 = MONTHS[second_month[:3].lower()] if second_month else m1
    d1 = int(m.group(3))
    d2 = int(m.group(5)) if m.group(5) else d1
    start = date(year, m1, d1)
    end = date(year + (1 if m2 < m1 else 0), m2, d2)
    return start, end


def frac(s):
    """'5-3/4' or '5 3/4' -> 5.75, '1/4' -> 0.25, '6' -> 6.0"""
    s = s.strip().replace(" ", "-")
    if "/" in s and "-" in s:
        whole, part = s.split("-", 1)
    else:
        whole, part = s, ""
    if "/" in whole:
        n, d = whole.split("/")
        return int(n) / int(d)
    val = float(whole)
    if part:
        n, d = part.split("/")
        val += int(n) / int(d)
    return val


NUM = r"(\d+[- ]\d/\d|\d/\d|\d+(?:\.\d+)?)"
RANGE_PATTERNS = [
    # "maintain the target range for the federal funds rate at 3-1/2 to 3-3/4 percent"
    # "lower the target range for the federal funds rate by 1/2 percentage point, to 1 to 1-1/4 percent"
    re.compile(r"target range for the federal funds rate(?: by [\d/\-]+ percentage points?)?,?(?: unchanged)? (?:at|to|of) "
               + NUM + r" to " + NUM + r" percent"),
    # 2014-15: "the current 0 to 1/4 percent target range for the federal funds rate"
    re.compile(NUM + r" to " + NUM + r" percent target range for the federal funds rate"),
    # March 23, 2020 directive: "maintain the federal funds rate in a target range of 0 to 1/4 percent"
    re.compile(r"federal funds rate in a target range of " + NUM + r" to " + NUM + r" percent"),
]
POINT_RE = re.compile(
    r"target for the federal funds rate(?: by)?(?: \d+ basis points)?(?: unchanged)? (?:at|to|of) "
    + NUM + r" percent")


def parse_target(text):
    """First stated fed funds target in the text -> (low, high). Point targets give low == high."""
    hits = []
    for rx in RANGE_PATTERNS:
        m = rx.search(text)
        if m:
            hits.append((m.start(), frac(m.group(1)), frac(m.group(2))))
    m = POINT_RE.search(text)
    if m:
        v = frac(m.group(1))
        hits.append((m.start(), v, v))
    if not hits:
        return None, None
    _, lo, hi = min(hits)
    return lo, hi


VOTE_PREFACE_RE = re.compile(r"approved the following statement for release by a (\d+)\s*[–-]\s*(\d+) vote")
NAME_RE = re.compile(r"[A-Z][a-z]+(?: [A-Z]\.)?(?: [A-Z][a-zA-Z'\-]+)+")


def is_vote_paragraph(p, prev_is_vote):
    if p.startswith("Voting") or VOTE_PREFACE_RE.search(p):
        return True
    if re.match(r"^(Governor|President)\b.*\bpreferred\b", p):
        return True
    # a voter list that continues onto a second paragraph
    if prev_is_vote and re.match(r"^[A-Z][a-z]+ [A-Z]\. [A-Z][a-zA-Z'\-]+;", p):
        return True
    return False


def parse_dissents(vote_text):
    """Returns (count, names) or (None, []) when the statement has no voting record.
    Handles 'were A, B, and C, who ...', 'was A, who ..., and B, who ...',
    'were A, who ...; and B, C, and D, who ...' and 'was A. Mr. A believed ...'."""
    if not vote_text:
        return None, []
    names = []
    followed_by_who = re.compile(NAME_RE.pattern + r"(?=(?:, Jr\.)?,? who\b)")
    for m in re.finditer(r"Voting against\b(.*?)(?=Voting for\b|$)", vote_text):
        seg = re.sub(r"^[^:]*?\b(?:was|were)\b:?\s*|^[^:]*?:\s*", "", m.group(1), count=1)
        if re.match(r"\s*none", seg, re.I):
            continue
        for chunk in seg.split(";"):
            lead = re.split(r",? who\b", chunk)[0]
            lead = re.split(r"(?<!\b[A-Z])(?<!Jr)\.\s", lead)[0]
            names += NAME_RE.findall(lead) + followed_by_who.findall(chunk)
    # de-duplicate by surname ("Edward M. Gramlich" / "Governor Gramlich")
    seen, unique = set(), []
    for n in names:
        surname = n.split()[-1]
        if surname not in seen:
            seen.add(surname)
            unique.append(n)
    tally = VOTE_PREFACE_RE.search(vote_text)
    count = int(tally.group(2)) if tally else len(unique)
    return count, unique


def main():
    data = json.loads(RAW.read_text(encoding="utf-8"))
    records = data["records"]
    for r in records:
        r["_release_date"] = parse_release_date(r["release_date_raw"])
    records.sort(key=lambda r: (r["_release_date"], r["link_index"]))

    OUT_DIR.mkdir(exist_ok=True)
    rows, used_names, prev_hi = [], set(), None
    for r in records:
        rd = r["_release_date"]
        paras = r["paragraphs"]
        full_text = "\n\n".join(paras)

        vote_flags, prev = [], False
        for p in paras:
            prev = is_vote_paragraph(p, prev)
            vote_flags.append(prev)
        policy_paras = [p for p, v in zip(paras, vote_flags) if not v]
        vote_text = " ".join(p for p, v in zip(paras, vote_flags) if v)

        base = f"{rd.isoformat()}_statement"
        fname, k = f"{base}.txt", 2
        while fname in used_names:
            fname, k = f"{base}_{k}.txt", k + 1
        used_names.add(fname)
        (OUT_DIR / fname).write_text(full_text + "\n", encoding="utf-8")

        start, end = parse_meeting_dates(r["meeting_heading"], r["year"])
        heading = r["meeting_heading"]
        scheduled = ("Conference Call" not in heading and "unscheduled" not in heading.lower()
                     and end is not None and rd == end)

        lo, hi = parse_target(" ".join(policy_paras))
        if hi is not None and prev_hi is not None:
            change = round((hi - prev_hi) * 100)
            decision = "hike" if change > 0 else "cut" if change < 0 else "hold"
        elif hi is not None:  # first statement in the sample: read the direction from the text
            mm = re.search(r"(raise|lower|increase|decrease|reduce)\w* its target for the federal funds rate (?:by )?(\d+) basis points", full_text)
            change = (int(mm.group(2)) * (1 if mm.group(1).startswith(("raise", "increase")) else -1)) if mm else ""
            decision = ("hike" if change and change > 0 else "cut" if change and change < 0 else "hold") if mm else ""
        else:
            change, decision = "", ""
        if hi is not None:
            prev_hi = hi

        dissent_count, dissenters = parse_dissents(vote_text)
        time_et, tz = parse_release_time(r["release_time_raw"])
        rows.append({
            "statement_id": fname[:-4],
            "file_name": f"statements/{fname}",
            "release_date": rd.isoformat(),
            "release_weekday": rd.strftime("%A"),
            "release_time_et": time_et,
            "release_time_zone": tz,
            "release_time_text": r["release_time_raw"],
            "year": rd.year,
            "chair": chair_on(rd),
            "meeting_label": heading,
            "meeting_start_date": start.isoformat() if start else "",
            "meeting_end_date": end.isoformat() if end else "",
            "meeting_type": "scheduled" if scheduled else "unscheduled",
            "statement_kind": "main" if r["link_index"] == 0 else "supplementary",
            "page_title": r["page_title"],
            "fed_funds_target_low": "" if lo is None else lo,
            "fed_funds_target_high": "" if hi is None else hi,
            "rate_change_bp": change,
            "rate_decision": decision,
            "has_voting_record": bool(vote_text),
            "dissent_count": "" if dissent_count is None else dissent_count,
            "dissenters": "; ".join(dissenters),
            "press_conference": r["press_conference_listed"],
            "word_count": len(full_text.split()),
            "word_count_excl_votes": len(" ".join(policy_paras).split()),
            "paragraph_count": len(paras),
            "char_count": len(full_text),
            "source_url": r["fetched_url"],
        })

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} statements to {OUT_DIR} and {CSV_PATH}")


if __name__ == "__main__":
    main()
