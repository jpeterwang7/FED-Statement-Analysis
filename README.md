# FOMC statement corpus (2000 – Sept 2026)

Every FOMC policy statement from February 2000 through September 16, 2026, taken from federalreserve.gov
(meeting calendar + "Historical materials by year" pages). The FOMC meets 8 times a year, not monthly,
so there are 8 regular statements per year plus emergency/intermeeting ones (e.g. Jan 2001, Jan 2008, Mar 2020).

- **227 statements** total: 213 regular post-meeting statements + 14 unscheduled or supplementary ones
- The next statement (Oct 28, 2026 meeting) is not included yet

## What's in the folder

| Path | What it is |
|---|---|
| `statements/YYYY-MM-DD_statement.txt` | One statement per file, plain text, paragraphs separated by a blank line |
| `fomc_statements_metadata.csv` | One row per statement (columns below) |
| `raw/fomc_statements_raw.json` | Raw scrape: every statement's paragraphs, source URL, release date/time as printed |
| `scripts/build_corpus.py` | Rebuilds the txt files and CSV from the raw JSON: `python3 scripts/build_corpus.py` |
| `fomc_sentences.csv` | One row per sentence (voting records left out), with decision/admin tags. Columns below |
| `scripts/segment_sentences.py` | Rebuilds `fomc_sentences.csv` from the txt files: `python3 scripts/segment_sentences.py` (run after `build_corpus.py`) |

For most analysis, filter to `meeting_type == "scheduled"` and `statement_kind == "main"` (213 rows).

## About the text

- Verbatim statement text. Removed: "related links" lists, media-contact lines, and pointers to other central banks' sites.
- Voting records are kept in the text. Use `word_count_excl_votes` when comparing statement length across chairs,
  since voting lists got longer over time and the 2026 statements use a one-line vote tally instead.
- Fractions appear as printed by the Fed (`3-1/2`); a few 2002 statements print them as `1 3/4`.
- Two pages had text outside the normal HTML paragraphs and were patched (first sentence of 2003-08-12; voting record of 2009-06-24). Both were checked against the live page.
- Two files share a date: `2019-01-30_statement.txt` (regular) and `2019-01-30_statement_2.txt` (balance-sheet normalization statement).

## CSV columns

| Column | Meaning |
|---|---|
| statement_id, file_name | ID and path of the matching txt file |
| release_date, release_weekday | Date the statement was published |
| release_time_et, release_time_zone, release_time_text | Release time as printed (e.g. `14:00`, `EDT`). Blank when the page only says "For immediate release" (most pages before ~2015) |
| year, chair | Chair at release (Greenspan, Bernanke, Yellen, Powell, Warsh) |
| meeting_label, meeting_start_date, meeting_end_date | Meeting the Fed lists the statement under |
| meeting_type | `scheduled`, or `unscheduled` (conference calls, emergency meetings, statements released on a different day from the meeting) |
| statement_kind | `main`, or `supplementary` (a second statement for the same meeting, e.g. 2019 balance-sheet statements) |
| page_title | Title of the Fed page (blank for pre-2006 pages) |
| fed_funds_target_low / high | Fed funds target after the decision, in percent. Before Dec 2008 it was a single number, so low = high |
| rate_change_bp, rate_decision | Change vs. the previous statement's target (upper bound), in basis points; `hike` / `cut` / `hold`. Dec 2008 shows -75 (1% to 0–0.25%). Blank for statements with no target (e.g. Aug 2007 liquidity, swap-line statements) |
| has_voting_record | Whether the statement lists votes (statements only started including votes in March 2002) |
| dissent_count, dissenters | Number and names of members voting against. Blank when there is no voting record |
| press_conference | Whether the Fed lists a press conference for that meeting |
| word_count, word_count_excl_votes, paragraph_count, char_count | Length measures (whitespace word count) |
| source_url | Link to the statement on federalreserve.gov |

## Sentences CSV

`fomc_sentences.csv` splits every statement into sentences (2,980 rows). The sentence is the unit for FinBERT / LLM labels,
dictionary scores and BERTopic; average back up to the statement with `statement_id`.

| Column | Meaning |
|---|---|
| sentence_id | `<statement_id>_sNN`. Unique key for joining scores and hand labels |
| statement_id, release_date | Join to `fomc_statements_metadata.csv` |
| paragraph | Paragraph number in the txt file, counting from 1 (vote paragraphs included in the count) |
| sentence | Position of the sentence in the statement, counting from 1 (vote paragraphs left out) |
| text | Sentence text with whitespace normalized |
| word_count | Whitespace word count |
| is_decision | Sentence announces the fed funds rate setting (e.g. "decided to raise the target range…"). Leave these out to measure language beyond the rate decision. Some also carry guidance ("…and anticipates that ongoing increases will be appropriate"), and three 2013 statements have two |
| is_admin | Procedural text: discount-rate and other Board actions, the 2019–2020 Desk directives, changes to how votes and minutes are released. Usually left out of tone measures |

- Left out: voting paragraphs (same rule as `build_corpus.py`) and the "Federal Reserve Actions" / "Joint Statement by Central Banks" headings in 2008–2010.
- Fixed while splitting: a footnote marker in 2010-08-10 (`securities.1 The`) and a sentence broken across two paragraphs in 2004-12-14.
- Only about half the sentences are unique (1,521 of 2,980); boilerplate like "Consistent with its statutory mandate…" repeats dozens of times. Dedupe on `text` before fitting BERTopic.
