# Theme codebook

Used for two hand tasks:

1. **Topic mapping**: assigning each BERTopic topic to a theme (`topics/topic_themes.csv`).
2. **Hand check**: labeling sample sentences blind (`topics/hand_check_sample.csv`).

This codebook was written *before* any topic output was viewed, so that the definitions come from the research question and not from what the model found. If a definition changes after you see the topics, record the change and the reason in the change log at the end, so the change is visible.

## General rules

1. **One theme per sentence.** Pick the theme the sentence is mainly *about*: what it reports or assesses. Ignore the reason or context it mentions in passing.
2. **Main subject wins.** Most of these sentences start with what they're about. In "Inflation has picked up, reflecting higher energy prices", inflation is the subject, so the theme is **inflation**, not energy.
3. **Policy actions are always `policy`,** whatever the reason given. "In light of the effects of the coronavirus, the Committee decided to lower the target range…" is **policy**, not pandemic. The one exception is asset purchases and holdings, which go to `balance_sheet` (rule 4).
4. **The Fed's own securities holdings go to `balance_sheet`; markets and lending go to `financial_stability`.**
   - Buying, reinvesting or running off Treasuries or mortgage-backed securities is `balance_sheet`.
   - Strains in markets, credit conditions, bank stress, emergency lending facilities and swap lines are `financial_stability`.
5. **A list of several indicators goes to `growth`,** unless one of them is plainly the subject. "Household spending, business investment and employment have…" is **growth**. "Job gains have been strong and the unemployment rate has declined" is **labor**.
6. **Use `other` sparingly,** only when no definition fits. When two themes fit equally, choose one using the rules above and add a note. Don't fall back to `other` in that case.
7. **Judge the text, not the date.** Don't use what you know happened in a given year. A 2020 sentence about household spending is **growth**, not pandemic, unless it mentions the virus.

## Themes

### `inflation`
**About:** prices and inflation. This covers current inflation readings, the inflation outlook, inflation expectations, and core or headline prices.
- **Includes:**
  - "inflation has moved up", "core inflation has remained subdued"
  - "longer-term inflation expectations remain well anchored"
  - "inflation pressures", "price stability is at risk", "elevated inflation"
- **Excludes:**
  - the 2% goal stated as a mandate or goal → `policy`
  - energy or commodity prices as the main subject → `energy_geopolitics`
- **Edge cases:**
  - "Inflation has eased, partly reflecting lower energy prices" → `inflation`
  - "Readings on inflation expectations…" → `inflation`

### `labor`
**About:** jobs and the labor market. This covers employment, unemployment, payrolls, labor force participation, wages and labor market slack.
- **Includes:**
  - "job gains have slowed", "the unemployment rate has edged up"
  - "labor market conditions have improved further"
  - "labor market slack", "labor utilization"
- **Excludes:**
  - "maximum employment" as a goal or mandate → `policy`
- **Edge case:** "Indicators of labor market conditions and economic activity have…" → choose by the subject; if it's genuinely equal, use `growth` (rule 5).

### `growth`
**About:** overall economic activity other than jobs and prices. This covers GDP or output, household spending and consumption, business investment, housing, exports and trade, manufacturing and productivity.
- **Includes:**
  - "economic activity has been rising at a solid rate"
  - "household spending has picked up"
  - "the housing sector remains weak"
  - "business fixed investment has softened"
- **Excludes:**
  - the virus as the cause or subject → `pandemic`
  - credit or financial market conditions → `financial_stability`
- **Edge cases:**
  - "Fiscal policy is restraining economic growth" → `growth`
  - "Economic growth abroad has slowed" → `growth`. Only war or a geopolitical event as the subject goes to `energy_geopolitics`.

### `financial_stability`
**About:** stress in financial markets and institutions, and the Fed's emergency responses to it. This covers credit conditions, tight or strained financial markets, bank stress, liquidity facilities, swap lines, and coordination with other central banks during a crisis.
- **Includes:**
  - "strains in financial markets have increased significantly"
  - "tighter credit conditions"
  - "the Federal Reserve will continue to provide liquidity as necessary"
  - "the U.S. banking system is sound and resilient"
- **Excludes:**
  - the Fed buying or holding securities → `balance_sheet`
  - "financial conditions" listed only as one input the Committee will monitor → `policy`
- **Edge cases:**
  - Liquidity statements from August 2007 → `financial_stability`
  - "Financial market developments" listed among the risks to the outlook → `financial_stability` if it is the main subject

### `pandemic`
**About:** COVID-19 and its direct effects: the virus, public health, vaccinations, social distancing, reopening, and the economic disruption explicitly blamed on the pandemic.
- **Includes:**
  - "the coronavirus outbreak is causing tremendous human and economic hardship"
  - "the path of the economy will depend significantly on the course of the virus"
  - "progress on vaccinations"
  - "sectors most adversely affected by the pandemic"
- **Excludes:**
  - pandemic-era sentences that don't mention the virus or pandemic (rule 7)
  - policy actions taken because of the pandemic → `policy` or `balance_sheet` (rule 3)
- **Edge case:** "Supply and demand imbalances related to the pandemic and the reopening of the economy have contributed to sizable price increases" → the subject is the cause, the pandemic, so `pandemic`. If the sentence led with "Inflation is elevated…", it would be `inflation`.

### `balance_sheet`
**About:** the Fed's own holdings of securities. This covers large-scale asset purchases (QE), their pace and tapering, reinvestment of principal payments, balance-sheet runoff (QT), the maturity extension program, and the size and composition of the balance sheet.
- **Includes:**
  - "the Committee will continue to purchase Treasury securities at a pace of…"
  - "maintaining its existing policy of reinvesting principal payments"
  - "will begin reducing its holdings of Treasury securities and agency debt and agency mortgage-backed securities"
- **Excludes:**
  - emergency lending or liquidity programs → `financial_stability`
  - the rate decision → `policy`
- **Edge case:** "This policy should help maintain accommodative financial conditions", following a sentence about purchases → `balance_sheet`, because it continues the purchase discussion. Judge it in context if you can see the neighbouring sentences; otherwise use `policy`.

### `energy_geopolitics`
**About:** energy and commodity prices as the subject, and geopolitical events: war, conflicts, terrorist attacks, global shocks that aren't purely macroeconomic, and supply disruptions from those events.
- **Includes:**
  - "energy prices have risen sharply"
  - "the war in Ukraine and related events are creating additional upward pressure on inflation" (the war is the subject)
  - "the terrorist attacks have significantly heightened uncertainty"
  - "elevated prices of oil and other commodities"
- **Excludes:**
  - inflation readings that only mention energy as a cause → `inflation` (rule 2)
- **Edge case:** "Higher energy prices have boosted inflation" → `energy_geopolitics`, because energy is the subject.

### `policy`
**About:** monetary policy itself, except the balance sheet. This covers:
- the rate decision
- the stance of policy
- forward guidance about the future path of rates
- data dependence
- the Committee's goals and mandate
- the balance of risks to its goals
- how it will judge future adjustments

**Includes:**
- "the Committee decided to maintain the target range…"
- "policy accommodation can be removed at a pace that is likely to be measured"
- "the Committee seeks to achieve maximum employment and inflation at the rate of 2 percent over the longer run"
- "the risks to achieving its goals are roughly in balance"
- "the Committee will continue to monitor the implications of incoming information"
- "the Committee would be prepared to adjust the stance of monetary policy as appropriate"

**Excludes:** asset purchases and holdings → `balance_sheet`.

**Edge case:** "Uncertainty about the economic outlook has increased" → `other`, unless it is framed as a risk to the Committee's goals, in which case `policy`.

### `other`
**For:** sentences that fit none of the themes above even after applying the rules, such as generic uncertainty statements, headings that slipped through, or one-off topics like the Y2K transition in 1999–2000.
- **Includes:**
  - "Uncertainty about the outlook remains elevated"
  - "the Committee will closely monitor developments"
- **Note:** when `other` is used for a whole BERTopic topic, say why in `notes`.

## How to fill in the two files

**`topics/topic_themes.csv`, one row per BERTopic topic:**
- Read the `representative_docs` before the keywords.
- Choose the theme that fits most of the topic's sentences.
- If about a third or more of the sentences belong to another theme, write `mixed: <other theme>` in `notes`.

**`topics/hand_check_sample.csv`, one row per sentence:**
- Fill in `my_theme` with exactly one theme name from this file.
- Optionally fill in `confidence`: 1 = guess, 2 = fairly sure, 3 = certain.
- Do not open `hand_check_key.csv` or `unique_sentence_topics.csv` until all rows are labeled.
- Label the sample **before** filling in `topic_themes.csv` if you can, so that the topic keywords don't shape your sentence labels.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-16 | Initial version, written before viewing topic output | |
| 2026-11-16 | Topic mapping revised after the first-pass hand check (first pass kept in `topic_themes_first_pass.csv`): 2 balance sheet → financial stability; 24 growth → policy; 25 financial stability → policy; 31 labor → growth; 38 growth → balance sheet; 39 energy_geopolitics → growth | Brought into line with the definitions above: lending facilities are `financial_stability`; the balance of risks to goals and rate-path-conditioned outlook are `policy`; productivity is `growth`; asset-purchase conditions are `balance_sheet`; "downside risks to growth" has growth as its subject. No codebook definitions changed. Topic 2 is mixed (asset purchases + 2008 liquidity facilities); topics 3 and 44 (mapped to inflation) contain the energy/war sentences |
