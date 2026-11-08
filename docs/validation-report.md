# RegZE Assistant: validation report

The full evaluation of [RegZE Assistant](../README.md), written as a model validation report: intended use, data, evaluation method, performance on a held-out test set, error analysis, experiments, reproducibility, deployment, limitations and monitoring. Every headline number comes from one run on the held-out test split (`r20261107T174057Z`), with its sample size beside it. Setup and commands are in the [README](../README.md#quick-start).

> Not legal or compliance advice. Card network rules, state law and an issuer's own account terms are out of scope.

## Contents

1. [Intended use](#1-intended-use)
2. [How it works](#2-how-it-works)
3. [Data](#3-data)
4. [Evaluation method](#4-evaluation-method)
5. [Results](#5-results)
6. [Error analysis](#6-error-analysis)
7. [Experiments](#7-experiments)
8. [Reproducibility](#8-reproducibility)
9. [Deployment](#9-deployment)
10. [Limitations](#10-limitations)
11. [Monitoring](#11-monitoring)
12. [Next steps](#12-next-steps)

## 1. Intended use

**Users.** Two kinds of internal staff, one answer format:

- a **frontline colleague** in a contact centre or branch, relaying a customer's situation in everyday words ("her debit card was stolen on Friday and she called us on Tuesday");
- a **compliance or operations specialist**, asking in regulatory terms ("under 1026.13(c)(2), what is the outer limit for resolving a billing error?").

Every metric is reported for both, because they stress search in opposite ways: everyday wording shares few words with the regulation, while paragraph numbers are exact strings.

**What it answers.** Questions answerable from Regulation Z subparts A, B and G (general, open-end credit, credit card accounts) and Regulation E subpart A, with their official interpretations, as published on one date (23 October 2026). Answers are two to six sentences with a citation on every claim. Deadlines and liability tiers come from a tool, not from the model.

**What it declines.** Card network rules, state law, an issuer's own terms, the statutes behind the regulations and other regulations or court decisions, the parts of Z and E that aren't indexed (closed-end credit, mortgages, remittances, model forms), rules that are not in force, and requests for a compliance sign-off.

**What it is not.** A source of legal advice, a compliance approval, or a system to put in front of customers. Its output is a starting point that a trained person checks against the cited paragraphs.

## 2. How it works

```
OFFLINE    eCFR API (XML, one "as of" date) -> parse into provisions -> status register
           -> chunks with headers -> Titan Text Embeddings V2 -> index/ (committed)

ONLINE     question -> API Gateway (key, throttle, daily quota) -> Lambda:
             hybrid search (vectors + BM25, reciprocal rank fusion; named paragraphs first) -> top 5 passages
             -> Claude Sonnet 4.6 on Bedrock, temperature 0
                  <-> dispute_timeline tool (at most 2 tool turns)
             -> citation guard -> Pydantic response
           MCP server (stdio): one tool, ask_card_rules, calling the API

EVAL       golden set (24 dev / 36 test) -> the same answer code -> code checks
           -> LLM judge (Claude Sonnet 4.5) -> runs.db (SQLite) -> report -> the author's review
```

- **Search** (`app/retrieve.py`). Vector search (cosine over Titan V2 vectors, 1,024 dimensions) and BM25 (built from the committed chunks at load), fused by reciprocal rank fusion. A paragraph the question names, such as "1026.13(c)(2)" or "Reg E 6(b)", ranks first.
- **Tool** (`app/tools.py`). `dispute_timeline` is plain Python: Reg E unauthorized transfers (1005.6) and errors (1005.11), Reg Z unauthorized use (1026.12(b)) and billing errors (1026.13). It returns deadlines, the liability tiers and whether notice was timely. Business days follow the Federal Reserve's holiday calendar for 2025–2027 (`data/us_federal_holidays.json`), a stated assumption in every result.
- **Status register** (`corpus/status.json`). The eCFR prints two 2024 rules that never took effect: the credit card late-fee rule (vacated by a federal court in April 2025) and the overdraft lending rule (disapproved by Congress in May 2025). The register marks 86 provisions as vacated, nullified or partly not in force, and every passage the model sees says so in its header. The prompt tells the model to decline on them.
- **Citation guard** (`app/citations.py`, `app/answer.py`). Any citation that isn't inside a retrieved passage or a tool result is removed from the answer and flagged. An answer left with no valid citation becomes a decline. The eval scores the model's citations before the guard.
- **One configuration** (`app/config.py`), hashed. Every response, logged run and deployed answer carries the hash: `b2bef7bd2dd0` for everything reported here.

## 3. Data

### Corpus

| | |
|---|---|
| Source | eCFR versioner API, Title 12, Parts 1005 and 1026, as of **2026-10-23** (`corpus/manifest.json`, with SHA-256 of both XML files) |
| Indexed | Reg Z subparts A, B, G and Reg E subpart A, with the official interpretations (Supplement I) for those sections |
| Provisions | 3,400: 1,827 rule paragraphs and 1,573 official interpretations |
| Search chunks | 1,115 (379 rule, 736 interpretation; 783 Reg Z, 332 Reg E), up to 4,000 characters each |
| Size | about 509,000 tokens as embedded, so the corpus doesn't fit in one prompt: search is needed |
| Not in force | 86 provisions in 47 chunks: 10 vacated, 38 nullified, 38 partly not in force |

The parser corrects two eCFR numbering defects against the CFPB's published commentary and stops on anything it can't place. One paragraph the eCFR no longer prints but that is still law (1026.2(a)(15)(ii)(C)) is added as its own chunk with a note. US federal regulations are not under copyright, so the corpus and index are committed.

### Golden set (`eval/golden.toml`)

60 questions, all written and labelled by the author, with no outside reviewer. Labels come only from the corpus text (looked up with `eval/lookup.py`, never with the assistant's search), and deadlines were computed by hand, then cross-checked against the tool.

| | all 60 | dev 24 | test 36 |
|---|---|---|---|
| frontline / compliance | 30 / 30 | 12 / 12 | 18 / 18 |
| answerable / should decline | 40 / 20 | 16 / 8 | 24 / 12 |
| expect a tool call | 14 | 6 | 8 |

- **Answerable types:** deadline calculation 8, Reg E account topics 7, payments and disclosures 6, unauthorized use 5, disputes 5, fees and terms 5, cross-regulation 4.
- **Decline categories:** excluded parts of Z and E 4, network rules 3, other law 3, state law 3, not in force 3, issuer's terms 2, compliance sign-off 2.
- **Traps on purpose:** the same section number in both parts (1005.11 vs 1026.11), the same story with a debit or a credit card, and questions naming a vacated rule's figure.
- Each item has a reference answer, key facts, gold paragraph IDs and, where relevant, the expected tool call and its result.

**Split.** Made once, after all 60 existed, and never changed. Questions q001–q010 were pinned to dev because their answers had already been seen while building; the other 50 were shuffled with seed 20261105 inside four groups (answerable or decline × persona), and the first draw meeting the balance rules was kept (`eval/split.py`). All tuning used dev only. The test split ran once at the end, and CI never touches it.

### Blind re-label (self-agreement)

With no second labeller, the check on the answer key is the author labelling again without looking. On 7 November, 3 days after the first labels (the plan said a week), the author re-labelled the 14 test questions from the second batch, seeing only each question, before any final-run answer existed (`eval/relabel/compare.json`):

| field | agreement |
|---|---|
| should the assistant decline? | 13 of 14 (Wilson 95%: 68–99%), kappa 0.84 |
| question type | 10 of 14 (45–88%), kappa 0.66 |
| decline category (both declines) | 4 of 4 |
| regulation (both answerable) | 8 of 9 |
| gold paragraphs: same IDs, or the same paragraphs covered | 1 of 9 (the other 8 overlap: usually a parent paragraph vs its child, or a rule vs its interpretation) |
| key facts mean the same (the author's own comparison) | 4 of 14 |

The decline decision is steady; the exact key facts are not. Most changed key facts only add a fact, but four say something different (q014, q015, q017, q030). Under a rule fixed before the comparison was read, the author corrected seven first labels he judged wrong on re-reading the regulation, in one commit before the final run (`bc20816`), and agreement is reported against the first labels:

- **q011, q013:** gold paragraphs (one missed an interpretation, one listed a parent and its children);
- **q012, q022:** question type (and q022's regulation);
- **q014:** the new-account rule applies (20 business days), so the deadline, the tool arguments, the gold paragraphs and the key facts changed. The question was kept as written; it gives the account-opening date, not the first-deposit date;
- **q015:** one key fact narrowed, with its interpretation added to gold;
- **q017:** a key fact the question's numbers don't settle was rewritten.

Two other changes were left as first labelled (q027, q030). The split's balance rules read the labels as drawn (`eval/split_labels.json`), so the corrections don't move the split; by type it is now slightly uneven (2 of 8 deadline questions in dev).

## 4. Evaluation method

**Code checks** (`eval/run.py`):

- **Search:** hit@1/3/5 and MRR of the first gold paragraph. A gold ID covers itself, the paragraphs under it and the interpretations on any of them.
- **Citations:** precision (cited IDs in a gold family over all cited IDs), and citations that weren't in the passages given.
- **Abstention:** the answer's `abstained` flag against the label.
- **Tool:** the right call with the right arguments, or no call when none is needed.
- **Calculation:** every expected date, and every Reg E liability tier amount, appears in the answer.
- **Errors:** status errors (citing a provision not in force) and regulation errors (citing the other regulation).

**LLM judge** (`eval/judge.py`, Claude Sonnet 4.5, rubric `j2`). It lists each claim in the answer as supported or not by the passages and tool results (never by the reference answer), labels the answer faithful, partially faithful or unfaithful, and marks each key fact covered or not.

**Judge calibration.** The author labelled answers before seeing any verdict; the code refuses to run the judge on a calibration set until its labels are complete and committed.

| | round 1: rubric j1, set 1 (40 answers) | round 2: set 2 (30 fresh answers), j1 | **round 2: set 2, j2** |
|---|---|---|---|
| faithfulness agreement | 19/40 (33–63%) | 18/30 (42–75%) | **21/30 (52–83%)** |
| faithfulness kappa | 0.13 (−0.10 to 0.35) | 0.37 (0.09 to 0.64) | **0.52 (0.25 to 0.77)** |
| key-fact kappa | 0.55 (0.33 to 0.76) | 0.59 (0.31 to 0.82) | **0.56 (0.27 to 0.80)** |

- Kappa intervals are bootstraps over whole questions (2,000 resamples); agreement intervals are Wilson intervals. Formulas were checked against scikit-learn and statsmodels.
- **Round 1** was used to design j2 and is not a clean measure: for 12 of its last 13 items, the author's notes describe a different question than the answer shown. On the first 27 items alone, faithfulness kappa was 0.21 and key-fact kappa 0.78.
- **Round 2** used fresh answers made after j2 was fixed, so it is held out from j2's design. The labelling sheet showed j2's definitions, which favours j2 over j1 on this set.
- **What the judge gets wrong:** of 14 answers j2 called faithful, the author agreed on 7; of 16 it flagged, on 15. The judge misses dropped or bent conditions. **So every faithfulness figure here is an upper bound, and the judge-based error tags (`memory_leak`, `unsupported_claim`) are a lower bound.**

**Error tags.** Each failure gets the first tag that applies, in a fixed order, from the code checks and the judge (`eval/report.py`): `retrieval_miss`, `over_abstention`, `wrong_reg`, `status_error`, `tool_misuse`, `calculation_error`, `memory_leak` (stating a law outside Z and E from memory), `unsupported_claim`, `wrong_citation`, `partial_retrieval`, `incomplete`; for declines `missed_abstention`, `memory_leak`, `unsupported_claim`. **The author then reviewed every automatic primary tag of the final run** and recorded a tag and a note for each (`eval/review/r20261107T174057Z.toml`).

**Rules fixed before the results they judge.** The winner rule for the experiments was fixed before the last experiment ran, the review policy and the reproducibility tolerance before the final run, and the label-correction rule before the re-label comparison was read. No limit was changed after seeing a result.

## 5. Results

Run `r20261107T174057Z`: test split, 36 questions, configuration `b2bef7bd2dd0` (hybrid search, 5 passages, prompt `p2`, tool on, interpretations included), commit `ffe32e6`, judge `j2`, run once with `--final`.

| | n | hit@5 | MRR | citation precision | abstention right | tool right | calculation exact | faithful* | key facts | failed (automatic) |
|---|---|---|---|---|---|---|---|---|---|---|
| **all** | 36 | 16/24 | 0.40 | 48/102 | 33/36 | 30/36 | 6/6 | 19/36 | 37/52 | 27/36 |
| frontline | 18 | 8/12 | 0.44 | 25/43 | 16/18 | 15/18 | 4/4 | 9/18 | 17/26 | 13/18 |
| compliance | 18 | 8/12 | 0.35 | 23/59 | 17/18 | 15/18 | 2/2 | 10/18 | 20/26 | 14/18 |

\* Judge faithfulness, an upper bound (section 4). Judged 36: faithful 19, partially faithful 13, unfaithful 4.

- **Search:** a gold paragraph at rank 1 for 5 of 24 answerable questions, in the top 3 for 15, in the top 5 for 16; 26 of the 49 gold paragraphs retrieved in all.
- **Abstention:** declined 10 of the 12 it should have; answered 23 of the 24 it should have. Of the 11 answers it marked as declined, 10 should have been.
- **Calculations:** all 6 graded calculations had every date (and Reg E tier) right. A Reg Z liability cap isn't graded as a calculation, because it depends on conditions the tool can't check; key facts cover it. The 6 tool errors are 3 unneeded calls (q011, q019, q021) and 3 calls with a wrong or invented argument (q012, q014, q047).
- **Key facts:** 37 of 52 (71%), judged by the LLM judge, whose key-fact kappa with the author was 0.56.
- **Personas:** no difference is claimed between frontline and compliance; every gap is one or two questions.
- **By type** (counts only, 2–6 questions each): reg_e_account failed 5 of 5 and cross_reg 3 of 3; their hit@5 was 2 of 5 and 1 of 3. The full by-type table is printed by `python -m eval.report r20261107T174057Z`.

**How to read "failed".** A question fails when any error tag applies, including a removed citation or one unsupported side sentence. Automatically, 27 of 36 failed; after the author's review, 25 (section 6). Run to run, a tag count moves by about 3 (section 8), so differences of that size mean nothing.

**Two headline columns count something the review saw differently.** The abstention column reads the `abstained` flag. On q026 and q056 (both about the vacated late-fee rule) the flag was false, so they count as missed declines, but the author's review found that both answers did decline in their text and then stated vacated amounts as current (worked example A). The status-error and regulation-error counts (3 and 1) include q027's citation of a nullified Reg Z paragraph, which the review found was cited to say it never took effect.

**Latency and cost.** Measured in the eval harness, which runs the same code as the deployed function: median 4.8 s, 95th percentile 10.8 s per answer. The deployed API adds about 0.6 s, and a cold start on a tool question took 10.2 s end to end against API Gateway's 29 s limit (one measurement). The answer model used 333,286 input and 19,034 output tokens over 36 questions: **$3.57 per 100 queries** at $3 / $15 per million tokens (Anthropic's list prices for Claude Sonnet 4.6 through a global inference profile; not yet confirmed against the AWS bill). The judge and the embeddings aren't counted.

## 6. Error analysis

| primary tag | automatic | after the author's review |
|---|---|---|
| retrieval_miss (no gold paragraph in the passages) | 8 | 8 |
| unsupported_claim | 8 | 8 |
| memory_leak (outside law stated from memory) | 4 | 4 |
| tool_misuse | 3 | 2 |
| wrong_citation | 1 | 2 |
| partial_retrieval | 1 | 1 |
| missed_abstention | 2 | 0 |
| **questions failed** | **27 of 36** | **25 of 36** |

**The review changed 8 of the 27 primary tags.** Two failures turned out not to be failures: q013 (the flagged sentence is supported by a passage) and q058 (both flagged sentences only decline a sign-off). Four moved to `unsupported_claim`:
- q019: a business-day conclusion with no tool result;
- q026 and q056: declines that state vacated amounts as current;
- q041: an outside-law mention that only repeated the question, plus a claim scoped wider than its passage.

One moved to `wrong_citation` (q015, whose flagged sentences only said the passages don't cover a fee), and one to `memory_leak` (q043, which names state unclaimed-property law as governing). Every change and its reason is in the review file.

**Where it fails.**
- **Search is the main weakness.** 8 of 24 answerable questions had no gold paragraph in their passages. When that happens the model usually still answers, from what it remembers. In 3 of the 8 (q012, q039, q051) the answer got there by another route, through the tool or an interpretation that restates the rule, so hit@5 is stricter than answer quality.
- **The model cites from memory.** 7 of 24 answerable answers cited a paragraph that wasn't in their passages; the guard removed every one, but the guard checks citations, not the sentences around them (example B).
- **Declines that say too much.** 10 of 12 declines declined, but after review 4 named or described the outside law as governing (`memory_leak`: state law 2, network rules 1, other law 1), which the assistant can't check (example C).
- **Rules not in force.** Both not-in-force declines stated amounts from a vacated interpretation as the current law, although every such passage carries a "not in force" header (example A).

### Worked example A: q056, a vacated rule's figures stated as current

> *Our card issuer's pricing team wants to reduce its credit-card late fee to $8 next year to comply with a federal rule issued in 2024. Can we rely on that rule's $8 amount?*

The answer said, as the label expects, that the 2024 late-fee rule was vacated and that the team can't rely on the $8 amount, pointing to the passage marked partly not in force. Then it added that the earlier $25 and $35 safe-harbor amounts "remain the operative figures", taking them from passage 2, an official interpretation that is itself part of the vacated rule and is marked so.

- **The code** tagged it a missed decline, because the `abstained` flag was false.
- **The judge** called the answer fully faithful: it accepted the vacated passage as support for a claim about current law.
- **The author's review** caught it: `unsupported_claim`. q026, the frontline version of the same trap, did the same.

What it shows: the status register gets the right text to the model with the right label, and the model still uses vacated text as current law. The judge doesn't check status either, which is part of why its faithful count is an upper bound. A user reading this answer would set fees from figures in a vacated interpretation.

### Worked example B: q033, a wrong answer after a partial search miss

> *A customer sees a $73.40 charge described as "SQ \*BLUEHARBOR" on her credit-card statement. She does not recognize the merchant and has written to us asking what the charge is, but she has not said that the charge is incorrect. Does this inquiry qualify as a billing-error notice?*

Search found one of the two gold paragraphs (the notice rule, 1026.13(b), at rank 2) but not the one the label rests on (1026.13(a)(6), which lists a request for clarification as a billing error). The answer said the letter does **not** qualify, because a notice must allege an error. Its first sentence had cited 1026.13(a) from memory; the guard replaced the citation with "[citation removed]" but left the sentence. The conclusion ended with a citation to an interpretation of 13(b) that was in the passages.

- Judge: unfaithful, 0 of 2 key facts; tags `unsupported_claim`, `wrong_citation`, `partial_retrieval`.

What it shows: the most harmful failure in the run is a confident, cited answer that is wrong when the deciding paragraph isn't retrieved. A frontline colleague following it would not treat the letter as a billing-error notice, which the reference answer says it is. The guard removes invented citations, not the claims they were meant to support.

### Worked example C: q030, a correct decline that states the outside law

> *Our bank is about to charge its standard credit-card interest rate of 24% to a customer who is currently serving on active duty in the US military. The customer opened the credit-card account and incurred the outstanding balance before beginning active-duty service on 1 September 2026. What restrictions apply to the interest rate we can charge?*

The assistant declined, as it should: none of its passages covers servicemembers, and the label puts the question outside Regulations Z and E (category: other law). But on the way it named a statute and stated its rate cap from memory.

- Judge: partially faithful, both key facts covered, the statute sentence unsupported; tag `memory_leak` (confirmed in review).

What it shows: the golden set's decline policy is that a decline says the topic is outside Regulations Z and E and cites what the corpus does say, without naming or describing the outside law, because the assistant can't check that law or know whether it has changed. Here the decline is right and the answer is still graded a failure. Four of the 12 declines did this.

## 7. Experiments

Each ran on the dev split (24 questions), one setting changed at a time. **Winner rule, fixed before the last experiment ran:** keep the default unless the new option is better on that experiment's main metric at p < 0.05 (exact McNemar test). At 24 questions only a large effect could pass, and none did, so **the configuration was never changed**; the results are measurements with their sample sizes.

| # | change from the default | main metric | result | kept |
|---|---|---|---|---|
| 1 | rule text only (no official interpretations) | hit@5, key facts, over-abstention | hit@5 12/16 vs 9/16 (p 0.38), MRR 0.57 vs 0.44, all of the gain on compliance wording; key facts 28/36 vs 24/36; no new over-abstention. On one question it surfaced the overturned overdraft rule's own text and cited it | interpretations in |
| 2 | vector search only (no BM25) | hit@5 by persona | hit@5 11/16 vs 9/16 (p 0.62), but ranked lower: MRR 0.34 vs 0.44 | hybrid |
| 3 | tool off, prompt telling the model to count days itself | calculation exact | 2 of 5 vs 5 of 5; key facts fewer on 5–6 of 16 answerable questions and more on none (p 0.06 and 0.03 against the two baseline runs); 36% fewer input tokens | tool on |
| 4 | prompt with explicit decline rules and 8 examples | abstention, memory leaks in declines | declined 5 of 8 vs 6 and 5 of 8; lost one stable decline; added unneeded tool calls; 29% more input tokens | `p2` |

**The baseline repeated** (same settings, run twice): search identical; 16 of 24 questions changed something in their answer, and the judge's label flipped on 8. That noise is the yardstick for every answer-side difference above.

## 8. Reproducibility

Two checks, with rules fixed before the final run (`eval/repro.py`, result in `eval/final/reproducibility.json`):

1. **Re-grading.** Re-grading the logged answers of both runs reproduces every logged metric exactly, and the report rebuilt from them is identical. **Passed.**
2. **Re-generating.** The same configuration run again on the test split (`r20261107T175037Z`) must give: identical search; every headline count within max(3, 12.5% of its total); no yes/no measure different at McNemar p < 0.05; input tokens within 5%. **Failed on one of the four rules:**
   - search identical (hit@5 16/24 and MRR 0.40 in both): pass;
   - counts within limits (largest move: judge faithful 19 → 22 of 36, limit 5; tool 30 → 31; key facts 37 → 38): pass;
   - smallest McNemar p 0.25: pass;
   - **input tokens 333,286 → 314,532, a 5.6% difference against a 5% limit: fail.**

   21 of 36 questions changed something. The 5% limit was proposed from a dev repeat in which tokens moved 0.04%; the difference here is about 19,000 tokens, roughly two fewer passes of the passages to the model (the re-run made one fewer unneeded tool call). Under the rule fixed in advance, the first run stays the published one and no third run was made.

## 9. Deployment

- **API.** AWS SAM (`template.yaml`), region ap-south-1: API Gateway REST API with a required key, 1 request/s (burst 2) and 50 requests a day per key; Lambda on Python 3.13, 1,024 MB, 29 s timeout. The index ships inside the function package.
- **Request.** `POST /ask` with `{"question": "...", "today": "YYYY-MM-DD"}` (`today` optional). The response holds the answer, citations, tool calls, the abstained flag and reason, retrieved IDs, the corpus date, the disclaimer and the configuration hash. It never includes the answer before the guard.
- **Least privilege.** The function's role can write its own logs, call Titan V2 in-region and call Sonnet 4.6 through the global inference profile, and nothing else (no judge model, no `*` resources). Tests pin the policy.
- **Logs** hold numbers only: latency, tokens, model and tool calls, flags, the configuration hash. Never question or answer text, and AWS error codes without their messages.
- **Smoke test** (`deploy/smoke.py`): no key → 403; a tool question, a rule question and an out-of-scope question → 200; a misspelt field → 400. All five passed after deploy.
- **MCP tool** (`mcp_server/`). One stdio tool, `ask_card_rules(question, today?)`, built with the MCP Python SDK 2.3.0, calling the deployed API. The API URL and key come from `REGZE_API_URL` and `REGZE_API_KEY`, which an MCP app's config must pass by name.
- **CI** (GitHub Actions, no AWS credentials): the full test suite (714 tests) and a dev retrieval check on every push and pull request. The retrieval check uses committed question vectors and fails if any dev question's first gold paragraph ranks lower than in `eval/baseline.json`. It was shown failing once on a deliberately broken search change. Branch protection is off, so a failing check warns but doesn't block a merge.

The deployed API key isn't public. To call it, deploy your own copy (see the README's quick start).

## 10. Limitations

- **One author, no outside review.** The author wrote and labelled every question and reviewed the failures; nobody with US banking experience checked them. Self-agreement on the decline decision is high (13 of 14) and on key facts low (4 of 14 the same).
- **Small samples.** 36 test questions, 24 answerable. Persona slices have 12 answerable questions each and type slices 2–6; no difference between slices is claimed. A difference of about 3 in any tag count is run-to-run noise.
- **The judge.** Faithfulness is an upper bound: the judge agreed with the author on only half of the answers it called faithful, and it misses dropped conditions and, as example A shows, vacated sources. The judge (Sonnet 4.5) and the answer model (Sonnet 4.6) come from the same model family, which may share blind spots.
- **Search on interpretations.** Interpretations make up two thirds of the chunks and crowd rule text out of the top 5 for compliance-worded questions (experiment 1). Retrieval misses are the largest error class.
- **Status.** The status register is hand-checked for the corpus date only. The model can still state vacated text as current (example A).
- **The guard checks citations, not claims** (example B).
- **The `abstained` flag** sometimes disagrees with the answer text. Anything built on the API that relies on the flag alone will mis-handle those answers.
- **Business days** follow the Federal Reserve holiday calendar for 2025–2027; an institution's own business days may differ, and dates outside the table are refused.
- **Scope.** Two regulations, four subparts, one date. Card network rules, state law, issuer terms and statutes are declined, though they often give customers more than the regulation does.
- **Labels and measurement quirks.** The split's by-type balance is slightly uneven after the label corrections. Three dev items carry known quirks (q001 and q003 expect a 60-day date the question doesn't ask for; q006 has no fixed "today"). One test decline's key facts (q059) name the outside law, which the decline policy says not to do; declines' key facts aren't counted in any figure. Prices for the cost figure are list prices, not yet checked against AWS.

## 11. Monitoring

What exists:
- **Every response** carries the configuration hash and the corpus date, so any answer can be traced to the exact code, prompt and regulation text.
- **Function logs** record latency, tokens, tool calls and flags per request (an invalid citation removed, a repaired output, a decline). The rate of `invalid_citations` and declines is the cheapest live signal of drift.
- **CI** re-runs the tests and the dev retrieval check on every change.
- **Corpus refresh:** `python -m ingest.compare` lists every provision that changed between eCFR versions, which is how a new "as of" date would be reviewed, together with a re-check of the status register.

What doesn't: alerting on those rates, any record of live questions (the logs deliberately hold no text), and a scheduled refresh. A production version would need all three, plus periodic re-labelling and re-running of the golden set.

## 12. Next steps

- **Search the rule text first, then attach each rule's own interpretations**, instead of letting interpretations compete for the five places (from experiment 1).
- **Make status binding in code:** if a cited passage is marked not in force, the answer must decline, rather than leaving it to the model.
- **Check claims, not only citations:** drop or flag a sentence whose citation the guard removes.
- **Make the flag agree with the text,** for example by asking for the decision before the answer.
- **A third calibration set** to measure a rubric that checks conditions and source status.
- **An outside reviewer** for the golden set, and more questions per type.
