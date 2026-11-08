# RegZE Assistant

[![CI](https://github.com/PritRai75/regze-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/PritRai75/regze-assistant/actions/workflows/ci.yml)

An internal assistant for staff at a US card issuer and bank. It answers questions about Regulation Z (credit cards) and Regulation E (debit cards and electronic transfers), cites the paragraph it relies on, works out dispute deadlines and liability limits with a deterministic tool, and declines when the regulations don't answer the question.

**Status:** complete, one release (November 2026). Deployed as an API on AWS and as an MCP tool; evaluated once on a held-out test set. The full evaluation is in the **[validation report](docs/validation-report.md)**.

## Why

Contact-centre and compliance staff at a card issuer answer questions every day whose answers sit in these two regulations: how much a customer can lose on a stolen card, how long the bank has to resolve a dispute, whether a fee is allowed. A wrong deadline or liability amount harms the customer and breaks the rule. A general chatbot answers such questions confidently from memory, including from rules that a court has since struck down.

So this assistant answers only from the regulation text as published on one date, cites every claim, leaves date arithmetic to code, and says so when a question is outside the regulations. Because banks validate a model before staff rely on it, the project is evaluated and reported the way a model validation review would want: a held-out test set, a calibrated LLM judge, reviewed failures and the limitations stated plainly.

## Example

A real answer from the deployed API, asked through the MCP tool on 8 November 2026 (`python -m mcp_server.try_it "…" --today 2026-10-26`) and shown exactly as returned. The question is not one of the 60 evaluation questions.

**Question**

> Our customer lost her debit card while hiking on Sunday, 18 October 2026, but only realised it was gone on Tuesday, 20 October. She phoned us on Monday, 26 October, and before that call someone had taken two $200 ATM withdrawals with the card. What is the most she can be held responsible for?

**What the MCP tool returned**

> The customer learned of the loss on October 20 and had until October 22 (two business days later) to notify the bank and keep her liability at $50 or less [§ 1005.6(b)(1)]. She called on October 26, missing that deadline, so the second liability tier applies [§ 1005.6(b)(2); comment 1005.6(b)(2)-1]. Her maximum exposure is the lesser of $500 or the sum of up to $50 for transfers within the first two business days plus the amount of transfers after that period and before her notice that the institution shows timely notice would have prevented [§ 1005.6(b)(2)(i)–(ii)]. Since the total actual loss is $400 (two $200 ATM withdrawals), the most she can be held responsible for is $400 — below the $500 cap — provided the institution establishes the transfers would not have occurred had she notified on time [§ 1005.6(b)(2)(ii)]. If her delay was due to extenuating circumstances, the institution must extend the time limits to a reasonable period, which could lower her liability [§ 1005.6(b)(4)]. No statement date was provided, so the 60-day statement rule was not evaluated and should be checked separately if a statement has been sent [§ 1005.6(b)(3)].
>
> Citations (12 CFR): 1005.6(b)(1), 1005.6(b)(2), 1005.6(b)(2)(i), 1005.6(b)(2)(ii), 1005.6(b)(2)-1, 1005.6(b)(3), 1005.6(b)(4)
>
> Deadline calculation used: `{"event": "unauthorized", "learned_of_loss": "2026-10-20", "notice_date": "2026-10-26", "regulation": "E"}`
>
> Regulations as of 2026-10-23. General information from 12 CFR Parts 1005 and 1026 as of the date shown, not legal or compliance advice. Card network rules, state law and the account terms may give the customer more protection.
>
> (RegZE Assistant, configuration b2bef7bd2dd0)

The two-business-day deadline (22 October) came from the `dispute_timeline` tool, not from the model, and every citation was checked by the guard against the passages it was given. The configuration hash is the one the evaluation measured.

## Results

Held-out test split: 36 questions (24 answerable, 12 that should be declined), written and labelled by the author. One run, `r20261107T174057Z`.

| | |
|---|---|
| A gold paragraph in the top 5 passages | 16 of 24 |
| Declined or answered as it should | 33 of 36 |
| Deadline and liability calculations exactly right | 6 of 6 |
| Reference key facts in the answer | 37 of 52 |
| Fully faithful to its passages (LLM judge, an upper bound) | 19 of 36 |
| Questions with no error at all (after the author's review) | 11 of 36 |
| Latency, median / 95th percentile | 4.8 s / 10.8 s |
| Cost per 100 queries (answer model) | $3.57 |

The tool's calculations were exact on all 6 graded questions. The weak point is search: when it misses, the model fills the gap from memory. The LLM judge was calibrated against the author's own labels (kappa 0.52 on 30 fresh answers) and is lenient, so its faithful count is an upper bound. Every figure, the error analysis and three worked failures are in the [validation report](docs/validation-report.md).

## How it works

```
question -> API Gateway (key, throttle, daily quota) -> AWS Lambda:
              hybrid search over 1,115 chunks of the regulations (vectors + BM25) -> top 5 passages
              -> Claude Sonnet 4.6 on Amazon Bedrock
                   <-> dispute_timeline tool (deadlines and liability tiers, plain Python)
              -> citation guard -> answer with citations, or a decline
MCP server (stdio): one tool, ask_card_rules, calling the API
```

- **Corpus.** Regulation Z subparts A, B and G and Regulation E subpart A, with their official interpretations, from the eCFR API as of 23 October 2026. The text and the search index are committed.
- **Rules not in force.** The eCFR still prints two 2024 rules that never took effect (one vacated by a court, one overturned by Congress). A hand-checked register marks the 86 affected provisions, and every passage carries its status.
- **Citation guard.** A citation that isn't in the passages or a tool result is removed and flagged; an answer left with no valid citation becomes a decline.
- **Evaluation.** 60 questions (24 dev, 36 test), code checks for search, citations, abstention, tool calls and calculations, an LLM judge (Claude Sonnet 4.5) for faithfulness and key facts, and the author's review of every failure. Tuning used dev only; the test split ran once.
- **CI.** 714 tests and a dev search check, with no AWS credentials, on every push.

## What it gets wrong

- **Search misses** are the largest error class: 8 of 24 answerable questions had no gold paragraph in their passages, and the model usually answered anyway, from memory.
- **The citation guard checks citations, not claims.** A wrong answer can keep its sentence after the guard removes its invented citation ([example B](docs/validation-report.md#worked-example-b-q033-a-wrong-answer-after-a-partial-search-miss)).
- **Vacated text stated as current law,** although every such passage is labelled; the judge missed it too ([example A](docs/validation-report.md#worked-example-a-q056-a-vacated-rules-figures-stated-as-current)).
- **Declines that say too much:** 4 of 12 named or described the outside law ([example C](docs/validation-report.md#worked-example-c-q030-a-correct-decline-that-states-the-outside-law)).
- **One author:** every question, label and review is the author's, with no outside reviewer; samples are small (36 test questions).

Next steps, from these results: search the rule text first and attach its interpretations, make a passage's "not in force" status binding in code, and drop sentences whose citation the guard removes. Details in the report's [next steps](docs/validation-report.md#12-next-steps).

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows Git Bash: source .venv/Scripts/activate
pip install -r requirements-dev.txt
pytest -q                          # 714 tests; no AWS needed
python -m eval.retrieval           # the dev search check CI runs, from committed vectors
```

With AWS access to Bedrock (Titan Text Embeddings V2 and Claude Sonnet 4.6 / 4.5 in ap-south-1, through global inference profiles):

```bash
python -m eval.run --split dev                      # answer the dev questions; logs to runs.db
python -m eval.judge RUN_ID                         # RUN_ID is printed as "Run r...Z" when the run ends
python -m eval.report RUN_ID                        # metrics, tags and failures
python -m eval.report RUN_ID --show q004            # one question in full
python -m deploy.stage && sam build && sam deploy   # your own copy of the API (AWS SAM)
```

The deployed API needs a key, which isn't public, so to call it deploy your own copy. `runs.db` is not in the repository, so the published run can't be re-graded from a clone; the golden set, labels, calibration sets, review and reproducibility result are.

## Repository layout

| folder | what it holds |
|---|---|
| `ingest/` | download, parse, chunk, embed, compare eCFR versions |
| `corpus/` | the XML, the manifest, the status register, provisions and chunks |
| `index/` | chunks, vectors and metadata (what search reads) |
| `app/` | search, tool, prompt, answer flow, citation guard, schema, Lambda handler |
| `eval/` | golden set, split, runs, judge, calibration, report, re-label, reproducibility, review |
| `deploy/`, `template.yaml` | staging, smoke test, the SAM template |
| `mcp_server/` | the MCP tool and a test client |
| `docs/` | the validation report |
| `tests/` | 714 tests |

## Disclaimer

General information from 12 CFR Parts 1005 and 1026 as of the date shown, not legal or compliance advice. Card network rules, state law and an issuer's own account terms are out of scope and may give a customer more protection. US federal regulations are not under copyright, so the regulation text is included in this repository.
