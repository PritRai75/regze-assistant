"""Run the golden set through the assistant and grade the answers by code (spec: Evaluation method,
layer 1; tracker t6-run). The LLM judge (layer 2) comes later, in eval/judge.py.

From the repo root:
    python -m eval.run --split unsplit           every labelled question (until the dev/test split exists)
    python -m eval.run --split dev               the dev split
    python -m eval.run --ids q001 q004           only these questions (any split)
    python -m eval.run --split test --final      the test split: once, at the very end (refused, like any run
                                                 of a test question, until the blind re-label is committed:
                                                 eval/relabel.py, session 24)
    python -m eval.run --k 2 --tool-off          the dev split with weaker settings (judge calibration)
    python -m eval.run --prompt p3               the dev split with another prompt (experiment 4)
    python -m eval.run --tool-off --prompt p4    the tool off, the model counting days itself (experiment 3)
    python -m eval.run --no-interpretations      search the rule text only (experiment 1)
    python -m eval.run --runs                    list the logged runs
    python -m eval.run --regrade RUN_ID          grade a logged run again from its stored answers, no model calls

A real run needs `aws login` (Bedrock for the answers, Titan for the question embeddings). Every run
is logged in runs.db (SQLite, git-ignored): one `runs` row with the git commit, config hash and
settings, and one `results` row per question with the golden item and the full response as JSON,
plus the metrics. --regrade re-scores those stored answers and says whether the numbers match the
logged ones, which is the spec's first reproducibility check.

What is skipped: drafts, and labelled items with uncertain = true (labelling rule 4).

How each metric is graded (choices marked * are Claude's, session 14, open to Prit's push-back):
- Gold IDs match as a family (Prit's choice): a gold ID covers itself, the paragraphs under it and
  the comments on any of them (eval.golden.in_family).
- Retrieval, answerable questions only: the rank of the first retrieved chunk holding a provision in
  a gold family; hit@1/3/5; MRR over the k chunks the model was given (* a gold chunk below rank k
  scores 0); gold recall = gold IDs with any family member retrieved / gold IDs.
- Citations come from the model's raw citation strings, before the citation guard, each read with
  app.citations (g3, session 15): a packed string, a range or a comment sub-point counts as the IDs it
  names. Precision = cited IDs in a gold family / everything cited, over answerable questions the
  assistant answered*; "everything cited" also counts, once each, the strings the reader couldn't
  turn into an index ID (too broad, not in the index, no part, unreadable), since none of them points
  to a provision. Invented = IDs that exist in the index but are in no retrieved chunk and no tool
  result, worked out here rather than taken from the guard. guard_rejected records what the guard
  removed at run time, which depends on the guard version the run used.
- Abstention: the response's `abstained` against `should_abstain`; precision and recall are on
  "should abstain"; declines are also counted by category. An abstention caused by the guard or a
  failure (no_valid_citation, output_invalid, tool_limit) is recorded as its cause.
- Tool calls: no call expected -> correct if none was made. A call expected -> correct if any call has
  the expected name and arguments*, compared after dropping empty arguments, writing dates as
  YYYY-MM-DD and sorting date lists; cycle_closing_dates also match when the call's list starts with
  the label's (g5, Prit's choice, session 18). Outcomes: correct_call, correct_no_call, missed_call,
  unneeded_call, wrong_args (with the differing argument names).
- Calculation exactness, items with expected_result: every expected deadline date must appear in the
  final answer text*, in any common form (2026-11-20, November 20, 2026, 20 November, 11/20/2026; a
  date written without a year counts), and for Regulation E calls the max_50 / max_500 tiers as
  "$50" / "$500". Only the dates are checked, not which deadline each is written for: deadlines
  sharing a date are met by one mention. A Regulation Z tier is a cap that depends on conditions the
  tool can't check (1026.12(b)), not the outcome, so its amount isn't required in the text; the
  final amount is graded by the key facts and the judge (Prit's choice, session 14). Tiers "none"
  and "later_transfers_unprotected" and notice_timely can't be read reliably from prose: they are left
  to the judge, and an item with nothing code-checkable gets None. An abstention is never exact.
  tool_result_right separately records whether a tool result the model got agrees with the label,
  so a wrong argument and a misreported right result show up apart (spec).
- Regulation and status errors, answered questions only: wrong_reg = a cited ID from the other part
  (1026 for a regulation = "E" question, 1005 for "Z"; never for "both"); status_error = a cited ID
  whose status isn't in_force (* counted even if the answer says it isn't in force; the prompt tells
  the assistant to decline those, decision 27).
- Questions with no "today" are asked as of EVAL_TODAY*, logged with the run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from app import citations as cit      # the citation guard's own reader; boto3 is only imported for AWS calls
from app.answer import PROVISION       # how the guard finds IDs in a tool result
from app.prompts import PROMPTS
from eval import golden as G

REPO = Path(__file__).resolve().parents[1]
DB_FILE = REPO / "runs.db"
GRADER_VERSION = "g5"                 # bump when a grading rule changes, so old and new numbers aren't mixed
# g1 first harness; g2 tier amounts in the text for Reg E only (S14); g3 citations read with app.citations (S15);
# g4 bracketed comment sub-points, "-2(ii)", read as the comment (S15); g5 cycle_closing_dates match when the
# call's list starts with the label's (S18)
EVAL_TODAY = date(2026, 11, 3)        # "today" for questions that don't state their own
GUARD_CAUSES = ("no_valid_citation", "output_invalid", "tool_limit")


# ================================================================ the index, as grading needs it

@dataclass(frozen=True)
class IndexView:
    chunk_provisions: dict          # chunk ID -> its provision IDs
    spelling: dict                  # lower-case, no "§"/"comment" -> the index's spelling
    status: dict                    # provision ID -> status


def index_view(path: Path = G.CHUNKS_FILE) -> IndexView:
    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)
    return IndexView(chunk_provisions={c["id"]: c["provision_ids"] for c in chunks},
                     spelling={p.lower(): p for c in chunks for p in c["provision_ids"]},
                     status={p: c["status"] for c in chunks for p in c["provision_ids"]})


# ================================================================ reading dates and amounts in prose

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September",
     "October", "November", "December"], 1)}
MONTH_RE = (r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|"
            r"Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\.?")   # capitalised: "may" the verb isn't a month
DAY_RE = r"(\d{1,2})(?:st|nd|rd|th)?"
YEAR_RE = r"(?:,?\s+(\d{4}))?"
DATE_PATTERNS = [
    ("ymd", re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")),
    ("mdy_num", re.compile(r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4})(?![\d/])")),
    ("month_day", re.compile(r"\b" + MONTH_RE + r"\s+" + DAY_RE + r"\b" + YEAR_RE)),
    ("day_month", re.compile(r"(?<![\d/])" + DAY_RE + r"\s+(?:of\s+)?" + MONTH_RE + r"(?![a-z])" + YEAR_RE)),
]
AMOUNTS = {"max_50": re.compile(r"\$\s?50(?:\.00)?(?!\d|,\d)"),
           "max_500": re.compile(r"\$\s?500(?:\.00)?(?!\d|,\d)")}


def month_number(word: str) -> int:
    return next(n for name, n in MONTHS.items() if name.lower().startswith(word.lower()[:3]))


def dates_in(text: str) -> set[tuple[int | None, int, int]]:
    """Every calendar date written in the text, as (year or None, month, day)."""
    found = set()
    for kind, pattern in DATE_PATTERNS:
        for m in pattern.finditer(text):
            if kind == "ymd":
                y, mo, d = int(m[1]), int(m[2]), int(m[3])
            elif kind == "mdy_num":
                y, mo, d = int(m[3]), int(m[1]), int(m[2])
            elif kind == "month_day":
                y, mo, d = (int(m[3]) if m[3] else None), month_number(m[1]), int(m[2])
            else:
                y, mo, d = (int(m[3]) if m[3] else None), month_number(m[2]), int(m[1])
            if 1 <= mo <= 12 and 1 <= d <= 31:
                found.add((y, mo, d))
    return found


def date_written(when: date, found: set) -> bool:
    return (when.year, when.month, when.day) in found or (None, when.month, when.day) in found


# ================================================================ grading one answer

def norm_args(args: dict) -> dict:
    """Tool arguments in one comparable form: no empty values, dates as strings, date lists sorted."""
    out = {}
    for k, v in G.as_strings(args or {}).items():
        if v is None or v == [] or v == "":
            continue
        out[k] = sorted(v) if isinstance(v, list) else v
    return out


def arg_matches(name: str, want, got) -> bool:
    """One tool argument, label against call (both through norm_args). Equal, except (g5, session 18):
    a call's cycle_closing_dates matches when it starts with the label's list, since later closings
    don't change a Reg Z resolution deadline; a shorter or different list does (the tool then falls
    back to the 90-day limit). A label without the list still needs a call without it."""
    if name == "cycle_closing_dates" and want and got:
        return got[:len(want)] == want
    return want == got


def grade_tool(item: G.GoldenItem, calls: list[dict]) -> dict:
    exp = item.expected_tool_call
    m = {"tool_calls": len(calls), "tool_errors": sum("error" in (c.get("result") or {}) for c in calls),
         "arg_diffs": []}
    if exp is None:
        m["tool_outcome"] = "correct_no_call" if not calls else "unneeded_call"
    elif not calls:
        m["tool_outcome"] = "missed_call"
    else:
        want = norm_args(exp.args)
        diffs = []
        for c in calls:
            got = norm_args(c.get("args"))
            names = sorted(k for k in set(want) | set(got) if not arg_matches(k, want.get(k), got.get(k)))
            diffs.append(names + (["name"] if c.get("name") != exp.name else []))
        best = min(diffs, key=len)
        m["tool_outcome"] = "correct_call" if not best else "wrong_args"
        m["arg_diffs"] = best
    m["tool_correct"] = m["tool_outcome"] in ("correct_call", "correct_no_call")
    return m


def grade_calc(item: G.GoldenItem, resp: dict) -> dict:
    want = item.expected_result
    if want is None:
        return {"calc_checks": 0, "calc_missing": [], "calc_exact": None, "tool_result_right": None}
    found = dates_in(resp["answer"])
    checks, missing = 0, []
    for d in want.deadlines:
        checks += 1
        if not date_written(d.date, found):
            missing.append(f"{d.rule} {d.date.isoformat()}")
    reg_e = (item.expected_tool_call.args.get("regulation") if item.expected_tool_call else None) == "E"
    for tier in (want.liability_tiers if reg_e else []):          # a Reg Z tier is a cap, not the outcome
        if tier in AMOUNTS:
            checks += 1
            if not AMOUNTS[tier].search(resp["answer"]):
                missing.append(f"tier {tier}")
    exact = None if checks == 0 else (not resp["abstained"] and not missing)
    results = [c.get("result") or {} for c in resp["tool_calls"] if c.get("name") == "dispute_timeline"]
    right = any(not G.compare_result(want, r) for r in results) if results else False
    return {"calc_checks": checks, "calc_missing": missing, "calc_exact": exact, "tool_result_right": right}


def retrieval_metrics(gold: list[str], retrieved: list[str], ix: IndexView) -> dict:
    """Rank of the first retrieved chunk holding a provision in a gold family, hits, reciprocal rank,
    and gold recall. Shared with the CI retrieval check (eval/retrieval.py)."""
    def holds_gold(chunk_id):
        return any(G.in_family(g, p) for g in gold for p in ix.chunk_provisions.get(chunk_id, []))
    rank = next((r for r, cid in enumerate(retrieved, 1) if holds_gold(cid)), None)
    in_retrieved = [p for cid in retrieved for p in ix.chunk_provisions.get(cid, [])]
    return {"k": len(retrieved), "first_gold_rank": rank,
            "hit_at_1": rank is not None and rank <= 1, "hit_at_3": rank is not None and rank <= 3,
            "hit_at_5": rank is not None and rank <= 5, "rr": 1 / rank if rank else 0.0,
            "gold_total": len(gold),
            "gold_found": sum(any(G.in_family(g, p) for p in in_retrieved) for g in gold)}


def grade(item: G.GoldenItem, resp: dict, ix: IndexView) -> dict:
    """Every code-graded metric for one question. `resp` is an app.schema.Response as a dict."""
    answered = not resp["abstained"]
    gold = item.gold_ids or []
    m = {"answerable": not item.should_abstain, "should_abstain": item.should_abstain,
         "abstained": resp["abstained"], "abstain_category": item.abstain_category}

    # abstention
    if item.should_abstain:
        m["abstain_outcome"] = "correct_decline" if resp["abstained"] else "missed_abstention"
    else:
        m["abstain_outcome"] = "over_abstention" if resp["abstained"] else "correct_answer"
    cause = next((f for f in resp["flags"] if f in GUARD_CAUSES), None)
    m["abstain_cause"] = (cause or "model") if resp["abstained"] else None

    # retrieval
    retrieved = resp["retrieved_ids"]
    in_retrieved = [p for cid in retrieved for p in ix.chunk_provisions.get(cid, [])]
    m.update(retrieval_metrics(gold, retrieved, ix))

    # citations (raw strings, before the guard, read with the guard's reader)
    known, readings = cit.read_all([c for c in resp["raw_citations"] if c.strip()], ix.spelling)
    kinds = {k: list(dict.fromkeys(x for rd in readings for x in getattr(rd, k)))
             for k in ("too_broad", "not_found", "no_part")}
    unreadable = list(dict.fromkeys(rd.written for rd in readings if rd.unreadable))
    from_tools = {p.lower() for call in resp["tool_calls"] for p in PROVISION.findall(json.dumps(call["result"]))}
    supported = {p.lower() for p in in_retrieved} | from_tools
    m.update({"cited": len(known) + sum(map(len, kinds.values())) + len(unreadable),
              "cited_gold": sum(any(G.in_family(g, p) for g in gold) for p in known),
              "invented": sum(p.lower() not in supported for p in known),
              "invented_ids": [p for p in known if p.lower() not in supported],
              "cited_not_in_index": len(kinds["not_found"]), "cited_too_broad": len(kinds["too_broad"]),
              "cited_no_part": len(kinds["no_part"]), "cited_unreadable": len(unreadable),
              "unread_citations": kinds["too_broad"] + kinds["not_found"] + kinds["no_part"] + unreadable,
              "guard_rejected": len(resp["invalid_citations"])})

    # regulation and status
    other = {"E": "1026.", "Z": "1005."}.get(item.regulation or "")
    m["wrong_reg_ids"] = [p for p in known if other and p.startswith(other)] if answered else []
    m["not_in_force_ids"] = [p for p in known if ix.status.get(p) != "in_force"] if answered else []
    m["wrong_reg"], m["status_error"] = bool(m["wrong_reg_ids"]), bool(m["not_in_force_ids"])

    m.update(grade_tool(item, resp["tool_calls"]))
    m.update(grade_calc(item, resp))
    m.update({"latency_ms": resp["latency_ms"], "input_tokens": resp["usage"].get("inputTokens", 0),
              "output_tokens": resp["usage"].get("outputTokens", 0), "flags": resp["flags"]})
    return m


# ================================================================ summarising many

def share(n: int, of: int) -> str:
    return f"{n} of {of}" + (f" ({100 * n / of:.0f}%)" if of else "")


def tally(ms: list[dict]) -> dict:
    """Counts for one slice. Every rate is kept as (numerator, denominator) so reports show both."""
    ans = [m for m in ms if m["answerable"]]
    answered = [m for m in ans if not m["abstained"]]
    should = [m for m in ms if m["should_abstain"]]
    did = [m for m in ms if m["abstained"]]
    calc = [m for m in ms if m["calc_exact"] is not None]
    lat = sorted(m["latency_ms"] for m in ms)
    return {
        "n": len(ms), "answerable": len(ans), "declines": len(should),
        "hit_at_1": (sum(m["hit_at_1"] for m in ans), len(ans)),
        "hit_at_3": (sum(m["hit_at_3"] for m in ans), len(ans)),
        "hit_at_5": (sum(m["hit_at_5"] for m in ans), len(ans)),
        "mrr": round(statistics.fmean(m["rr"] for m in ans), 3) if ans else None,
        "gold_recall": (sum(m["gold_found"] for m in ans), sum(m["gold_total"] for m in ans)),
        "citation_precision": (sum(m["cited_gold"] for m in answered), sum(m["cited"] for m in answered)),
        "invented": sum(m["invented"] for m in ms),
        "cited_not_in_index": sum(m["cited_not_in_index"] for m in ms),
        "cited_too_broad": sum(m.get("cited_too_broad", 0) for m in ms),
        "cited_no_part": sum(m.get("cited_no_part", 0) for m in ms),
        "cited_unreadable": sum(m.get("cited_unreadable", 0) for m in ms),
        "guard_rejected": sum(m.get("guard_rejected", 0) for m in ms),
        "abstain_precision": (sum(m["should_abstain"] for m in did), len(did)),
        "abstain_recall": (sum(m["abstained"] for m in should), len(should)),
        "over_abstention": sum(m["abstain_outcome"] == "over_abstention" for m in ms),
        "missed_abstention": sum(m["abstain_outcome"] == "missed_abstention" for m in ms),
        "tool_correct": (sum(m["tool_correct"] for m in ms), len(ms)),
        "calc_exact": (sum(bool(m["calc_exact"]) for m in calc), len(calc)),
        "wrong_reg": sum(m["wrong_reg"] for m in ms),
        "status_error": sum(m["status_error"] for m in ms),
        "latency_p50_ms": statistics.median(lat) if lat else None,
        "latency_p95_ms": lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))] if lat else None,
        "input_tokens": sum(m["input_tokens"] for m in ms), "output_tokens": sum(m["output_tokens"] for m in ms),
    }


def summarise(rows: list[tuple[G.GoldenItem, dict | None, str | None]]) -> dict:
    """rows: (item, metrics or None, error or None). Overall, by persona, by type, declines by category."""
    graded = [(i, m) for i, m, e in rows if m is not None]
    ms = [m for _, m in graded]
    out = {"overall": tally(ms), "errors": [(i.id, e) for i, m, e in rows if e],
           "by_persona": {}, "by_type": {}, "declines_by_category": {}, "tool_outcomes": {}}
    for field, key in (("persona", "by_persona"), ("type", "by_type")):
        for value in sorted({getattr(i, field) for i, _ in graded}):
            out[key][value] = tally([m for i, m in graded if getattr(i, field) == value])
    for cat in sorted({m["abstain_category"] for m in ms if m["should_abstain"]}):
        group = [m for m in ms if m["should_abstain"] and m["abstain_category"] == cat]
        out["declines_by_category"][cat] = (sum(m["abstained"] for m in group), len(group))
    out["tool_outcomes"] = dict(sorted(Counter(m["tool_outcome"] for m in ms).items()))
    return out


def report(s: dict) -> str:
    o = s["overall"]
    lines = [f"{o['n']} questions graded: {o['answerable']} answerable, {o['declines']} declines"
             + (f"; {len(s['errors'])} failed to run" if s["errors"] else "")]
    if o["answerable"]:
        lines += ["", "Retrieval (answerable questions)",
                  f"  hit@1 {share(*o['hit_at_1'])}   hit@3 {share(*o['hit_at_3'])}   hit@5 {share(*o['hit_at_5'])}",
                  f"  MRR {o['mrr']}   gold IDs retrieved {share(*o['gold_recall'])}"]
    lines += ["", "Citations (raw, before the guard)",
              f"  precision {share(*o['citation_precision'])} of cited IDs in a gold family (answered questions)",
              f"  invented (real IDs, not retrieved or in a tool result) {o['invented']}",
              f"  not read as an index ID: not in the index {o['cited_not_in_index']}, too broad (a whole "
              f"section) {o['cited_too_broad']}, no part {o['cited_no_part']}, unreadable {o['cited_unreadable']}",
              f"  removed by the guard at run time {o['guard_rejected']}",
              "", "Abstention",
              f"  precision {share(*o['abstain_precision'])}   recall {share(*o['abstain_recall'])}",
              f"  over-abstentions {o['over_abstention']}   missed abstentions {o['missed_abstention']}"]
    if s["declines_by_category"]:
        lines.append("  declined, by category: " + ", ".join(
            f"{k} {n}/{d}" for k, (n, d) in s["declines_by_category"].items()))
    lines += ["", "Tool and calculation",
              f"  tool use correct {share(*o['tool_correct'])}: " + ", ".join(f"{k} {v}" for k, v in s["tool_outcomes"].items()),
              f"  calculation exact {share(*o['calc_exact'])} (items with code-checkable dates or tiers)",
              "", f"Regulation errors {o['wrong_reg']}   status errors {o['status_error']}",
              f"Latency p50 {o['latency_p50_ms']} ms, p95 {o['latency_p95_ms']} ms   "
              f"tokens in {o['input_tokens']}, out {o['output_tokens']}"]
    for key, title in (("by_persona", "By persona"), ("by_type", "By type (counts only; too few to compare)")):
        if s[key]:
            lines += ["", title]
            for name, t in s[key].items():
                bits = [f"n {t['n']}"]
                if t["answerable"]:
                    bits.append(f"hit@5 {t['hit_at_5'][0]}/{t['hit_at_5'][1]}")
                    bits.append(f"cit. prec. {t['citation_precision'][0]}/{t['citation_precision'][1]}")
                bits += [f"over-abst. {t['over_abstention']}", f"missed abst. {t['missed_abstention']}",
                         f"tool ok {t['tool_correct'][0]}/{t['tool_correct'][1]}"]
                if t["calc_exact"][1]:
                    bits.append(f"calc {t['calc_exact'][0]}/{t['calc_exact'][1]}")
                lines.append(f"  {name:22} " + ", ".join(bits))
    if s["errors"]:
        lines += ["", "Failed to run:"] + [f"  {q}: {e}" for q, e in s["errors"]]
    return "\n".join(lines)


# ================================================================ choosing and running questions

def select(items: list[G.GoldenItem], split: str | None, ids: list[str] | None) -> tuple[list, list[str]]:
    """The items to run, and notes on what was skipped."""
    notes = []
    if ids:
        known = {i.id for i in items}
        if (missing := [q for q in ids if q not in known]):
            raise SystemExit(f"not in the golden set: {', '.join(missing)}")
        chosen = [i for i in items if i.id in ids]
    else:
        want = None if split == "unsplit" else split
        chosen = [i for i in items if i.split == want]
    drafts = [i.id for i in chosen if i.stage == "draft"]
    unsure = [i.id for i in chosen if i.stage == "labelled" and i.uncertain]
    if drafts:
        notes.append(f"skipped {len(drafts)} draft(s): {', '.join(drafts)}")
    if unsure:
        notes.append(f"skipped {len(unsure)} uncertain: {', '.join(unsure)}")
    return [i for i in chosen if i.stage == "labelled" and not i.uncertain], notes


def answer_items(items, config, *, client=None, embed=None, today=EVAL_TODAY, pause=0.0, on_result=None):
    """Answer each item with the assistant. Yields (item, response dict or None, error or None).
    `client` and `embed` stand in for Bedrock and Titan in tests. A failure on the first question
    stops the run (usually a login or setup problem); later failures are recorded and the run goes on."""
    from app.answer import answer
    for n, item in enumerate(items):
        if n and pause:
            time.sleep(pause)
        try:
            vec = embed(item.question) if embed else None
            resp = answer(item.question, item.today or today, config=config, client=client,
                          query_vec=vec).model_dump(mode="json")
            row = (item, resp, None)
        except Exception as e:                  # noqa: BLE001 - logged per question; Ctrl+C still stops
            if n == 0:
                raise
            row = (item, None, f"{type(e).__name__}: {e}")
        if on_result:
            on_result(*row)
        yield row


# ================================================================ runs.db

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, started_at TEXT, finished_at TEXT, git_commit TEXT, git_dirty INTEGER,
    config_hash TEXT, config_json TEXT, prompt_version TEXT, as_of TEXT, split TEXT, ids TEXT,
    golden_sha256 TEXT, today_default TEXT, grader_version TEXT, n_items INTEGER, note TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT, qid TEXT, persona TEXT, type TEXT, split TEXT, item_json TEXT, response_json TEXT,
    error TEXT, metrics_json TEXT, abstained INTEGER, should_abstain INTEGER, hit_at_5 INTEGER,
    first_gold_rank INTEGER, cited INTEGER, cited_gold INTEGER, invented INTEGER, tool_outcome TEXT,
    calc_exact INTEGER, latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER,
    PRIMARY KEY (run_id, qid)
);
"""


def connect(path: Path = DB_FILE) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def git_state() -> tuple[str | None, bool | None]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        return head, bool(dirty)
    except (OSError, subprocess.CalledProcessError):
        return None, None


def log_result(db, run_id, item, resp, error, metrics):
    m = metrics or {}
    db.execute("INSERT INTO results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
        run_id, item.id, item.persona, item.type, item.split, item.model_dump_json(),
        json.dumps(resp) if resp else None, error, json.dumps(metrics) if metrics else None,
        m.get("abstained"), m.get("should_abstain"), m.get("hit_at_5"), m.get("first_gold_rank"),
        m.get("cited"), m.get("cited_gold"), m.get("invented"), m.get("tool_outcome"),
        None if m.get("calc_exact") is None else int(m["calc_exact"]),
        m.get("latency_ms"), m.get("input_tokens"), m.get("output_tokens")))
    db.commit()


def run(items, config, *, db, split=None, ids=None, client=None, embed=None, today=EVAL_TODAY, pause=0.0,
        note="", golden_path: Path = G.GOLDEN_FILE, ix: IndexView | None = None, progress=print) -> tuple[str, dict]:
    """Answer, grade and log every item. Returns (run ID, summary)."""
    from app import retrieve as rt
    from app.config import config_hash
    ix = ix or index_view()
    as_of = rt.load()[0][0]["as_of"]
    started = datetime.now(timezone.utc)
    run_id = started.strftime("r%Y%m%dT%H%M%SZ")
    commit, dirty = git_state()
    db.execute("INSERT INTO runs (run_id, started_at, git_commit, git_dirty, config_hash, config_json, prompt_version,"
               " as_of, split, ids, golden_sha256, today_default, grader_version, n_items, note)"
               " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (run_id, started.isoformat(), commit, None if dirty is None else int(dirty), config_hash(config, as_of),
                json.dumps(asdict(config), sort_keys=True), config.prompt, as_of, split, " ".join(ids or []) or None,
                hashlib.sha256(golden_path.read_bytes()).hexdigest(), today.isoformat(), GRADER_VERSION, len(items), note))
    db.commit()
    rows = []
    for n, (item, resp, error) in enumerate(answer_items(items, config, client=client, embed=embed,
                                                         today=today, pause=pause), 1):
        metrics = grade(item, resp, ix) if resp else None
        log_result(db, run_id, item, resp, error, metrics)
        rows.append((item, metrics, error))
        status = error or (f"{metrics['abstain_outcome']}, hit@5 {metrics['hit_at_5']}, tool {metrics['tool_outcome']}"
                           + (f", calc {metrics['calc_exact']}" if metrics["calc_exact"] is not None else ""))
        progress(f"  [{n}/{len(items)}] {item.id}: {status}")
    summary = summarise(rows)
    db.execute("UPDATE runs SET finished_at = ?, summary_json = ? WHERE run_id = ?",
               (datetime.now(timezone.utc).isoformat(), json.dumps(summary), run_id))
    db.commit()
    return run_id, summary


def metric_changes(old: dict, new: dict) -> list[str]:
    """How one question's metrics changed: "key: old -> new" for changed values, then any added or
    dropped keys by name (a newer grader version adds fields)."""
    lines = [f"{k}: {old[k]!r} -> {new[k]!r}" for k in new if k in old and old[k] != new[k]]
    added, dropped = [k for k in new if k not in old], [k for k in old if k not in new]
    if added:
        lines.append("new fields: " + ", ".join(added))
    if dropped:
        lines.append("dropped fields: " + ", ".join(dropped))
    return lines


def regrade(db, run_id: str, ix: IndexView | None = None) -> tuple[dict, list[tuple[str, list[str]]], str]:
    """Grade a logged run again from its stored items and answers. Returns (summary, [(question, its
    metric changes)] for the questions whose metrics differ from the logged ones, the grader version
    the run was logged with)."""
    ix = ix or index_view()
    head = db.execute("SELECT grader_version FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if head is None:
        raise SystemExit(f"no run {run_id!r} in the database; see python -m eval.run --runs")
    rows, changed = [], []
    for qid, item_json, resp_json, error, metrics_json in db.execute(
            "SELECT qid, item_json, response_json, error, metrics_json FROM results WHERE run_id = ? ORDER BY qid", (run_id,)):
        item = G.GoldenItem.model_validate_json(item_json)
        if resp_json is None:
            rows.append((item, None, error))
            continue
        metrics = grade(item, json.loads(resp_json), ix)
        new, old = json.loads(json.dumps(metrics)), json.loads(metrics_json)
        if new != old:
            changed.append((qid, metric_changes(old, new)))
        rows.append((item, metrics, None))
    return summarise(rows), changed, head[0]


def list_runs(db) -> str:
    rows = db.execute("SELECT run_id, split, ids, n_items, config_hash, git_commit, git_dirty, finished_at, note "
                      "FROM runs ORDER BY run_id").fetchall()
    if not rows:
        return "No runs logged yet."
    out = []
    for run_id, split, ids, n, chash, commit, dirty, finished, note in rows:
        what = ids or split or "?"
        state = "finished" if finished else "unfinished"
        out.append(f"{run_id}  {what:10} {n:3} items  config {chash}  commit {(commit or '?')[:7]}"
                   f"{' (uncommitted changes)' if dirty else ''}  {state}{'  ' + note if note else ''}")
    return "\n".join(out)


# ================================================================ command line

def config_from_args(args):
    """The default config with --k, --tool-off, --search, --prompt and --no-interpretations applied. All are logged with the run
    (config_json, config_hash), so a weak or experiment run is never mistaken for a default one."""
    from app.config import DEFAULT
    changes = {}
    if getattr(args, "search", None):
        changes["search_mode"] = args.search
    if getattr(args, "prompt", None):
        changes["prompt"] = args.prompt
    if args.k is not None:
        if args.k < 1:
            raise SystemExit("--k must be at least 1")
        changes["k"] = args.k
    if args.tool_off:
        changes["tool_on"] = False
    if getattr(args, "no_interpretations", False):
        changes["interpretations"] = False
    try:
        return DEFAULT.with_(**changes)
    except ValueError as e:                        # e.g. --prompt p4 without --tool-off
        raise SystemExit(str(e)) from None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--split", choices=["dev", "test", "unsplit"], default="dev",
                    help="which questions: a split, or 'unsplit' for labelled ones with no split yet (default dev)")
    ap.add_argument("--ids", nargs="+", metavar="QID", help="only these questions, e.g. q001 q004")
    ap.add_argument("--final", action="store_true", help="required with --split test (run the test split once, at the end)")
    ap.add_argument("--today", type=date.fromisoformat, default=EVAL_TODAY,
                    help=f"'today' for questions that don't state one (default {EVAL_TODAY})")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--k", type=int, metavar="N", help="passages given to the model (default: the config's k)")
    ap.add_argument("--tool-off", action="store_true", help="run without the dispute_timeline tool")
    ap.add_argument("--search", choices=["vector", "bm25", "hybrid"],
                    help="search mode (default: the config's, hybrid; experiment 2 compares them)")
    ap.add_argument("--prompt", choices=sorted(PROMPTS),
                    help="the answer model's prompt (default: the config's, p2; experiment 4 compares p3; "
                         "p4 goes with --tool-off for experiment 3)")
    ap.add_argument("--no-interpretations", action="store_true",
                    help="search the rule text only, without the official interpretations (experiment 1)")
    ap.add_argument("--note", default="", help="a short note saved with the run")
    ap.add_argument("--db", type=Path, default=DB_FILE, help="the SQLite file (default runs.db)")
    ap.add_argument("--runs", action="store_true", help="list the logged runs and stop")
    ap.add_argument("--regrade", metavar="RUN_ID", help="grade a logged run again, no model calls")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to another encoding

    db = connect(args.db)
    if args.runs:
        print(list_runs(db))
        return 0
    if args.regrade:
        summary, changed, version = regrade(db, args.regrade)
        print(report(summary))
        same = "the same as" if not changed else f"different from (in {', '.join(q for q, _ in changed)})"
        note = "" if version == GRADER_VERSION else f" The run was graded with {version}, this is {GRADER_VERSION}."
        if changed:
            print("\nWhat changed, by question:")
            for qid, lines in changed:
                print(f"  {qid}")
                print("\n".join(f"    {line}" for line in lines))
        print(f"\nRe-graded {args.regrade}: metrics are {same} the logged ones.{note}")
        return 0 if not changed else 1
    if args.split == "test" and not args.ids and not args.final:
        print("The test split runs once, at the very end. Add --final if that's now.")
        return 2

    config = config_from_args(args)
    items, notes = select(G.load(), None if args.ids else args.split, args.ids)
    for n in notes:
        print(n)
    if not items:
        hint = " Questions without a split are run with --split unsplit." if args.split == "dev" and not args.ids else ""
        print(f"No labelled questions to run.{hint}")
        return 2
    if any(i.split == "test" for i in items):        # session 24: the blind re-label comes first
        from eval.relabel import final_block
        if (why := final_block()):
            print(f"Not run: {why}")
            return 2
    changed = {k: v for k, v in (("k", args.k), ("tool_on", False if args.tool_off else None),
                                 ("search", args.search), ("prompt", args.prompt),
                                 ("interpretations", False if args.no_interpretations else None)) if v is not None}
    print(f"Running {len(items)} question(s)"
          + (f" with {', '.join(f'{k} = {v}' for k, v in changed.items())}" if changed else "")
          + f"; answers are logged as they come in ({args.db.name}).")
    try:
        run_id, summary = run(items, config, db=db, split=None if args.ids else args.split, ids=args.ids,
                              today=args.today, pause=args.pause, note=args.note)
    except Exception as e:                          # noqa: BLE001 - the first question failed
        print(f"\nThe first question failed, so the run stopped: {type(e).__name__}: {e}")
        print("This is usually the AWS login: run `aws login`, check `aws sts get-caller-identity`, and try again.")
        return 1
    print(f"\nRun {run_id}\n")
    print(report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
