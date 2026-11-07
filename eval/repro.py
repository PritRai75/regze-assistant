"""Both reproducibility checks for the final run (spec: Evaluation method; tracker t9-repro; session 24).

    python -m eval.repro FINAL_RUN SECOND_RUN            both checks; exit 0 = both pass, 1 = a check failed
    python -m eval.repro FINAL_RUN SECOND_RUN --save     also write eval/final/reproducibility.json

FINAL_RUN is the published run of the test split (`eval.run --split test --final`), SECOND_RUN the same
configuration run again on the test split; both judged with the same rubric (default j2). No model calls.

The rule (Prit chose it in session 24, before the final run, from the session 22 dev repeat, where the
largest move was 3 of 24 questions, search was exact and input tokens differed by 0.04%):

Check 1, exact. Re-grading the logged answers reproduces the logged code metrics exactly (eval.run's
regrade), for both runs, and the report built twice from the stored answers and verdicts is identical.

Check 2, the re-run, passes only if all four hold:
  1. search identical: hit@1, hit@3, hit@5, MRR and gold recall equal;
  2. every headline count moves by at most max(3, 12.5% of its total, rounded up): passed (no error tag),
     judge faithful, key facts covered, declines declined, answerable answered, tool correct,
     calculation exact (each out of the questions it applies to in both runs; key facts out of the
     larger of the two totals), status errors and wrong-regulation errors (out of the questions);
  3. no yes/no measure differs measurably: exact McNemar p >= 0.05 for every measure eval.report
     --compare pairs (hit@5, abstention correct, declines declined, answerable answered, declines
     without memory_leak, tool correct, calculation exact, judge faithful, passed);
  4. input tokens within 5% of the first run's.
Latency is shown, not checked: it depends on the network.

Before any verdict, the two runs must be comparable: both on the whole test split, the same config
hash, app/ unchanged between their commits, the same questions with the same labels, every question
answered in both, and every answer judged (or skipped as a guard fallback) with the rubric. Otherwise
nothing is checked (exit 2).

If check 2 fails, the first run's numbers stay the published ones and the README reports the failure
with both runs side by side; there is no third run.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from eval import judge as J
from eval import report as P
from eval import run as R

LIMIT_SHARE = 0.125          # the dev repeat's largest move: 3 of 24
LIMIT_MIN = 3
TOKEN_SHARE = 0.05
ALPHA = 0.05
SEARCH = ("hit_at_1", "hit_at_3", "hit_at_5", "mrr", "gold_recall")
COUNTED = ("passed", "faithful", "declined", "answered", "tool_ok", "calc")    # keys in report.PAIRED
REPRO_FILE = R.REPO / "eval" / "final" / "reproducibility.json"


def limit(total: int) -> int:
    return max(LIMIT_MIN, math.ceil(LIMIT_SHARE * total))


def head(db, run_id: str) -> dict | None:
    row = db.execute("SELECT run_id, split, ids, config_hash, git_commit, git_dirty, grader_version, n_items "
                     "FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return None if row is None else dict(zip(("run_id", "split", "ids", "config_hash", "git_commit", "git_dirty",
                                              "grader_version", "n_items"), row))


def not_comparable(ha, hb, rows_a, rows_b, version: str, app_changed=P.app_changed) -> list[str]:
    """Why the two runs can't be checked against each other (empty: they can)."""
    p = []
    for name, h in (("first", ha), ("second", hb)):
        if h["split"] != "test" or h["ids"]:
            p.append(f"the {name} run ({h['run_id']}) isn't a run of the whole test split")
    if ha["config_hash"] != hb["config_hash"]:
        p.append(f"the config hashes differ ({ha['config_hash']} vs {hb['config_hash']})")
    if app_changed(ha["git_commit"], hb["git_commit"]):
        p.append("app/ changed between the two runs' commits")
    by_a, by_b = {r.item.id: r for r in rows_a}, {r.item.id: r for r in rows_b}
    if set(by_a) != set(by_b):
        p.append("the runs hold different questions: only in the first " + (", ".join(sorted(set(by_a) - set(by_b))) or "none")
                 + "; only in the second " + (", ".join(sorted(set(by_b) - set(by_a))) or "none"))
    relabelled = sorted(q for q in set(by_a) & set(by_b) if by_a[q].item != by_b[q].item)
    if relabelled:
        p.append("labels changed between the runs: " + ", ".join(relabelled))
    for name, rows in (("first", rows_a), ("second", rows_b)):
        failed = [r.item.id for r in rows if r.metrics is None]
        if failed:
            p.append(f"failed to run in the {name} run: {', '.join(failed)} (re-run those with --ids first)")
        unjudged = [r.item.id for r in rows if r.metrics is not None and r.verdict.status in ("missing", "error")]
        if unjudged:
            p.append(f"no {version} verdict in the {name} run for {', '.join(unjudged)} (judge the run first)")
    return p


# ================================================================ check 1

def exact_check(db, run_id: str, version: str, ix) -> dict:
    """Re-grade the stored answers; build the report twice from stored answers and verdicts."""
    _, changed, logged_with = R.regrade(db, run_id, ix)
    first, second = P.build(db, run_id, version, ix=ix), P.build(db, run_id, version, ix=ix)
    same_report = P.text(first) == P.text(second)
    return {"run": run_id, "regrade_same": not changed, "changed": [q for q, _ in changed],
            "logged_with": logged_with, "report_same": same_report, "pass": not changed and same_report}


# ================================================================ check 2

def rerun_check(rows_a, rows_b) -> dict:
    by_b = {r.item.id: r for r in rows_b}
    pairs = [(a, by_b[a.item.id]) for a in rows_a]
    sa, sb = P.stats(rows_a), P.stats(rows_b)

    search = [{"measure": k, "first": sa[k], "second": sb[k], "pass": sa[k] == sb[k]} for k in SEARCH]
    rank_moves = [a.item.id for a, b in pairs if not a.item.should_abstain
                  and (a.metrics["first_gold_rank"], a.metrics["gold_found"]) != (b.metrics["first_gold_rank"], b.metrics["gold_found"])]

    paired = {d["key"]: d for d in P.paired_counts(pairs)}
    counts = []
    for key in COUNTED:
        d = paired[key]
        counts.append({"measure": d["label"], "first": d["a"], "second": d["b"], "total": d["n"],
                       "move": abs(d["a"] - d["b"]), "limit": limit(d["n"])})
    (ka, ta), (kb, tb) = sa["key_facts"], sb["key_facts"]
    counts.append({"measure": "key facts covered", "first": ka, "second": kb, "total": max(ta, tb),
                   "move": abs(ka - kb), "limit": limit(max(ta, tb))})
    for key, label in (("status_error", "status errors"), ("wrong_reg", "wrong-regulation errors")):
        counts.append({"measure": label, "first": sa[key], "second": sb[key], "total": sa["n"],
                       "move": abs(sa[key] - sb[key]), "limit": limit(sa["n"])})
    for c in counts:
        c["pass"] = c["move"] <= c["limit"]

    mcnemar = [{"measure": d["label"], "first": d["a"], "second": d["b"], "total": d["n"],
                "second_only": d["b_only"], "first_only": d["a_only"], "p": d["p"], "pass": d["p"] >= ALPHA}
               for d in paired.values() if d["n"]]

    tin_a, tin_b = sa["input_tokens"], sb["input_tokens"]
    share = abs(tin_b - tin_a) / tin_a if tin_a else float("inf")
    tokens = {"first": tin_a, "second": tin_b, "share": share, "limit": TOKEN_SHARE, "pass": share <= TOKEN_SHARE}
    latency = {"first_p50_ms": sa["latency_p50_ms"], "second_p50_ms": sb["latency_p50_ms"],
               "first_p95_ms": sa["latency_p95_ms"], "second_p95_ms": sb["latency_p95_ms"]}
    parts = {"search": all(s["pass"] for s in search), "counts": all(c["pass"] for c in counts),
             "mcnemar": all(m["pass"] for m in mcnemar), "tokens": tokens["pass"]}
    return {"search": search, "rank_moves": rank_moves, "counts": counts, "mcnemar": mcnemar, "tokens": tokens,
            "latency": latency, "parts": parts, "pass": all(parts.values()),
            "questions_changed": sum(bool(P.what_changed(a, b)) for a, b in pairs), "n": len(pairs)}


# ================================================================ output

def mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def value(v) -> str:
    if isinstance(v, (list, tuple)):
        return f"{v[0]}/{v[1]}"
    return "-" if v is None else str(v)


def text(res: dict) -> str:
    c1, c2 = res["check1"], res["check2"]
    lines = [f"Reproducibility: first run {res['first']} (published), second run {res['second']}; "
             f"judge {res['judge_version']}, grader {res['grader']}, config {res['config_hash']}", "",
             f"Check 1, exact: {mark(all(x['pass'] for x in c1))}"]
    for x in c1:
        lines.append(f"  {x['run']}: re-graded metrics {'the same' if x['regrade_same'] else 'DIFFERENT on ' + ', '.join(x['changed'])}"
                     f" (logged with {x['logged_with']}); report rebuilt {'identical' if x['report_same'] else 'DIFFERENT'}")
    lines += ["", f"Check 2, the re-run: {mark(c2['pass'])}  ({c2['questions_changed']} of {c2['n']} questions changed something)",
              f"  1. search identical: {mark(c2['parts']['search'])}"]
    lines += [f"       {s['measure']:12} {value(s['first']):>8} {value(s['second']):>8}  {mark(s['pass'])}" for s in c2["search"]]
    if c2["rank_moves"]:
        lines.append("       (questions whose first gold rank or gold found moved: " + ", ".join(c2["rank_moves"]) + ")")
    lines.append(f"  2. headline counts within max({LIMIT_MIN}, {LIMIT_SHARE:.1%} of the total): {mark(c2['parts']['counts'])}")
    lines += [f"       {c['measure']:30} {c['first']:>4} -> {c['second']:<4} of {c['total']:<4} move {c['move']}, limit {c['limit']}  {mark(c['pass'])}"
              for c in c2["counts"]]
    lines.append(f"  3. no yes/no measure at McNemar p < {ALPHA}: {mark(c2['parts']['mcnemar'])}")
    lines += [f"       {m['measure']:30} {m['first']:>4} -> {m['second']:<4} of {m['total']:<4} "
              f"second only {m['second_only']}, first only {m['first_only']}, p {m['p']:.2f}  {mark(m['pass'])}"
              for m in c2["mcnemar"]]
    t = c2["tokens"]
    lines.append(f"  4. input tokens within {TOKEN_SHARE:.0%}: {t['first']:,} -> {t['second']:,} ({t['share']:.1%})  {mark(t['pass'])}")
    lat = c2["latency"]
    fmt = lambda ms: "-" if ms is None else f"{ms / 1000:.1f} s"  # noqa: E731
    lines += ["", f"Latency (shown, not checked): p50 {fmt(lat['first_p50_ms'])} -> {fmt(lat['second_p50_ms'])}, "
                  f"p95 {fmt(lat['first_p95_ms'])} -> {fmt(lat['second_p95_ms'])}"]
    if not c2["pass"]:
        lines += ["", "Check 2 failed: the first run's numbers stay the published ones; the README reports this "
                      "failure with both runs side by side. No third run."]
    return "\n".join(lines)


def check(db, run_a: str, run_b: str, version: str = J.JUDGE_VERSION, ix=None, app_changed=P.app_changed) -> dict:
    """Both checks, or {"problems": [...]} when the runs aren't comparable."""
    ix = ix or R.index_view()
    ha, hb = head(db, run_a), head(db, run_b)
    missing = [r for r, h in ((run_a, ha), (run_b, hb)) if h is None]
    if run_a == run_b:
        return {"problems": ["give two different runs"]}
    if missing:
        return {"problems": [f"no run {r} in the database (python -m eval.run --runs)" for r in missing]}
    (_, rows_a), (_, rows_b) = P.load_rows(db, run_a, version, ix), P.load_rows(db, run_b, version, ix)
    problems = not_comparable(ha, hb, rows_a, rows_b, version, app_changed)
    if problems:
        return {"problems": problems}
    c1 = [exact_check(db, r, version, ix) for r in (run_a, run_b)]
    c2 = rerun_check(rows_a, rows_b)
    return {"first": run_a, "second": run_b, "judge_version": version, "grader": R.GRADER_VERSION,
            "config_hash": ha["config_hash"], "check1": c1, "check2": c2,
            "pass": all(x["pass"] for x in c1) and c2["pass"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.repro", description=__doc__.split("\n\n")[0])
    ap.add_argument("first", help="the published final run")
    ap.add_argument("second", help="the same configuration run again on the test split")
    ap.add_argument("--judge", default=J.JUDGE_VERSION, help=f"rubric whose verdicts to use (default {J.JUDGE_VERSION})")
    ap.add_argument("--save", action="store_true", help="also write eval/final/reproducibility.json")
    ap.add_argument("--db", type=Path, default=R.DB_FILE, help="the SQLite file (default runs.db)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if not args.db.exists():
        print(f"Not checked: {args.db} doesn't exist.")
        return 2
    res = check(J.connect(args.db), args.first, args.second, args.judge)
    if "problems" in res:
        print("Not checked; the two runs aren't comparable:\n  " + "\n  ".join(res["problems"]))
        return 2
    print(text(res))
    if args.save:
        REPRO_FILE.parent.mkdir(parents=True, exist_ok=True)
        REPRO_FILE.write_bytes((json.dumps(res, indent=2, default=str) + "\n").encode("utf-8"))
        shown = REPRO_FILE.relative_to(R.REPO).as_posix() if REPRO_FILE.is_relative_to(R.REPO) else REPRO_FILE
        print(f"\nSaved {shown}")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
