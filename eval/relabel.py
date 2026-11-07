"""The blind re-label (spec: labeling rule 5; tracker t3-relabel; session 24).

Some days after labelling, Prit labels a set of golden questions again without looking at his first
labels, and the two are compared: how often he agrees with himself, and which items changed. This
catches inconsistent labelling. It can't catch a misreading made the same way both times; the
README says so.

The items: batch 2's 14 test questions (Prit's choice, session 24). Their answers have never been
shown to him, because the test split has not run. Batch 2 was labelled on 4 Nov 2026 and the
re-label sheet made on 7 Nov, a 3-day gap instead of the spec's week (Prit's choice, so the final
run need not wait); the report states the gap.

From the repo root:
    python -m eval.relabel --sheet      write eval/relabel/labels.toml (once) and meta.json
    python -m eval.relabel --check      check the new labels with the golden set's own checks
    python -m eval.relabel --compare    first labels against new ones (refused until all are done and committed);
                                        writes eval/relabel/review.toml the first time, for the key-fact review
    python -m eval.relabel --compare --save   also write eval/relabel/compare.json (needs the review done)

How it stays blind:
- The sheet shows each question's id, persona and text only. No first label, and no pass-1 trap note
  (a decline's note names its category).
- Don't open eval/golden.toml, and don't run python -m eval.golden, eval.report --show or
  eval.judge --show, until the new labels are committed. python -m eval.lookup reads only the
  corpus and is fine.
- --compare refuses until the 14 blocks pass --check and labels.toml and meta.json are committed, so
  the comparison is always against labels made before seeing the first ones.
- eval.run --split test --final refuses until the same is true (final_block), so no test answer
  exists before the re-label is fixed.

What is compared (fields are exact unless said otherwise):
- should_abstain and type over all 14 (with Cohen's kappa); uncertain and today over all 14.
- abstain_category where both labels are declines; regulation where both are answerable.
- gold IDs where both are answerable, by what their families cover in the index (the grader scores
  by family): "same IDs", "same coverage" (different IDs covering the same provisions), "overlap",
  or "no overlap". Agreement = same IDs or same coverage.
- the tool: whether a call is expected (all 14); where both expect one, the arguments and the
  expected result (liability tiers, notice_timely, and the set of deadline dates with their rules).
- key facts: by Prit's own reading in review.toml ("same" = the new facts state what the old ones
  do, wording and order aside; "different" otherwise). Reference answers are shown, not scored.
Agreement is k of n with a Wilson 95% interval. An item "changed" when any compared field differs or
its key facts are marked "different".

The golden set's first labels stay as they are unless Prit decides otherwise before comparing
(session 24); the comparison is always against the labels as they were when the sheet was made
(meta.json holds their fingerprint, and --compare refuses if they have changed since).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from datetime import date
from pathlib import Path

from eval import calibrate as C
from eval import golden as G

IDS = ("q011", "q012", "q013", "q014", "q015", "q017", "q019", "q021", "q022", "q025", "q026", "q027", "q029", "q030")
FIRST_LABELLED = date(2026, 11, 4)         # batch 2's labels (session 16)
RELABEL_DIR = G.REPO / "eval" / "relabel"
LABELS_FILE = RELABEL_DIR / "labels.toml"
META_FILE = RELABEL_DIR / "meta.json"
REVIEW_FILE = RELABEL_DIR / "review.toml"
COMPARE_FILE = RELABEL_DIR / "compare.json"
FACTS = ("same", "different")

# The fields a labelled block may hold on the sheet: the question's own fields plus pass 2's.
SHOWN = ("id", "stage", "persona", "question")


# ================================================================ the sheet

HEADER = f'''\
# Blind re-label (spec: labeling rule 5). Written and labelled by Prit Raithatha.
#
# Label each question below again, from the question and the regulation text only, as you did in
# pass 2 (python -m eval.lookup is fine). Don't open eval/golden.toml, and don't run
# python -m eval.golden, eval.report --show or eval.judge --show, until this file is committed.
# Don't change id, persona or question.
#
# Fields (the same as in pass 2):
#   type              one of {", ".join(G.TYPES)}
#   regulation        "E", "Z" or "both" (leave out for most declines)
#   today             the date the question treats as today, if it says one (unquoted: 2026-11-03)
#   gold_ids          provision IDs exactly as in the index; the narrowest paragraph that fully answers
#   reference_answer  two to four sentences a correct answer would say
#   key_facts         1 to 3 short facts (0 to 3 for a decline)
#   should_abstain    true for a decline; then type = "decline" and abstain_category is one of
#                     {", ".join(G.ABSTAIN_CATEGORIES)}
#   uncertain         true only when unsure
#   expected_tool_call / expected_result   only when the answer needs dispute_timeline; compute the
#                     result by hand first, as before. Put the two tables after the plain fields of
#                     their block. Shape (placeholder values, not a real label):
#
# [question.expected_tool_call]
# name = "dispute_timeline"
# args = {{ regulation = "E", event = "error", notice_date = "2026-11-05", transfer_kind = "standard" }}
#
# [question.expected_result]
# notice_timely = true
# deadlines = [
#   {{ what = "first deadline in your words", date = 2026-11-19, rule = "1005.x(y)(1)" }},
# ]
#
# Check any time with:  python -m eval.relabel --check
'''

BLOCK_FIELDS = '''\
type = ""                   # see the list at the top
# regulation = "E"          # "E", "Z" or "both"; leave out for most declines
# today = 2026-11-03        # only if the question names its own "today"
gold_ids = []
reference_answer = """
"""
key_facts = [""]
# should_abstain = false    # true for a decline, and then also abstain_category
# abstain_category = ""
# uncertain = true          # only when unsure
# Tool call and result, only if the answer needs dispute_timeline.'''


def first_labels(items: list[G.GoldenItem] | None = None) -> dict[str, G.GoldenItem]:
    items = G.load() if items is None else items
    by_id = {i.id: i for i in items}
    missing = [q for q in IDS if q not in by_id]
    if missing:
        raise ValueError(f"not in the golden set: {missing}")
    return {q: by_id[q] for q in IDS}


def fingerprint(first: dict[str, G.GoldenItem]) -> str:
    """SHA-256 over the first labels of the re-label items (everything in their blocks), so a later
    change to any of them is noticed. It says nothing about what the labels are."""
    blob = json.dumps([first[q].model_dump(mode="json") for q in IDS], sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def toml_string(s: str) -> str:
    """A TOML multi-line basic string holding s exactly."""
    esc = s.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    return f'"""\n{esc}"""'


def sheet_text(first: dict[str, G.GoldenItem]) -> str:
    parts = [HEADER]
    for q in IDS:
        item = first[q]
        parts.append(f'\n[[question]]\nid = "{q}"\nstage = "labelled"\npersona = "{item.persona}"\n'
                     f"question = {toml_string(item.question)}\n{BLOCK_FIELDS}\n")
    return "".join(parts)


def write_sheet(*, items=None, labels=LABELS_FILE, meta=META_FILE, made: date | None = None) -> None:
    """Write the sheet and meta.json, once. Refuses if either exists."""
    for f in (labels, meta):
        if f.exists():
            raise FileExistsError(f"{f.name} already exists; the sheet is written once")
    first = first_labels(items)
    made = made or date.today()
    labels.parent.mkdir(parents=True, exist_ok=True)
    labels.write_bytes(sheet_text(first).encode("utf-8"))
    meta.write_bytes((json.dumps({
        "ids": list(IDS), "first_labelled": FIRST_LABELLED.isoformat(), "sheet_made": made.isoformat(),
        "gap_days": (made - FIRST_LABELLED).days, "first_labels_sha256": fingerprint(first),
    }, indent=2) + "\n").encode("utf-8"))


# ================================================================ checking the new labels

def check(labels=LABELS_FILE, first: dict[str, G.GoldenItem] | None = None
          ) -> tuple[dict[str, G.GoldenItem], list[str]]:
    """(the new labels that pass, every problem). A block passes when it has the right id, persona
    and question and passes the golden set's own checks, including the tool cross-check."""
    first = first_labels() if first is None else first
    if not labels.exists():
        return {}, [f"{labels.name} doesn't exist yet (python -m eval.relabel --sheet)"]
    try:
        raw = G.read(labels)
    except (tomllib.TOMLDecodeError, ValueError) as e:
        return {}, [f"{labels.name} can't be read: {e}"]
    problems = []
    ids = [b.get("id") for b in raw]
    if sorted(map(str, ids)) != sorted(IDS):
        problems.append(f"the file must hold exactly these ids once each: {' '.join(IDS)}")
    statuses = G.provision_statuses()
    done = {}
    for block in raw:
        q = block.get("id")
        if q not in first or ids.count(q) > 1:
            continue
        own = []
        for f in ("persona", "question"):
            if block.get(f) != getattr(first[q], f) and not (f == "question" and isinstance(block.get(f), str)
                                                             and block[f].strip() == first[q].question):
                own.append(f"{f} differs from the golden set's; don't change it")
        if block.get("stage") != "labelled":
            own.append('stage must be "labelled"')
        if "split" in block or "trap" in block:
            own.append("split and trap don't belong on the re-label sheet")
        items, parse_problems = G.parse([block])
        own += [p.split(": ", 1)[1] if p.startswith(f"{q}: ") else p for p in parse_problems]
        if items:
            own += G.check_item(items[0], statuses)
            own += [p.split(": ", 1)[1] for p in G.compare_with_tool(items[0])]
        if own:
            problems += [f"{q}: {p}" for p in own]
        else:
            done[q] = items[0]
    return done, problems


def files_dirty(dirty=None, labels=LABELS_FILE, meta=META_FILE) -> bool:
    dirty = dirty or C.git_dirty
    return any(dirty(f) for f in (labels, meta))


def not_ready(labels=LABELS_FILE, meta=META_FILE, dirty=None, first=None) -> str | None:
    """Why the re-label isn't fixed yet, or None: every block must pass and both files be committed."""
    if not labels.exists() or not meta.exists():
        return "the re-label sheet hasn't been made yet (python -m eval.relabel --sheet)"
    done, problems = check(labels, first)
    if problems or len(done) < len(IDS):
        return f"{len(done)} of {len(IDS)} re-labels are done; see python -m eval.relabel --check"
    if files_dirty(dirty, labels, meta):
        return "eval/relabel/labels.toml or meta.json has uncommitted changes: commit them first"
    return None


def final_block(labels=LABELS_FILE, meta=META_FILE, dirty=None, first=None) -> str | None:
    """Why the final test run must wait, or None (the blind re-label comes first; session 24)."""
    why = not_ready(labels, meta, dirty, first)
    if why is None:
        return None
    return ("the blind re-label comes before any test-split answer exists (session 24), and "
            + why + ".")


# ================================================================ comparing

def coverage(gold_ids: list[str], provisions: list[str]) -> frozenset[str]:
    """Every index provision some gold ID's family covers (the grader's rule)."""
    return frozenset(p for p in provisions if any(G.in_family(g, p) for g in gold_ids))


def gold_relation(a: list[str], b: list[str], provisions: list[str]) -> str:
    if sorted(a) == sorted(b):
        return "same IDs"
    ca, cb = coverage(a, provisions), coverage(b, provisions)
    if ca == cb:
        return "same coverage"
    return "overlap" if ca & cb else "no overlap"


def result_key(r: G.ToolResult | None):
    if r is None:
        return None
    return (tuple(sorted(r.liability_tiers)), r.notice_timely,
            tuple(sorted((d.date.isoformat(), d.rule) for d in r.deadlines)))


def args_key(c: G.ToolCall | None):
    return None if c is None else tuple(sorted(G.as_strings(c.args).items()))


def field_rows(old: G.GoldenItem, new: G.GoldenItem, provisions: list[str]) -> dict[str, dict]:
    """Each compared field for one question: {"agree": bool, "old": ..., "new": ...}; a field that
    doesn't apply to this question (e.g. regulation when either label is a decline) is left out."""
    rows = {}

    def put(name, a, b, agree=None):
        rows[name] = {"agree": (a == b) if agree is None else agree, "old": a, "new": b}

    put("should_abstain", bool(old.should_abstain), bool(new.should_abstain))
    put("type", old.type, new.type)
    put("uncertain", old.uncertain, new.uncertain)
    put("today", old.today.isoformat() if old.today else None, new.today.isoformat() if new.today else None)
    if old.should_abstain and new.should_abstain:
        put("abstain_category", old.abstain_category, new.abstain_category)
    if not old.should_abstain and not new.should_abstain:
        put("regulation", old.regulation, new.regulation)
        rel = gold_relation(old.gold_ids, new.gold_ids, provisions)
        rows["gold_ids"] = {"agree": rel in ("same IDs", "same coverage"), "relation": rel,
                            "old": old.gold_ids, "new": new.gold_ids}
    put("tool expected", old.expected_tool_call is not None, new.expected_tool_call is not None)
    if old.expected_tool_call is not None and new.expected_tool_call is not None:
        put("tool arguments", G.as_strings(old.expected_tool_call.args), G.as_strings(new.expected_tool_call.args),
            agree=args_key(old.expected_tool_call) == args_key(new.expected_tool_call))
        put("tool result", old.expected_result.model_dump(mode="json"), new.expected_result.model_dump(mode="json"),
            agree=result_key(old.expected_result) == result_key(new.expected_result))
    return rows


FIELD_ORDER = ("should_abstain", "type", "abstain_category", "regulation", "gold_ids", "today",
               "tool expected", "tool arguments", "tool result", "uncertain")


def compare(first: dict[str, G.GoldenItem], new: dict[str, G.GoldenItem], review: dict[str, dict] | None,
            provisions: list[str]) -> dict:
    per_item = {q: field_rows(first[q], new[q], provisions) for q in IDS}
    fields = {}
    for f in FIELD_ORDER:
        rows = [(q, per_item[q][f]) for q in IDS if f in per_item[q]]
        if not rows:
            continue
        k = sum(r["agree"] for _, r in rows)
        lo, hi = C.wilson(k, len(rows))
        fields[f] = {"agree": k, "n": len(rows), "wilson95": [round(lo, 3), round(hi, 3)],
                     "changed": [q for q, r in rows if not r["agree"]]}
        if f in ("should_abstain", "type"):
            kap = C.kappa([(r["old"], r["new"]) for _, r in rows])
            fields[f]["kappa"] = None if kap is None else round(kap, 3)
    facts = None
    if review is not None:
        same = [q for q in IDS if review[q]["facts"] == "same"]
        lo, hi = C.wilson(len(same), len(IDS))
        facts = {"agree": len(same), "n": len(IDS), "wilson95": [round(lo, 3), round(hi, 3)],
                 "changed": [q for q in IDS if review[q]["facts"] != "same"]}
    changed = [q for q in IDS if any(not r["agree"] for r in per_item[q].values())
               or (review is not None and review[q]["facts"] != "same")]
    rel = {}
    for q in IDS:
        if "gold_ids" in per_item[q]:
            r = per_item[q]["gold_ids"]["relation"]
            rel[r] = rel.get(r, 0) + 1
    return {"fields": fields, "key_facts": facts, "gold_relations": rel, "items": per_item,
            "changed": changed, "unchanged": [q for q in IDS if q not in changed]}


# ================================================================ the key-fact review

REVIEW_HEADER = '''\
# Key-fact review for the blind re-label (written by python -m eval.relabel --compare).
# For each question, read your first key facts and your new ones (both are in the --compare output
# above each block, and below as comments) and set facts to
#   "same"       the new facts state what the first ones do (wording and order aside)
#   "different"  otherwise: a fact added, dropped, or saying something else
# and, if useful, a short note. Then run python -m eval.relabel --compare again.
'''


def review_text(first, new) -> str:
    parts = [REVIEW_HEADER]
    for q in IDS:
        old_f = "\n".join(f"#   - {f}" for f in first[q].key_facts) or "#   (none)"
        new_f = "\n".join(f"#   - {f}" for f in new[q].key_facts) or "#   (none)"
        parts.append(f'\n[[item]]\nid = "{q}"\n# first:\n{old_f}\n# new:\n{new_f}\n'
                     f'facts = ""                  # "same" or "different"\nnote = ""\n')
    return "".join(parts)


def read_review(path=REVIEW_FILE) -> tuple[dict[str, dict] | None, list[str]]:
    """(the review if every item is filled in, else None; problems)."""
    if not path.exists():
        return None, []
    try:
        with open(path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        return None, [f"{path.name} can't be read: {e}"]
    rows = data.get("item", [])
    problems = []
    if sorted(str(r.get("id")) for r in rows) != sorted(IDS):
        problems.append(f"{path.name} must hold exactly these ids once each: {' '.join(IDS)}")
    out = {}
    for r in rows:
        if r.get("facts") not in FACTS:
            problems.append(f"{r.get('id')}: facts must be \"same\" or \"different\"")
        else:
            out[r["id"]] = {"facts": r["facts"], "note": str(r.get("note", ""))}
    if problems or len(out) < len(IDS):
        return None, problems
    return out, []


# ================================================================ output

def _pct(lo_hi):
    return f"{lo_hi[0] * 100:.0f}–{lo_hi[1] * 100:.0f}%"


def show(v) -> str:
    if isinstance(v, dict):
        return ", ".join(f"{k} {show(x)}" for k, x in v.items())
    if isinstance(v, list):
        return "[" + ", ".join(map(str, v)) + "]"
    return str(v)


def compare_report(c: dict, meta: dict, first, new, review) -> str:
    lines = [f"Blind re-label: {len(IDS)} questions (batch 2's test split). First labels {meta['first_labelled']}, "
             f"sheet made {meta['sheet_made']}: a {meta['gap_days']}-day gap.", "",
             "Agreement with the first labels (Wilson 95% interval):"]
    for f, s in c["fields"].items():
        extra = f", kappa {s['kappa']}" if "kappa" in s else ""
        extra += "" if f != "gold_ids" else " (" + ", ".join(f"{k} {n}" for k, n in sorted(c["gold_relations"].items())) + ")"
        scope = {"abstain_category": " (both declines)", "regulation": " (both answerable)",
                 "gold_ids": " (both answerable; same IDs or same coverage)",
                 "tool arguments": " (both expect a call)", "tool result": " (both expect a call)"}.get(f, "")
        lines.append(f"  {f + scope:52} {s['agree']}/{s['n']}  {_pct(s['wilson95'])}{extra}")
    if c["key_facts"] is None:
        lines.append(f"  {'key facts (your review)':52} not reviewed yet: fill in eval/relabel/review.toml")
    else:
        s = c["key_facts"]
        lines.append(f"  {'key facts (your review: same meaning)':52} {s['agree']}/{s['n']}  {_pct(s['wilson95'])}")
    lines += ["", f"Unchanged: {', '.join(c['unchanged']) or 'none'}",
              f"Changed:   {', '.join(c['changed']) or 'none'}" + ("" if review else " (key facts not reviewed yet)")]
    lines += ["", "Every question, first label against new (differences marked with !):"]
    for q in IDS:
        rows = c["items"][q]
        lines.append(f"\n{q}  ({first[q].persona}): {first[q].question[:90]}{'…' if len(first[q].question) > 90 else ''}")
        for f in FIELD_ORDER:
            if f not in rows:
                continue
            r = rows[f]
            mark = " " if r["agree"] else "!"
            if r["old"] == r["new"]:
                lines.append(f"  {mark} {f}: {show(r['old'])}")
            else:
                rel = f"  [{r['relation']}]" if f == "gold_ids" else ""
                lines.append(f"  {mark} {f}: first {show(r['old'])}  |  new {show(r['new'])}{rel}")
        mark = " " if not review else (" " if review[q]["facts"] == "same" else "!")
        lines.append(f"  {mark} key facts, first: " + " / ".join(first[q].key_facts or ["(none)"]))
        lines.append(f"    key facts, new:   " + " / ".join(new[q].key_facts or ["(none)"]))
        if review and review[q]["note"]:
            lines.append(f"    your note: {review[q]['note']}")
        lines.append("    reference, first: " + " ".join(first[q].reference_answer.split()))
        lines.append("    reference, new:   " + " ".join(new[q].reference_answer.split()))
    return "\n".join(lines)


# ================================================================ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.relabel", description=__doc__.split("\n\n")[0])
    ap.add_argument("--sheet", action="store_true", help="write eval/relabel/labels.toml and meta.json (once)")
    ap.add_argument("--check", action="store_true", help="check the new labels")
    ap.add_argument("--compare", action="store_true", help="first labels against new (needs them all done and committed)")
    ap.add_argument("--save", action="store_true", help="with --compare: also write eval/relabel/compare.json")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.sheet:
        try:
            write_sheet(labels=LABELS_FILE, meta=META_FILE)
        except FileExistsError as e:
            print(f"Nothing written: {e}.")
            return 2
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        print(f"Wrote eval/relabel/labels.toml ({len(IDS)} questions) and meta.json "
              f"(gap since the first labels: {meta['gap_days']} days).\n"
              "Label every block, then: python -m eval.relabel --check\n"
              "Until the file is committed, don't open eval/golden.toml or run python -m eval.golden.")
        return 0
    if args.check:
        done, problems = check(LABELS_FILE)
        print(f"{len(done)} of {len(IDS)} re-labelled.")
        for p in problems:
            print(f"  {p}")
        if not problems and len(done) == len(IDS):
            print("All done and well-formed: commit eval/relabel/ (labels.toml and meta.json) before --compare.")
        return 1 if problems else 0
    if args.compare:
        if (why := not_ready(LABELS_FILE, META_FILE)):
            print(f"Not compared: {why}.")
            return 2
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        first = first_labels()
        if fingerprint(first) != meta["first_labels_sha256"]:
            print("Not compared: the first labels of these questions in eval/golden.toml have changed since the "
                  "sheet was made, so this would not compare against the labels as they were. "
                  + ("The comparison saved before the change is eval/relabel/compare.json."
                     if COMPARE_FILE.exists() else ""))
            return 2
        new, _ = check(LABELS_FILE)
        review, review_problems = read_review(REVIEW_FILE)
        if not REVIEW_FILE.exists():
            REVIEW_FILE.write_bytes(review_text(first, new).encode("utf-8"))
        provisions = list(G.provision_statuses())
        c = compare(first, new, review, provisions)
        print(compare_report(c, meta, first, new, review))
        if review is None:
            print("\nKey facts: mark each question in eval/relabel/review.toml (\"same\" or \"different\"), "
                  "then run --compare again." + ("\n  " + "\n  ".join(review_problems) if review_problems else ""))
        if args.save:
            if review is None:
                print("\nNot saved: the key-fact review isn't complete yet.")
                return 2
            out = {**meta, "labels_sha256": hashlib.sha256(LABELS_FILE.read_bytes()).hexdigest(),
                   "review": review, **{k: v for k, v in c.items() if k != "items"},
                   "items": {q: {f: r for f, r in rows.items()} for q, rows in c["items"].items()}}
            COMPARE_FILE.write_bytes((json.dumps(out, indent=2, default=str) + "\n").encode("utf-8"))
            print("\nSaved eval/relabel/compare.json")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
