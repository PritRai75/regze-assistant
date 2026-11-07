"""Split the golden set into dev and test (spec: Golden set; tracker t3-rest; session 18).

    python -m eval.split            show the split the rules give, and its spread (changes nothing)
    python -m eval.split --write    write it into eval/golden.toml, once (refused if any split exists)

Dev is for looking, tuning and experiments; test runs once, at the very end, so its numbers are the
honest headline figures. A question never changes split.

How the split is drawn (Prit chose option B, session 18):
1. Batch 1 (q001-q010) goes to dev. Its answers were already seen in sessions 14-15 (and shaped the
   guard and grader there), so none of them may sit in test.
2. The other questions are shuffled with a fixed seed (SEED), inside four groups: answerable and
   frontline, answerable and compliance, decline and frontline, decline and compliance. Dev takes 40%
   of each group (8, 8, 4, 4 of 20, 20, 10, 10), batch 1 counting towards its group.
3. The draw is kept only if these balance rules hold, otherwise the same random sequence draws again:
   - every question type and every decline category has 40% of its questions in dev, rounded down or
     up, and at least one question in each split (`unauthorized` may have 3 of its 5 in dev, because
     batch 1 alone holds 3);
   - 5 or 6 of the 14 questions expecting the tool are in dev.
   Keeping the first draw that passes picks at random among all splits that meet the rules. The rules
   read only labels (persona, type, category, tool), never the assistant's answers.

The same file and the same seed give the same split on any machine; `python -m eval.split` on the
committed file says whether the split in it is exactly that draw. tests/test_split.py pins the dev list.

Labels may be corrected after the split (session 24, after the blind re-label), and a question never
changes split. So the rules are checked against the labels as they were when the split was drawn:
eval/split_labels.json holds them (persona, type, decline or not, category, uncertain, tool expected),
written once in session 24 and never edited. `as_drawn()` restores those fields; the seeded draw on
them must still give the committed split (tested), and `python -m eval.split` lists every label
changed since.
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
import tomllib
from collections import Counter
from pathlib import Path

from eval import golden as G

SEED = 20261105                                          # the date the split was drawn
DEV_SHARE = 0.4                                          # 24 of 60
PINNED_DEV = tuple(f"q{n:03d}" for n in range(1, 11))   # batch 1: answers already seen
TOOL_IN_DEV = (5, 6)                                     # of the 14 questions expecting the tool
CEILING_OVERRIDE = {"unauthorized": 3}                   # batch 1 holds 3 of the 5
MAX_DRAWS = 100_000


DRAWN_FILE = G.REPO / "eval" / "split_labels.json"
DRAWN_FIELDS = ("persona", "type", "should_abstain", "abstain_category", "uncertain")


def drawn_labels(path: Path = DRAWN_FILE) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))["items"] if path.exists() else {}


def as_drawn(items: list[G.GoldenItem], path: Path = DRAWN_FILE) -> list[G.GoldenItem]:
    """The items with the fields the balance rules read set back to their values when the split was
    drawn. A question not in the snapshot is left as it is."""
    snap, out = drawn_labels(path), []
    for i in items:
        s = snap.get(i.id)
        if s is None:
            out.append(i)
            continue
        upd = {f: s[f] for f in DRAWN_FIELDS}
        if s["tool"] != (i.expected_tool_call is not None):
            upd["expected_tool_call"] = G.ToolCall(name="dispute_timeline", args={}) if s["tool"] else None
            upd["expected_result"] = None
        out.append(i.model_copy(update=upd))
    return out


def label_changes(items: list[G.GoldenItem], path: Path = DRAWN_FILE) -> list[str]:
    """One line per field the balance rules read that differs from the split-time snapshot."""
    snap, lines = drawn_labels(path), []
    for i in items:
        s = snap.get(i.id)
        if s is None:
            continue
        now = {f: getattr(i, f) for f in DRAWN_FIELDS} | {"tool": i.expected_tool_call is not None}
        now["should_abstain"] = bool(now["should_abstain"])
        lines += [f"{i.id} {f}: {s[f]} -> {now[f]}" for f in (*DRAWN_FIELDS, "tool") if s[f] != now[f]]
    return lines


def group(item: G.GoldenItem) -> tuple[str, str]:
    return ("decline" if item.should_abstain else "answerable", item.persona)


def kind(item: G.GoldenItem) -> str:
    """The question type, or a decline's category."""
    return item.abstain_category if item.should_abstain else item.type


def dev_sizes(items: list[G.GoldenItem]) -> dict[tuple[str, str], int]:
    sizes = {}
    for g, n in sorted(Counter(group(i) for i in items).items()):
        want = DEV_SHARE * n
        if want != int(want):
            raise ValueError(f"group {g} has {n} questions; 40% of it isn't a whole number")
        sizes[g] = int(want)
    return sizes


def rule_problems(items: list[G.GoldenItem], dev: set[str]) -> list[str]:
    """Every balance rule the split breaks (empty: it passes)."""
    p = []
    for g, n in dev_sizes(items).items():
        got = sum(group(i) == g and i.id in dev for i in items)
        if got != n:
            p.append(f"group {g[0]}/{g[1]}: {got} in dev, needs {n}")
    for k, n in sorted(Counter(kind(i) for i in items).items()):
        got = sum(kind(i) == k and i.id in dev for i in items)
        lo, hi = max(1, math.floor(DEV_SHARE * n)), min(n - 1, CEILING_OVERRIDE.get(k, math.ceil(DEV_SHARE * n)))
        if not lo <= got <= hi:
            p.append(f"{k}: {got} of {n} in dev, needs {lo}" + (f" to {hi}" if hi != lo else ""))
    tool = sum(i.expected_tool_call is not None and i.id in dev for i in items)
    if not TOOL_IN_DEV[0] <= tool <= TOOL_IN_DEV[1]:
        p.append(f"tool: {tool} in dev, needs {TOOL_IN_DEV[0]} to {TOOL_IN_DEV[1]}")
    return p


def usable(items: list[G.GoldenItem]) -> list[str]:
    """Why the file can't be split yet (empty: it can)."""
    p = [f"{i.id}: still a draft" for i in items if i.stage != "labelled"]
    p += [f"{i.id}: uncertain = true" for i in items if i.uncertain]
    ids = {i.id for i in items}
    p += [f"{q}: pinned to dev but not in the file" for q in PINNED_DEV if q not in ids]
    return p


def draw(items: list[G.GoldenItem], seed: int = SEED) -> tuple[set[str], int]:
    """The dev IDs, and which draw of the seeded sequence passed the rules."""
    if (p := usable(items)):
        raise ValueError("can't split yet:\n" + "\n".join(p))
    rng = random.Random(seed)
    sizes = dev_sizes(items)
    groups = {g: sorted(i.id for i in items if group(i) == g) for g in sizes}
    for n in range(1, MAX_DRAWS + 1):
        dev = set()
        for g, ids in groups.items():                   # groups in sorted order: deterministic
            pinned = [q for q in ids if q in PINNED_DEV]
            rest = [q for q in ids if q not in PINNED_DEV]
            rng.shuffle(rest)
            dev |= set(pinned) | set(rest[:sizes[g] - len(pinned)])
        if not rule_problems(items, dev):
            return dev, n
    raise ValueError(f"no split met the rules in {MAX_DRAWS} draws")


def spread(items: list[G.GoldenItem], dev: set[str]) -> str:
    rows = [("answerable", lambda i: not i.should_abstain), ("declines", lambda i: i.should_abstain),
            ("frontline", lambda i: i.persona == "frontline"), ("compliance", lambda i: i.persona == "compliance"),
            ("expect the tool", lambda i: i.expected_tool_call is not None)]
    rows += [(f"regulation {r}", lambda i, r=r: i.regulation == r) for r in ("Z", "E", "both")]
    rows += [("batch 1 (q001-q010)", lambda i: i.id in PINNED_DEV)]
    rows += [(t, lambda i, t=t: not i.should_abstain and i.type == t)
             for t, _ in sorted(Counter(i.type for i in items if not i.should_abstain).items(), key=lambda x: -x[1])]
    rows += [(f"decline: {c}", lambda i, c=c: i.should_abstain and i.abstain_category == c)
             for c, _ in sorted(Counter(i.abstain_category for i in items if i.should_abstain).items(), key=lambda x: -x[1])]
    d = [i for i in items if i.id in dev]
    t = [i for i in items if i.id not in dev]
    lines = [f"  {'':26}{'dev':>4}{'test':>6}", f"  {'all':26}{len(d):>4}{len(t):>6}"]
    lines += [f"  {name:26}{sum(map(f, d)):>4}{sum(map(f, t)):>6}" for name, f in rows]
    return "\n".join(lines)


# ------------------------------------------------------------------ writing it into the file

DOC_AFTER = "#   uncertain"
DOC_LINE = "#   split             dev or test, written once by python -m eval.split --write (never change it)"
_ID_LINE = re.compile(r'^\s*id\s*=\s*"(q\d{3})"\s*(#.*)?$')


def add_splits(text: str, splits: dict[str, str]) -> str:
    """The file's text with `split = "dev"` or `"test"` added after each block's id line.

    Every block must get one, and no block may have one already (a question never changes split).
    Also adds a DOC_LINE to the header, after the line starting with DOC_AFTER, if that line exists.
    Nothing else changes, line endings included. The result is re-read with tomllib and must equal
    the old file plus exactly the split fields, or the change is refused (ValueError, nothing written).
    """
    nl = "\r\n" if "\r\n" in text else "\n"
    lines = text.replace("\r\n", "\n").split("\n")
    old = tomllib.loads(text).get("question", [])
    have = [b.get("id", "?") for b in old if "split" in b]
    if have:
        raise ValueError(f"{len(have)} question(s) already have a split ({', '.join(have[:5])}"
                         f"{', ...' if len(have) > 5 else ''}); a split is written once and never changed")
    ids = [b.get("id") for b in old]
    if sorted(ids) != sorted(splits) or any(v not in ("dev", "test") for v in splits.values()):
        raise ValueError("the split must name every question in the file, each as dev or test")
    starts = [n for n, line in enumerate(lines) if line.strip() == "[[question]]"]
    if len(starts) != len(old):
        raise ValueError(f"found {len(starts)} [[question]] lines but {len(old)} blocks")

    inserts = []
    for k, block in enumerate(old):
        s, e = starts[k], (starts[k + 1] if k + 1 < len(starts) else len(lines))
        id_lines = [n for n in range(s, e) if (m := _ID_LINE.match(lines[n])) and m.group(1) == block["id"]]
        if len(id_lines) != 1:
            raise ValueError(f"{block['id']}: expected one line id = \"{block['id']}\" in its block, found {len(id_lines)}")
        if any(lines[n].lstrip().startswith("[") for n in range(s + 1, id_lines[0])):
            raise ValueError(f"{block['id']}: its id line comes after a [question....] table")
        inserts.append((id_lines[0], f'split = "{splits[block["id"]]}"'))
    doc = [n for n, line in enumerate(lines[:starts[0]] if starts else lines) if line.startswith(DOC_AFTER)]
    if doc and DOC_LINE not in lines:
        inserts.append((doc[0], DOC_LINE))
    for after, line in sorted(inserts, key=lambda x: -x[0]):
        lines.insert(after + 1, line)
    new_text = nl.join(lines)

    expected = [{**b, "split": splits[b["id"]]} for b in old]
    if tomllib.loads(new_text).get("question", []) != expected:
        raise ValueError("the edited file doesn't read back as expected; nothing was changed")
    return new_text


def write(path: Path, items: list[G.GoldenItem], dev: set[str]) -> None:
    text = path.read_bytes().decode("utf-8")
    new_text = add_splits(text, {i.id: "dev" if i.id in dev else "test" for i in items})
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(new_text.encode("utf-8"))
    tmp.replace(path)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m eval.split", description=__doc__.split("\n\n")[0])
    ap.add_argument("path", nargs="?", type=Path, default=G.GOLDEN_FILE, help="default: eval/golden.toml")
    ap.add_argument("--write", action="store_true", help="write the split into the file (once)")
    args = ap.parse_args(argv)
    items = G.load(args.path)
    split_exists = any(i.split for i in items)
    try:
        dev, n = draw(as_drawn(items) if split_exists else items)
    except ValueError as e:
        print(e)
        return 1
    print(f"Seed {SEED}, draw {n} of the sequence: the first that meets the balance rules"
          + (" (on the labels as they were when the split was drawn, eval/split_labels.json)" if split_exists else "")
          + ".\n")
    print(spread(items, dev))
    print("\ndev: " + " ".join(sorted(dev)))
    committed = {i.id for i in items if i.split == "dev"}
    if split_exists:
        same = committed == dev and all(i.split for i in items)
        print("\nThe split in the file is " + ("exactly this draw." if same else "NOT this draw; it stays as "
              "committed (a split never changes), but find out why: was eval/split_labels.json or a split edited?"))
        changes = label_changes(items)
        print("\nLabels changed since the split (the split stays; the table above shows today's labels): "
              + ("none" if not changes else "\n  " + "\n  ".join(changes)))
        if args.write:
            print("\n--write: nothing written, the file already has a split.")
            return 1
        return 0
    if args.write:
        try:
            write(args.path, items, dev)
        except ValueError as e:
            print(f"\nNothing written: {e}")
            return 1
        print(f"\nWritten: {len(dev)} dev, {len(items) - len(dev)} test. Run python -m eval.golden, then pytest.")
    else:
        print("\nNothing written. Add --write to put this split into the file.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
