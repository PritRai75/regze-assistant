"""The dev/test split (eval/split.py, session 18) and the split field in eval/golden.toml."""
import tomllib

import pytest

from eval import golden as G
from eval import split as S

STATUSES = G.provision_statuses()

# The dev questions as written in session 18. A split never changes: if this test fails, the file
# was edited by mistake. Change this list only through a deliberate commit that says why.
FROZEN_DEV = {"q001", "q002", "q003", "q004", "q005", "q006", "q007", "q008", "q009", "q010",
              "q016", "q018", "q020", "q023", "q024", "q028", "q034", "q038", "q040", "q042",
              "q044", "q049", "q052", "q053"}


@pytest.fixture(scope="module")
def items():
    return G.load()


@pytest.fixture(scope="module")
def drawn(items):
    """The labels as they were when the split was drawn (eval/split_labels.json, session 24): labels may
    be corrected later, and the balance rules judge the split against the labels it was drawn from."""
    return S.as_drawn(items)


# ------------------------------------------------------------------ the committed split

def test_every_question_has_its_frozen_split(items):
    assert all(i.split for i in items)
    assert {i.id for i in items if i.split == "dev"} == FROZEN_DEV
    assert sum(i.split == "test" for i in items) == len(items) - len(FROZEN_DEV) == 36


def test_committed_split_meets_the_rules(drawn):
    assert S.rule_problems(drawn, FROZEN_DEV) == []
    assert set(S.PINNED_DEV) <= FROZEN_DEV                           # batch 1 is all in dev


def test_the_seeded_draw_gives_the_committed_split(drawn):
    """Anyone can re-draw the split from the seed and the split-time labels (eval/split_labels.json).
    This is also what proves the snapshot holds the labels the split was really drawn from."""
    dev, _ = S.draw(drawn)
    assert dev == FROZEN_DEV


def test_snapshot_covers_every_question(items):
    snap = S.drawn_labels()
    assert set(snap) == {i.id for i in items}
    assert all(set(v) == {*S.DRAWN_FIELDS, "tool"} for v in snap.values())


def test_a_corrected_label_never_moves_the_split(items):
    """Session 24's kind of correction: a test question's type changes after the split."""
    fixed = [i.model_copy(update={"type": {"q012": "deadline", "q022": "cross_reg"}[i.id]})
             if i.id in ("q012", "q022") else i for i in items]
    fixed = [i.model_copy(update={"expected_tool_call": None, "expected_result": None}) if i.id == "q012" else i
             for i in fixed]                                                  # and a tool call dropped
    changes = S.label_changes(fixed)
    assert [c for c in changes if c.startswith("q012 type")] and [c for c in changes if c.startswith("q022 type")]
    assert "q012 tool: True -> False" in changes
    again = S.as_drawn(fixed)
    assert S.rule_problems(again, FROZEN_DEV) == [] and S.draw(again)[0] == FROZEN_DEV
    assert [i.model_dump(include={"type", "persona", "abstain_category"}) for i in again] == \
           [i.model_dump(include={"type", "persona", "abstain_category"}) for i in S.as_drawn(items)]


# ------------------------------------------------------------------ the rules

def test_rules_catch_a_moved_question(drawn):
    moved = (FROZEN_DEV - {"q053"}) | {"q013"}                      # compliance dispute for compliance dispute: fine
    assert S.rule_problems(drawn, moved) == []
    unbalanced = (FROZEN_DEV - {"q024"}) | {"q033"}                 # a decline out, an answerable in
    problems = S.rule_problems(drawn, unbalanced)
    assert any("group answerable/frontline" in p for p in problems)
    assert any("group decline/frontline" in p for p in problems)


def test_rules_need_each_category_in_both_splits(drawn):
    no_sign_off = (FROZEN_DEV - {"q028"}) | {"q058"}               # q058 (sign_off, compliance) into dev, q028 out
    assert S.rule_problems(drawn, no_sign_off) == []               # one sign_off still on each side
    both = FROZEN_DEV | {"q058"}
    assert any(p.startswith("sign_off: 2 of 2") for p in S.rule_problems(drawn, both))


def test_draw_is_deterministic_and_seed_dependent(drawn):
    a, n = S.draw(drawn)
    b, m = S.draw(drawn)
    assert (a, n) == (b, m)
    c, _ = S.draw(drawn, seed=S.SEED + 1)
    assert S.rule_problems(drawn, c) == [] and c != a


def test_draw_refuses_drafts_and_uncertain(drawn):
    draft = drawn[0].model_copy(update={"stage": "draft"})
    with pytest.raises(ValueError, match="still a draft"):
        S.draw([draft] + drawn[1:])
    unsure = drawn[0].model_copy(update={"uncertain": True})
    with pytest.raises(ValueError, match="uncertain"):
        S.draw([unsure] + drawn[1:])


def test_a_question_marked_uncertain_later_keeps_its_split(items):
    later = [i.model_copy(update={"uncertain": True}) if i.id == "q014" else i for i in items]
    assert "q014 uncertain: False -> True" in S.label_changes(later)
    assert S.draw(S.as_drawn(later))[0] == FROZEN_DEV


# ------------------------------------------------------------------ writing the field

TWO = '''# header
#   uncertain         true leaves the question out of scoring (default false)
#   expected_tool_call ...

[[question]]
id = "q901"
stage = "labelled"
question = """One?"""

[question.expected_tool_call]
name = "dispute_timeline"

# a comment leading into the next block
[[question]]
id = "q902"   # with a comment
stage = "labelled"
question = """Two?"""
'''


def test_add_splits_inserts_after_each_id_and_documents_the_field():
    new = S.add_splits(TWO, {"q901": "dev", "q902": "test"})
    assert 'id = "q901"\nsplit = "dev"\nstage' in new
    assert 'id = "q902"   # with a comment\nsplit = "test"\nstage' in new
    assert S.DOC_LINE in new and new.index(S.DOC_LINE) < new.index("#   expected_tool_call")
    blocks = tomllib.loads(new)["question"]
    assert [b["split"] for b in blocks] == ["dev", "test"]
    assert blocks[0]["expected_tool_call"] == {"name": "dispute_timeline"}          # the table is untouched
    assert new.replace(S.DOC_LINE + "\n", "").replace('split = "dev"\n', "").replace('split = "test"\n', "") == TWO


def test_add_splits_keeps_windows_line_endings():
    crlf = TWO.replace("\n", "\r\n")
    new = S.add_splits(crlf, {"q901": "dev", "q902": "test"})
    assert "\r\n" in new and "\n" not in new.replace("\r\n", "")
    assert 'id = "q901"\r\nsplit = "dev"\r\n' in new


def test_add_splits_refuses_an_existing_split_or_a_partial_one():
    once = S.add_splits(TWO, {"q901": "dev", "q902": "test"})
    with pytest.raises(ValueError, match="already have a split"):
        S.add_splits(once, {"q901": "dev", "q902": "test"})
    with pytest.raises(ValueError, match="every question"):
        S.add_splits(TWO, {"q901": "dev"})
    with pytest.raises(ValueError, match="every question"):
        S.add_splits(TWO, {"q901": "dev", "q902": "train"})


def test_write_refuses_on_the_committed_file(tmp_path, capsys):
    copy = tmp_path / "golden.toml"
    copy.write_bytes(G.GOLDEN_FILE.read_bytes())
    assert S.main([str(copy), "--write"]) == 1
    assert "already has a split" in capsys.readouterr().out
    assert copy.read_bytes() == G.GOLDEN_FILE.read_bytes()


def test_preview_says_the_file_holds_exactly_this_draw(capsys):
    assert S.main([]) == 0
    out = capsys.readouterr().out
    assert "exactly this draw" in out and "Labels changed since the split" in out


# ------------------------------------------------------------------ the checker

def test_check_wants_a_split_on_every_question_once_one_has_it():
    text = '''
[[question]]
id = "q901"
split = "dev"
stage = "labelled"
persona = "compliance"
type = "reg_e_account"
question = """Under Reg E, what must the opt-in notice say?"""
regulation = "E"
gold_ids = ["1005.17(b)(1)"]
reference_answer = """Two to four sentences."""
key_facts = ["one fact"]
should_abstain = false

[[question]]
id = "q902"
stage = "labelled"
persona = "compliance"
type = "reg_e_account"
question = """Under Reg E, when does the opt-in rule apply?"""
regulation = "E"
gold_ids = ["1005.17(b)(1)"]
reference_answer = """Two to four sentences."""
key_facts = ["one fact"]
should_abstain = false
'''
    items, problems = G.parse(tomllib.loads(text)["question"])
    problems += G.check(items, STATUSES)
    assert problems == ["q902: no split; every question needs one (dev or test) once the split exists"]
    assert "split: dev 1, none 1" in G.summary(items)
