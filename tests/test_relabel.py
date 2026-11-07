"""Tests for eval/relabel.py (the blind re-label, session 24) and the lock in eval.run. No AWS call.
The sheet is filled from the real first labels, so "everything agrees" runs on the 14 real questions;
then single fields are changed to check each kind of disagreement."""
import json
import tomllib
from datetime import date

import pytest

from eval import golden as G
from eval import relabel as L
from eval import run as R

FIRST = L.first_labels()
PROVISIONS = list(G.provision_statuses())


def value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, date):
        return v.isoformat()
    if isinstance(v, list):
        return "[" + ", ".join(value(x) for x in v) + "]"
    return json.dumps(v)                       # a TOML basic string


def block(item: G.GoldenItem) -> str:
    """One filled-in sheet block holding this item's labels (no split, no trap), as Prit would write it."""
    out = ["[[question]]", f'id = "{item.id}"', 'stage = "labelled"', f'persona = "{item.persona}"',
           f"question = {L.toml_string(item.question)}", f'type = "{item.type}"']
    if item.regulation:
        out.append(f'regulation = "{item.regulation}"')
    if item.today:
        out.append(f"today = {item.today.isoformat()}")
    out += [f"gold_ids = {value(item.gold_ids)}", f"reference_answer = {L.toml_string(item.reference_answer)}",
            f"key_facts = {value(item.key_facts)}", f"should_abstain = {value(bool(item.should_abstain))}"]
    if item.abstain_category:
        out.append(f'abstain_category = "{item.abstain_category}"')
    if item.uncertain:
        out.append("uncertain = true")
    if item.expected_tool_call:
        out += ["[question.expected_tool_call]", 'name = "dispute_timeline"',
                "args = { " + ", ".join(f"{k} = {value(v)}" for k, v in item.expected_tool_call.args.items()) + " }"]
        r = item.expected_result
        out += ["[question.expected_result]", f"liability_tiers = {value(r.liability_tiers)}"]
        if r.notice_timely is not None:
            out.append(f"notice_timely = {value(r.notice_timely)}")
        out.append("deadlines = [" + ", ".join(
            f"{{ what = {value(d.what)}, date = {d.date.isoformat()}, rule = {value(d.rule)} }}"
            for d in r.deadlines) + "]")
    return "\n".join(out) + "\n"


def filled(items: dict) -> str:
    return L.HEADER + "\n" + "\n".join(block(items[q]) for q in L.IDS)


@pytest.fixture
def files(tmp_path):
    d = tmp_path / "relabel"
    return {"labels": d / "labels.toml", "meta": d / "meta.json"}


def changed(qid, **fields):
    return FIRST[qid].model_copy(update=fields)


# ------------------------------------------------------------------ the items and the sheet

def test_ids_are_batch_2s_test_questions():
    batch2_test = [i.id for i in G.load() if "q011" <= i.id <= "q030" and i.split == "test"]
    assert list(L.IDS) == batch2_test and len(L.IDS) == 14


def test_sheet_is_blind(files):
    L.write_sheet(**files, made=date(2026, 11, 7))
    text = files["labels"].read_text(encoding="utf-8")
    for item in FIRST.values():
        hidden = [*item.gold_ids, *item.key_facts, " ".join(item.reference_answer.split())[:40]]
        hidden += [item.trap] if item.trap else []
        assert not [h for h in hidden if h and h in text], item.id
        if item.abstain_category:
            assert f'abstain_category = "{item.abstain_category}"' not in text
    raw = G.read(files["labels"])
    assert [b["id"] for b in raw] == list(L.IDS)
    for b in raw:
        assert set(b) == {"id", "stage", "persona", "question", "type", "gold_ids", "reference_answer", "key_facts"}
        assert b["question"] == FIRST[b["id"]].question and b["persona"] == FIRST[b["id"]].persona
        assert b["type"] == "" and b["gold_ids"] == [] and b["key_facts"] == [""]
    meta = json.loads(files["meta"].read_text(encoding="utf-8"))
    assert meta["gap_days"] == 3 and meta["first_labelled"] == "2026-11-04" and meta["ids"] == list(L.IDS)
    assert meta["first_labels_sha256"] == L.fingerprint(FIRST)


def test_sheet_written_once(files):
    L.write_sheet(**files)
    with pytest.raises(FileExistsError):
        L.write_sheet(**files)


def test_fresh_sheet_has_nothing_done(files):
    L.write_sheet(**files)
    done, problems = L.check(files["labels"], FIRST)
    assert done == {} and len({p.split(":")[0] for p in problems}) == 14


def test_question_with_quotes_and_backslash_round_trips():
    s = 'He said "no" """ and C:\\path'
    assert tomllib.loads(f"x = {L.toml_string(s)}")["x"] == s


# ------------------------------------------------------------------ checking

def test_first_labels_written_again_all_pass(files):
    files["labels"].parent.mkdir(parents=True)
    files["labels"].write_text(filled(FIRST), encoding="utf-8")
    done, problems = L.check(files["labels"], FIRST)
    assert problems == [] and list(done) == list(L.IDS)
    for q in L.IDS:
        assert done[q].model_dump(exclude={"split", "trap"}) == FIRST[q].model_dump(exclude={"split", "trap"})


def test_check_catches_changed_question_and_bad_label(files):
    text = filled(FIRST).replace(L.toml_string(FIRST["q015"].question), L.toml_string("A different question?"))
    other = "frontline" if FIRST["q017"].persona == "compliance" else "compliance"
    old17 = block(FIRST["q017"])
    assert old17 in text
    text = text.replace(old17, old17.replace(f'persona = "{FIRST["q017"].persona}"', f'persona = "{other}"'))
    files["labels"].parent.mkdir(parents=True)
    files["labels"].write_text(text, encoding="utf-8")
    done, problems = L.check(files["labels"], FIRST)
    assert any(p.startswith("q015: question differs") for p in problems)
    assert any(p.startswith("q017: persona differs") for p in problems)
    assert "q015" not in done and "q017" not in done and len(done) == 12


def test_check_runs_the_golden_rules(files):
    answerable = next(q for q in L.IDS if not FIRST[q].should_abstain)
    items = {**FIRST, answerable: changed(answerable, gold_ids=["9999.1(a)"])}
    files["labels"].parent.mkdir(parents=True)
    files["labels"].write_text(filled(items), encoding="utf-8")
    done, problems = L.check(files["labels"], FIRST)
    assert any(p.startswith(f"{answerable}: gold_ids:") for p in problems) and answerable not in done


def test_check_runs_the_tool_cross_check(files):
    tool_q = next(q for q in L.IDS if FIRST[q].expected_tool_call)
    r = FIRST[tool_q].expected_result
    wrong = r.model_copy(update={"deadlines": [d.model_copy(update={"date": date(2027, 1, 4)}) for d in r.deadlines]})
    files["labels"].parent.mkdir(parents=True)
    files["labels"].write_text(filled({**FIRST, tool_q: changed(tool_q, expected_result=wrong)}), encoding="utf-8")
    done, problems = L.check(files["labels"], FIRST)
    assert any(p.startswith(f"{tool_q}: deadline") for p in problems) and tool_q not in done


# ------------------------------------------------------------------ the locks

def test_not_ready_until_done_and_committed(files):
    assert "hasn't been made" in L.not_ready(**files, dirty=lambda _: False, first=FIRST)
    L.write_sheet(**files)
    assert "0 of 14" in L.not_ready(**files, dirty=lambda _: False, first=FIRST)
    files["labels"].write_text(filled(FIRST), encoding="utf-8")
    assert "uncommitted" in L.not_ready(**files, dirty=lambda p: p == files["meta"], first=FIRST)
    assert L.not_ready(**files, dirty=lambda _: False, first=FIRST) is None
    assert L.not_ready(**files, dirty=lambda _: None, first=FIRST) is None        # git can't tell: allowed
    assert "re-label comes before" in L.final_block(**files, dirty=lambda _: True, first=FIRST)
    assert L.final_block(**files, dirty=lambda _: False, first=FIRST) is None


def test_run_refuses_test_questions_before_the_relabel(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(L, "final_block", lambda *a, **k: "the blind re-label comes first.")
    assert R.main(["--ids", "q011", "--db", str(tmp_path / "runs.db")]) == 2
    assert "Not run: the blind re-label" in capsys.readouterr().out
    assert R.main(["--split", "test", "--final", "--db", str(tmp_path / "runs.db")]) == 2


def test_run_lock_leaves_dev_runs_alone(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(L, "final_block", lambda *a, **k: "should not be asked")

    def boom(*a, **k):
        raise RuntimeError("no AWS here")
    monkeypatch.setattr(R, "run", boom)
    assert R.main(["--ids", "q001", "--db", str(tmp_path / "runs.db")]) == 1
    out = capsys.readouterr().out
    assert "Not run" not in out and "no AWS here" in out


def test_run_lock_lifts_once_the_relabel_is_committed(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(L, "final_block", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("no AWS here")
    monkeypatch.setattr(R, "run", boom)
    assert R.main(["--split", "test", "--final", "--db", str(tmp_path / "runs.db")]) == 1
    assert "no AWS here" in capsys.readouterr().out


# ------------------------------------------------------------------ comparing

def test_same_labels_agree_everywhere():
    c = L.compare(FIRST, FIRST, None, PROVISIONS)
    assert c["changed"] == [] and c["unchanged"] == list(L.IDS) and c["key_facts"] is None
    for f, s in c["fields"].items():
        assert s["agree"] == s["n"] and s["changed"] == [], f
    assert c["fields"]["should_abstain"]["n"] == 14 and c["fields"]["type"]["n"] == 14
    n_ans = sum(not FIRST[q].should_abstain for q in L.IDS)
    assert c["fields"]["gold_ids"]["n"] == n_ans == c["gold_relations"]["same IDs"]
    assert c["fields"]["tool expected"]["n"] == 14


def test_each_kind_of_change_is_seen():
    ans = [q for q in L.IDS if not FIRST[q].should_abstain]
    dec = [q for q in L.IDS if FIRST[q].should_abstain]
    other_type = next(t for t in G.TYPES if t not in (FIRST[ans[0]].type, "decline"))
    other_cat = next(k for k in G.ABSTAIN_CATEGORIES if k != FIRST[dec[0]].abstain_category)
    new = {**FIRST,
           ans[0]: changed(ans[0], type=other_type),
           dec[0]: changed(dec[0], abstain_category=other_cat),
           ans[1]: changed(ans[1], gold_ids=["1026.13(a)"] if "1026.13(a)" not in FIRST[ans[1]].gold_ids else ["1005.6(b)"]),
           dec[1]: changed(dec[1], should_abstain=False, abstain_category=None, type="dispute", regulation="E",
                           gold_ids=["1005.11(b)"])}
    c = L.compare(FIRST, new, None, PROVISIONS)
    assert ans[0] in c["fields"]["type"]["changed"]
    assert dec[0] in c["fields"]["abstain_category"]["changed"]
    assert dec[1] in c["fields"]["should_abstain"]["changed"] and dec[1] in c["fields"]["type"]["changed"]
    assert dec[1] not in [q for f in ("regulation", "gold_ids", "abstain_category") for q in c["fields"][f]["changed"]]
    assert c["items"][ans[1]]["gold_ids"]["relation"] in ("overlap", "no overlap")
    assert set(c["changed"]) == {ans[0], dec[0], ans[1], dec[1]}
    assert c["fields"]["should_abstain"]["kappa"] < 1


def test_gold_relations():
    assert L.gold_relation(["1005.6(b)"], ["1005.6(b)"], PROVISIONS) == "same IDs"
    sub = next(p for p in PROVISIONS if p.startswith("1005.6(b)(") and "-" not in p)
    assert L.gold_relation(["1005.6(b)"], ["1005.6(b)", sub], PROVISIONS) == "same coverage"
    assert L.gold_relation(["1005.6(b)"], [sub], PROVISIONS) == "overlap"
    assert L.gold_relation(["1005.6(b)"], ["1026.13(a)"], PROVISIONS) == "no overlap"


def test_tool_changes():
    tool_q = next(q for q in L.IDS if FIRST[q].expected_tool_call)
    r = FIRST[tool_q].expected_result
    later = r.model_copy(update={"deadlines": [d.model_copy(update={"date": date(2027, 1, 4)}) for d in r.deadlines]})
    c = L.compare(FIRST, {**FIRST, tool_q: changed(tool_q, expected_result=later)}, None, PROVISIONS)
    assert c["fields"]["tool result"]["changed"] == [tool_q] and c["fields"]["tool arguments"]["changed"] == []
    no_tool = changed(tool_q, expected_tool_call=None, expected_result=None)
    c = L.compare(FIRST, {**FIRST, tool_q: no_tool}, None, PROVISIONS)
    assert c["fields"]["tool expected"]["changed"] == [tool_q] and "tool result" not in c["items"][tool_q]


def test_key_fact_review(tmp_path):
    path = tmp_path / "review.toml"
    path.write_text(L.review_text(FIRST, FIRST), encoding="utf-8")
    assert L.read_review(path) == (None, [f'{q}: facts must be "same" or "different"' for q in L.IDS])
    text = path.read_text(encoding="utf-8").replace('facts = ""', 'facts = "same"')
    text = text.replace(f'id = "{L.IDS[3]}"\n', f'id = "{L.IDS[3]}"\n#\n', 1)
    path.write_text(text, encoding="utf-8")
    review, problems = L.read_review(path)
    assert problems == [] and set(review) == set(L.IDS)
    review[L.IDS[3]]["facts"] = "different"
    c = L.compare(FIRST, FIRST, review, PROVISIONS)
    assert c["key_facts"]["agree"] == 13 and c["changed"] == [L.IDS[3]]


def test_fingerprint_moves_with_any_first_label():
    other = {**FIRST, "q012": changed("q012", key_facts=[*FIRST["q012"].key_facts[:-1], "x"])}
    assert L.fingerprint(other) != L.fingerprint(FIRST)


def test_compare_command(tmp_path, monkeypatch, capsys):
    d = tmp_path / "relabel"
    paths = {"LABELS_FILE": d / "labels.toml", "META_FILE": d / "meta.json",
             "REVIEW_FILE": d / "review.toml", "COMPARE_FILE": d / "compare.json"}
    for k, v in paths.items():
        monkeypatch.setattr(L, k, v)
    monkeypatch.setattr(L.C, "git_dirty", lambda _: False)
    assert L.main(["--compare"]) == 2 and "hasn't been made" in capsys.readouterr().out
    assert L.main(["--sheet"]) == 0
    assert L.main(["--compare"]) == 2 and "0 of 14" in capsys.readouterr().out
    paths["LABELS_FILE"].write_text(filled(FIRST), encoding="utf-8")
    assert L.main(["--check"]) == 0 and "All done" in capsys.readouterr().out
    assert L.main(["--compare", "--save"]) == 2
    out = capsys.readouterr().out
    assert "14/14" in out and "not reviewed yet" in out and "Not saved" in out and paths["REVIEW_FILE"].exists()
    paths["REVIEW_FILE"].write_text(paths["REVIEW_FILE"].read_text(encoding="utf-8")
                                    .replace('facts = ""', 'facts = "same"'), encoding="utf-8")
    assert L.main(["--compare", "--save"]) == 0
    saved = json.loads(paths["COMPARE_FILE"].read_text(encoding="utf-8"))
    assert saved["changed"] == [] and saved["key_facts"]["agree"] == 14 and saved["gap_days"] >= 0
    meta = json.loads(paths["META_FILE"].read_text(encoding="utf-8"))
    paths["META_FILE"].write_text(json.dumps({**meta, "first_labels_sha256": "0" * 64}), encoding="utf-8")
    assert L.main(["--compare"]) == 2 and "have changed since the sheet" in capsys.readouterr().out


def test_compare_refused_while_uncommitted(tmp_path, monkeypatch, capsys):
    d = tmp_path / "relabel"
    monkeypatch.setattr(L, "LABELS_FILE", d / "labels.toml")
    monkeypatch.setattr(L, "META_FILE", d / "meta.json")
    monkeypatch.setattr(L.C, "git_dirty", lambda _: True)
    L.main(["--sheet"])
    (d / "labels.toml").write_text(filled(FIRST), encoding="utf-8")
    capsys.readouterr()
    assert L.main(["--compare"]) == 2 and "uncommitted" in capsys.readouterr().out
