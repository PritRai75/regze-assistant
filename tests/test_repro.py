"""Tests for eval/repro.py (both reproducibility checks, session 24). No AWS call: two made-up runs of
the test split over three real test questions (plus copies where a test needs many), graded by
eval.run.grade and judged by hand."""
import json

import pytest

from app.tools import dispute_timeline
from eval import golden as G
from eval import judge as J
from eval import repro as Q
from eval import run as R

IX = R.index_view()
BY_ID = {i.id: i for i in G.load()}
ANS, TOOL, DEC = BY_ID["q015"], BY_ID["q012"], BY_ID["q025"]       # Z fees, Z billing error (tool), decline
A, B = "r20991201T000000Z", "r20991202T000000Z"
SAME = lambda a, b: False                                           # noqa: E731 - app/ unchanged


def resp(retrieved=("1026.52(a)", "1026.4(a)"), cites=("1026.52(a)(1)",), tokens=1000, **over):
    r = {"answer": "An answer.", "citations": list(cites), "tool_calls": [], "abstained": False,
         "abstain_reason": None, "retrieved_ids": list(retrieved), "regulations_as_of": "2026-10-23",
         "disclaimer": "x", "raw_answer": "An answer.", "raw_citations": list(cites), "invalid_citations": [],
         "flags": [], "prompt_version": "p2", "config_hash": "b2bef7bd2dd0", "model_id": "m",
         "usage": {"inputTokens": tokens, "outputTokens": 100}, "model_calls": 1, "latency_ms": 4000}
    r.update(over)
    return r


def tool_resp(tokens=1000):
    args = dict(TOOL.expected_tool_call.args)
    dates = " and ".join(d.date.strftime("%B %d, %Y") for d in TOOL.expected_result.deadlines)
    return resp(retrieved=("1026.13(c)", "1026.13(b)"), cites=("1026.13(c)(1)",), tokens=tokens,
                tool_calls=[{"name": "dispute_timeline", "args": args, "result": dispute_timeline(**args)}],
                answer=f"Acknowledge and resolve by {dates}.")


def decline(tokens=1000):
    return resp(retrieved=("1026.4(a)",), cites=(), abstained=True, tokens=tokens, answer="Outside Regulations E and Z.")


def add_run(db, run_id, split="test", config_hash="b2bef7bd2dd0", commit="7eb3622abc", n=3):
    db.execute("INSERT INTO runs (run_id, split, git_commit, git_dirty, config_hash, config_json, prompt_version, "
               "grader_version, n_items, today_default) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (run_id, split, commit, 0, config_hash, json.dumps({"search_mode": "hybrid", "k": 5, "tool_on": True}),
                "p2", R.GRADER_VERSION, n, "2026-11-03"))


def log(db, run_id, item, r):
    R.log_result(db, run_id, item, r, None, R.grade(item, r, IX))


def verdict(db, run_id, item, *, label="faithful", covered=None, status="judged"):
    n = len(item.key_facts or [])
    covered = n if covered is None else covered
    vj = json.dumps({"claims": [{"claim": "ok", "supported": True, "source": "passage 1"}], "faithfulness": label,
                     "rationale": "why", "key_facts": [{"n": k, "covered": k <= covered} for k in range(1, n + 1)]})
    db.execute("INSERT INTO judgements (run_id, qid, judge_version, status, faithfulness, kf_covered, kf_total, "
               "verdict_json) VALUES (?,?,?,?,?,?,?,?)",
               (run_id, item.id, "j2", status, label if status == "judged" else None,
                covered if status == "judged" else None, n if status == "judged" else None,
                vj if status == "judged" else None))
    db.commit()


def fill(db, run_id, answers):
    """answers: [(item, response, verdict kwargs)]"""
    for item, r, v in answers:
        log(db, run_id, item, r)
        verdict(db, run_id, item, **v)


def base(tokens=1000):
    return [(ANS, resp(tokens=tokens), {}), (TOOL, tool_resp(tokens), {}), (DEC, decline(tokens), {})]


@pytest.fixture
def db(tmp_path):
    d = J.connect(tmp_path / "runs.db")
    add_run(d, A)
    fill(d, A, base())
    return d


def second(db, answers, **run):
    add_run(db, B, **run)
    fill(db, B, answers)


# ------------------------------------------------------------------ the rule's numbers

def test_limits_are_the_chosen_rule():
    assert (Q.LIMIT_SHARE, Q.LIMIT_MIN, Q.TOKEN_SHARE, Q.ALPHA) == (0.125, 3, 0.05, 0.05)
    assert [Q.limit(n) for n in (8, 12, 24, 36, 70)] == [3, 3, 3, 5, 9]
    assert set(Q.COUNTED) <= {k for k, *_ in __import__("eval.report", fromlist=["PAIRED"]).PAIRED}


# ------------------------------------------------------------------ passing

def test_identical_runs_pass_both_checks(db):
    second(db, base())
    res = Q.check(db, A, B, ix=IX, app_changed=SAME)
    assert res["pass"] and all(x["pass"] for x in res["check1"])
    c2 = res["check2"]
    assert all(c["move"] == 0 for c in c2["counts"]) and c2["questions_changed"] == 0
    assert {m["p"] for m in c2["mcnemar"]} == {1.0} and c2["tokens"]["share"] == 0


def test_ordinary_noise_passes(db):
    """One answer worse (judge partially faithful, a key fact missed) and 3% more tokens: within the rule."""
    second(db, [(ANS, resp(tokens=1030), {"label": "partially_faithful", "covered": 2}),
                (TOOL, tool_resp(1030), {}), (DEC, decline(1030), {})])
    res = Q.check(db, A, B, ix=IX, app_changed=SAME)
    c2 = res["check2"]
    assert res["pass"], Q.text(res)
    assert next(c for c in c2["counts"] if c["measure"] == "key facts covered")["move"] == 1
    assert c2["questions_changed"] == 1


# ------------------------------------------------------------------ each rule failing on its own

def test_search_must_be_identical(db):
    second(db, [(ANS, resp(retrieved=("1026.4(a)", "1026.52(a)")), {}), (TOOL, tool_resp(), {}), (DEC, decline(), {})])
    c2 = Q.check(db, A, B, ix=IX, app_changed=SAME)["check2"]
    assert not c2["parts"]["search"] and not c2["pass"] and c2["rank_moves"] == ["q015"]
    assert c2["parts"]["counts"] and c2["parts"]["mcnemar"] and c2["parts"]["tokens"]


def test_tokens_must_stay_within_5_percent(db):
    second(db, base(tokens=1060))
    c2 = Q.check(db, A, B, ix=IX, app_changed=SAME)["check2"]
    assert not c2["tokens"]["pass"] and round(c2["tokens"]["share"], 3) == 0.06 and not c2["pass"]
    assert c2["parts"]["search"] and c2["parts"]["counts"] and c2["parts"]["mcnemar"]


def test_a_large_move_fails_counts_and_mcnemar(tmp_path):
    """Eight copies of q015, all passing in the first run and all declined in the second (search the same)."""
    copies = [ANS.model_copy(update={"id": f"q9{n:02d}"}) for n in range(1, 9)]
    d = J.connect(tmp_path / "runs.db")
    add_run(d, A, n=8)
    fill(d, A, [(c, resp(), {}) for c in copies])
    add_run(d, B, n=8)
    fill(d, B, [(c, resp(abstained=True), {"label": "faithful", "covered": 0}) for c in copies])
    c2 = Q.check(d, A, B, ix=IX, app_changed=SAME)["check2"]
    answered = next(c for c in c2["counts"] if c["measure"] == "answerable answered")
    assert (answered["first"], answered["second"], answered["limit"], answered["pass"]) == (8, 0, 3, False)
    mc = next(m for m in c2["mcnemar"] if m["measure"] == "answerable answered")
    assert mc["first_only"] == 8 and mc["p"] < 0.05 and not mc["pass"]
    assert c2["parts"]["search"] and not c2["parts"]["counts"] and not c2["parts"]["mcnemar"] and not c2["pass"]


def test_check1_catches_a_logged_metric_that_regrading_doesnt_reproduce(db):
    second(db, base())
    m = json.loads(db.execute("SELECT metrics_json FROM results WHERE run_id = ? AND qid = 'q015'", (A,)).fetchone()[0])
    m["hit_at_5"] = not m["hit_at_5"]
    db.execute("UPDATE results SET metrics_json = ? WHERE run_id = ? AND qid = 'q015'", (json.dumps(m), A))
    res = Q.check(db, A, B, ix=IX, app_changed=SAME)
    one = res["check1"][0]
    assert not one["regrade_same"] and one["changed"] == ["q015"] and not res["pass"] and res["check2"]["pass"]


# ------------------------------------------------------------------ not comparable

@pytest.mark.parametrize("change, expect", [
    ({"run": {"config_hash": "edca2fd6f78d"}}, "config hashes differ"),
    ({"run": {"split": "dev"}}, "isn't a run of the whole test split"),
    ({"app": True}, "app/ changed"),
    ({"drop_verdict": True}, "no j2 verdict in the second run for q025"),
    ({"relabel": True}, "labels changed between the runs: q015"),
    ({"missing": True}, "only in the first q025"),
])
def test_runs_that_arent_comparable_are_not_checked(db, change, expect):
    answers = base()
    if change.get("relabel"):
        answers[0] = (ANS.model_copy(update={"key_facts": [*ANS.key_facts[:-1], "changed"]}), resp(), {})
    if change.get("missing"):
        answers = answers[:2]
    add_run(db, B, **change.get("run", {}))
    for item, r, v in answers:
        log(db, B, item, r)
        if not (change.get("drop_verdict") and item.id == "q025"):
            verdict(db, B, item, **v)
    res = Q.check(db, A, B, ix=IX, app_changed=(lambda a, b: True) if change.get("app") else SAME)
    assert "problems" in res and any(expect in p for p in res["problems"]), res


def test_a_failed_question_must_be_rerun_first(db):
    add_run(db, B)
    fill(db, B, base()[:2])
    R.log_result(db, B, DEC, None, "ThrottlingException", None)
    res = Q.check(db, A, B, ix=IX, app_changed=SAME)
    assert any("failed to run in the second run: q025" in p for p in res["problems"])


# ------------------------------------------------------------------ the command

def test_command(db, tmp_path, monkeypatch, capsys):
    second(db, base(tokens=1060))
    db.close()
    dbfile = tmp_path / "runs.db"
    monkeypatch.setattr(Q, "REPRO_FILE", tmp_path / "final" / "reproducibility.json")
    assert Q.main([A, B, "--db", str(dbfile), "--save"]) == 1           # tokens fail; same commit: app/ unchanged
    out = capsys.readouterr().out
    assert "Check 1, exact: PASS" in out and "Check 2, the re-run: FAIL" in out and "No third run" in out
    saved = json.loads((tmp_path / "final" / "reproducibility.json").read_text(encoding="utf-8"))
    assert saved["first"] == A and saved["pass"] is False and saved["check2"]["parts"]["tokens"] is False
    assert Q.main([A, A, "--db", str(dbfile)]) == 2
    assert Q.main([A, "r20991231T000000Z", "--db", str(dbfile)]) == 2 and "no run" in capsys.readouterr().out
    assert Q.main([A, B, "--db", str(tmp_path / "none.db")]) == 2
