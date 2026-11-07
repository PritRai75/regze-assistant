"""The eval harness (eval/run.py), on made-up responses. No AWS call: grading works on response dicts,
and the end-to-end test uses a fake Bedrock client and a ready-made question vector.
IDs and dates here are unrelated to the golden set's questions, so the tests don't hint at its labels."""
import copy
import json
import tomllib
from datetime import date
from pathlib import Path

import pytest

from app import retrieve as rt
from app.config import DEFAULT
from app.tools import dispute_timeline
from eval import golden as G
from eval import run as R

ROOT = Path(__file__).resolve().parents[1]
IX = R.index_view()
STATUSES = G.provision_statuses()
OWNER = {p: c for c, ps in IX.chunk_provisions.items() for p in ps}
GOLD_CHUNK = OWNER["1005.17(b)(1)"]              # "1005.17(b)"
OTHER_CHUNK = OWNER["1005.11(c)(1)"]             # unrelated to the gold family


def items_from(toml_text):
    items, problems = G.parse(tomllib.loads(toml_text)["question"])
    problems += G.check(items, STATUSES)
    assert problems == [], problems
    return items


ANSWERABLE = '''
[[question]]
id = "q901"
stage = "labelled"
persona = "compliance"
type = "reg_e_account"
question = """Under Reg E, what must the overdraft opt-in notice say?"""
regulation = "E"
gold_ids = ["1005.17(b)(1)"]
reference_answer = """Two to four sentences."""
key_facts = ["one fact"]
should_abstain = false
'''

DECLINE = '''
[[question]]
id = "q902"
stage = "labelled"
persona = "frontline"
type = "decline"
question = """Does the card network's own promise cover this?"""
gold_ids = []
reference_answer = """Not in the corpus."""
key_facts = []
should_abstain = true
abstain_category = "network_rules"
'''

ARGS = {"regulation": "E", "event": "error", "notice_date": "2026-12-01", "transfer_kind": "pos_debit"}
TOOL_RESULT = dispute_timeline(**ARGS)


def with_tool(base=ANSWERABLE.replace('type = "reg_e_account"', 'type = "deadline"')):
    rows = [f'  {{ what = "x", date = {d["date"]}, rule = "{d["rule"].split(",")[0]}" }},' for d in TOOL_RESULT["deadlines"]]
    return (base.replace('gold_ids = ["1005.17(b)(1)"]', 'gold_ids = ["1005.11(c)"]')
            + '\n[question.expected_tool_call]\nname = "dispute_timeline"\n'
            + 'args = { regulation = "E", event = "error", notice_date = 2026-12-01, transfer_kind = "pos_debit" }\n'
            + "\n[question.expected_result]\ndeadlines = [\n" + "\n".join(rows) + "\n]\n")


def resp(**over):
    """A made-up app.schema.Response as a dict."""
    base = {"answer": "An answer [§ 1005.17(b)(1)].", "citations": ["1005.17(b)(1)"], "tool_calls": [],
            "abstained": False, "abstain_reason": None, "retrieved_ids": [GOLD_CHUNK, OTHER_CHUNK],
            "regulations_as_of": "2026-10-23", "disclaimer": "x", "raw_answer": "An answer",
            "raw_citations": ["1005.17(b)(1)"], "invalid_citations": [], "flags": [], "prompt_version": "p2",
            "config_hash": "abc", "model_id": "m", "usage": {"inputTokens": 1000, "outputTokens": 100},
            "model_calls": 1, "latency_ms": 5000}
    base.update(over)
    return base


# ---------------------------------------------------------------- reading dates and amounts

def test_dates_in_reads_the_common_forms():
    text = ("Investigate by Tuesday, December 15, 2026, or 15 December; acknowledge by 2026-12-31 "
            "and 1/5/2027. The 3rd of March 2027 and Nov. 20 too.")
    assert R.dates_in(text) == {(2026, 12, 15), (None, 12, 15), (2026, 12, 31), (2027, 1, 5),
                                (2027, 3, 3), (None, 11, 20)}


def test_dates_in_ignores_numbers_that_are_not_dates():
    assert R.dates_in("You may 15 days later, within 60 days, 10 business days, in May 2026, $500.") == set()
    assert R.date_written(date(2026, 12, 15), R.dates_in("by December 15"))          # no year written
    assert not R.date_written(date(2026, 12, 15), R.dates_in("by December 15, 2027"))


def test_amounts_tell_50_from_500():
    hit = {t: [bool(R.AMOUNTS[t].search(s)) for s in ("$50.", "$ 50.00", "up to $50, and", "$500", "$50,000")]
           for t in R.AMOUNTS}
    assert hit["max_50"] == [True, True, True, False, False]
    assert hit["max_500"] == [False, False, False, True, False]


# ---------------------------------------------------------------- retrieval and citations

def test_retrieval_rank_hits_and_gold_recall():
    item = items_from(ANSWERABLE)[0]
    m = R.grade(item, resp(retrieved_ids=[OTHER_CHUNK, GOLD_CHUNK]), IX)
    assert (m["first_gold_rank"], m["hit_at_1"], m["hit_at_3"], m["hit_at_5"], m["rr"]) == (2, False, True, True, 0.5)
    assert (m["gold_found"], m["gold_total"]) == (1, 1)
    miss = R.grade(item, resp(retrieved_ids=[OTHER_CHUNK]), IX)
    assert (miss["first_gold_rank"], miss["hit_at_5"], miss["rr"], miss["gold_found"]) == (None, False, 0.0, 0)


def test_citations_count_the_gold_family_from_the_raw_list():
    item = items_from(ANSWERABLE)[0]
    raw = ["§ 1005.17(b)(1)(i)",          # under the gold ID: in the family, in a retrieved chunk
           "comment 1005.17(b)-1",       # a comment on the parent: not in the family, not retrieved
           "1026.52(b)",                 # the other regulation, not retrieved
           "1005.99(z)"]                 # doesn't exist
    m = R.grade(item, resp(raw_citations=raw, invalid_citations=["1026.52(b)", "1005.99(z)"]), IX)
    assert (m["cited"], m["cited_gold"], m["cited_not_in_index"], m["invented"]) == (4, 1, 1, 2)
    assert m["invented_ids"] == ["1005.17(b)-1", "1026.52(b)"]          # real IDs only; 1005.99(z) isn't one
    assert m["unread_citations"] == ["1005.99(z)"] and m["guard_rejected"] == 2
    assert m["wrong_reg"] and m["wrong_reg_ids"] == ["1026.52(b)"]


def test_g3_reads_packed_ranges_subpoints_and_sorts_the_rest():
    item = items_from(ANSWERABLE)[0]
    raw = ["1005.17(b)(1), § 1005.17(b)(1)(i)",   # packed: two IDs, both in the gold family
           "1005.11(c)(2)(iii)-(iv)",             # range: two IDs, retrieved (OTHER_CHUNK), not gold
           "1005.17(b)-1.ii",                     # a comment's sub-point: the comment, not retrieved
           "1005.11",                             # a whole section: too broad
           "comment 12(b)-1",                     # no part, exists in both
           "see the network rules"]               # no reference at all
    m = R.grade(item, resp(raw_citations=raw, invalid_citations=raw), IX)
    assert m["cited"] == 5 + 3 and m["cited_gold"] == 2
    assert (m["cited_too_broad"], m["cited_no_part"], m["cited_unreadable"], m["cited_not_in_index"]) == (1, 1, 1, 0)
    assert m["invented_ids"] == ["1005.17(b)-1"] and m["guard_rejected"] == 6
    assert m["unread_citations"] == ["1005.11", "comment 12(b)-1", "see the network rules"]


def test_an_id_from_a_tool_result_is_not_invented():
    item = items_from(ANSWERABLE)[0]
    call = {"name": "dispute_timeline", "args": ARGS, "result": TOOL_RESULT}
    cited = next(p for p in R.PROVISION.findall(json.dumps(TOOL_RESULT))
                 if not any(p.lower() == q.lower() for c in (GOLD_CHUNK, OTHER_CHUNK) for q in IX.chunk_provisions[c]))
    m = R.grade(item, resp(raw_citations=[cited], tool_calls=[call]), IX)
    assert m["invented"] == 0
    m = R.grade(item, resp(raw_citations=[cited]), IX)
    assert m["invented"] == 1


def test_status_error_when_an_answer_cites_a_provision_not_in_force():
    item = items_from(ANSWERABLE)[0]
    flagged = next(p for p, s in IX.status.items() if s != "in_force" and p.startswith("1005."))
    m = R.grade(item, resp(raw_citations=[flagged]), IX)
    assert m["status_error"] and m["not_in_force_ids"] == [flagged]
    declined = R.grade(item, resp(raw_citations=[flagged], abstained=True), IX)
    assert not declined["status_error"] and not declined["wrong_reg"]


# ---------------------------------------------------------------- abstention

def test_abstention_outcomes_and_causes():
    ans, dec = items_from(ANSWERABLE)[0], items_from(DECLINE)[0]
    assert R.grade(dec, resp(abstained=True), IX)["abstain_outcome"] == "correct_decline"
    assert R.grade(dec, resp(), IX)["abstain_outcome"] == "missed_abstention"
    assert R.grade(ans, resp(), IX)["abstain_outcome"] == "correct_answer"
    over = R.grade(ans, resp(abstained=True, flags=["invalid_citations", "no_valid_citation"]), IX)
    assert (over["abstain_outcome"], over["abstain_cause"]) == ("over_abstention", "no_valid_citation")
    assert R.grade(dec, resp(abstained=True), IX)["abstain_cause"] == "model"


# ---------------------------------------------------------------- tool calls

def call(args, result=None, name="dispute_timeline"):
    return {"name": name, "args": args, "result": result if result is not None else dispute_timeline(**args)}


def test_tool_outcomes():
    plain, tooled = items_from(ANSWERABLE)[0], items_from(with_tool())[0]
    assert R.grade(plain, resp(), IX)["tool_outcome"] == "correct_no_call"
    assert R.grade(plain, resp(tool_calls=[call(ARGS)]), IX)["tool_outcome"] == "unneeded_call"
    assert R.grade(tooled, resp(), IX)["tool_outcome"] == "missed_call"
    same = {**ARGS, "learned_of_loss": None, "cycle_closing_dates": []}          # empty extras don't count
    assert R.grade(tooled, resp(tool_calls=[call(same)]), IX)["tool_outcome"] == "correct_call"
    wrong = R.grade(tooled, resp(tool_calls=[call({**ARGS, "transfer_kind": "standard"})]), IX)
    assert (wrong["tool_outcome"], wrong["arg_diffs"], wrong["tool_correct"]) == ("wrong_args", ["transfer_kind"], False)
    retried = R.grade(tooled, resp(tool_calls=[call({**ARGS, "transfer_kind": "standard"}), call(ARGS)]), IX)
    assert retried["tool_outcome"] == "correct_call" and retried["tool_calls"] == 2   # any matching call counts


def test_g6_another_transfer_kind_with_the_same_timeline_matches():
    e = {"regulation": "E", "event": "error", "notice_date": "2026-10-28", "transfer_kind": "foreign"}
    new_acct = {**e, "first_deposit": "2026-10-01", "transfer_date": "2026-10-20"}
    def outcome(label, got):
        item = items_from(with_tool())[0].model_copy(
            update={"expected_tool_call": G.ToolCall(name="dispute_timeline", args=label)})
        m = R.grade_tool(item, [{"name": "dispute_timeline", "args": got}])
        return m["tool_outcome"], m["arg_diffs"]
    assert outcome(e, {**e, "transfer_kind": "pos_debit"}) == ("correct_call", [])        # both 90 days
    assert outcome(e, {**e, "transfer_kind": "standard"}) == ("wrong_args", ["transfer_kind"])   # 45 days instead
    assert outcome(new_acct, {**new_acct, "transfer_kind": "standard"}) == ("correct_call", [])  # new account: same
    assert outcome(new_acct, {**new_acct, "transfer_kind": "pos_debit"}) == ("correct_call", [])
    # another argument differing too: no excuse, both named
    assert outcome(e, {**e, "transfer_kind": "pos_debit", "notice_date": "2026-10-29"}) == \
        ("wrong_args", ["notice_date", "transfer_kind"])
    assert outcome(e, {**e, "transfer_kind": None}) == ("wrong_args", ["transfer_kind"])   # missing: the tool refuses
    assert outcome(e, {**e, "transfer_kind": "atm"}) == ("wrong_args", ["transfer_kind"])  # not a kind
    assert R.same_timeline(new_acct, {**new_acct, "transfer_kind": "standard"})
    assert not R.same_timeline(e, {**e, "transfer_kind": "standard"})


def test_g5_closing_dates_match_when_the_call_starts_with_the_label():
    z = {"regulation": "Z", "event": "error", "notice_date": "2027-03-02",
         "cycle_closing_dates": ["2027-03-10", "2027-04-10", "2027-05-10"]}
    item = items_from(with_tool())[0].model_copy(update={"expected_tool_call": G.ToolCall(name="dispute_timeline", args=z)})
    outcome = lambda dates: R.grade_tool(item, [{"name": "dispute_timeline", "args": {**z, "cycle_closing_dates": dates}}])
    assert outcome(z["cycle_closing_dates"])["tool_outcome"] == "correct_call"
    assert outcome(["2027-05-10", "2027-04-10", "2027-03-10"])["tool_outcome"] == "correct_call"   # order doesn't matter
    assert outcome(z["cycle_closing_dates"] + ["2027-06-10"])["tool_outcome"] == "correct_call"     # a later extra: same deadline
    for wrong in (["2027-03-10", "2027-04-10"],                         # too few: the tool falls back to 90 days
                  ["2027-02-10", "2027-03-10", "2027-04-10", "2027-05-10"],   # an earlier extra shifts the cycles
                  []):                                                  # none at all
        m = outcome(wrong)
        assert (m["tool_outcome"], m["arg_diffs"]) == ("wrong_args", ["cycle_closing_dates"]), wrong
    no_list = item.model_copy(update={"expected_tool_call": G.ToolCall(
        name="dispute_timeline", args={k: v for k, v in z.items() if k != "cycle_closing_dates"})})
    assert R.grade_tool(no_list, [{"name": "dispute_timeline", "args": z}])["tool_outcome"] == "wrong_args"
    assert R.arg_matches("notice_date", "2027-03-02", "2027-03-02") and not R.arg_matches("notice_date", "2027-03-02", None)


def test_norm_args_writes_dates_and_lists_one_way():
    assert R.norm_args({"notice_date": date(2026, 12, 1), "cycle_closing_dates": ["2027-01-05", "2026-12-05"],
                        "statement_sent": None}) == {"notice_date": "2026-12-01",
                                                     "cycle_closing_dates": ["2026-12-05", "2027-01-05"]}


# ---------------------------------------------------------------- calculation exactness

def answer_with(dates):
    return "The institution must act by " + " and ".join(d.strftime("%B %d, %Y").replace(" 0", " ") for d in dates) + "."


def test_calc_exact_needs_every_deadline_in_the_answer_text():
    item = items_from(with_tool())[0]
    dates = [d.date for d in item.expected_result.deadlines]
    m = R.grade(item, resp(answer=answer_with(dates), tool_calls=[call(ARGS)]), IX)
    assert (m["calc_checks"], m["calc_missing"], m["calc_exact"], m["tool_result_right"]) == (len(dates), [], True, True)
    last = item.expected_result.deadlines[-1]                 # the only deadline on its date
    assert dates.count(last.date) == 1
    short = R.grade(item, resp(answer=answer_with(dates[:-1]), tool_calls=[call(ARGS)]), IX)
    assert short["calc_exact"] is False and short["calc_missing"] == [f"{last.rule} {last.date}"]
    assert short["tool_result_right"]                       # the tool was right; the answer misreported it


def test_calc_wrong_tool_result_and_abstention():
    item = items_from(with_tool())[0]
    dates = [d.date for d in item.expected_result.deadlines]
    wrong = call({**ARGS, "transfer_kind": "standard"})
    assert R.grade(item, resp(answer=answer_with(dates), tool_calls=[wrong]), IX)["tool_result_right"] is False
    assert R.grade(item, resp(answer=answer_with(dates), abstained=True), IX)["calc_exact"] is False
    plain = items_from(ANSWERABLE)[0]
    assert R.grade(plain, resp(), IX)["calc_exact"] is None


def test_tiers_without_an_amount_are_left_to_the_judge():
    text = ANSWERABLE.replace('type = "reg_e_account"', 'type = "unauthorized"')
    text += ('\n[question.expected_tool_call]\nname = "dispute_timeline"\n'
             'args = { regulation = "E", event = "unauthorized", notice_date = 2026-12-01 }\n'
             '\n[question.expected_result]\nliability_tiers = ["none"]\n')
    raw = tomllib.loads(text)["question"]
    item = G.parse(raw)[0][0]
    m = R.grade(item, resp(), IX)
    assert m["calc_checks"] == 0 and m["calc_exact"] is None


# ---------------------------------------------------------------- summary and report

def test_summary_counts_by_slice():
    ans, dec = items_from(ANSWERABLE)[0], items_from(DECLINE)[0]
    rows = [(ans, R.grade(ans, resp(), IX), None),
            (dec, R.grade(dec, resp(abstained=True, latency_ms=9000), IX), None),
            (ans.model_copy(update={"id": "q903"}), None, "ClientError: throttled")]
    s = R.summarise(rows)
    o = s["overall"]
    assert (o["n"], o["answerable"], o["declines"]) == (2, 1, 1)
    assert o["hit_at_5"] == (1, 1) and o["abstain_recall"] == (1, 1) and o["abstain_precision"] == (1, 1)
    assert o["citation_precision"] == (1, 1) and o["tool_correct"] == (2, 2) and o["calc_exact"] == (0, 0)
    assert s["declines_by_category"] == {"network_rules": (1, 1)} and s["errors"] == [("q903", "ClientError: throttled")]
    assert set(s["by_persona"]) == {"compliance", "frontline"}
    text = R.report(s)
    assert "hit@5 1 of 1 (100%)" in text and "network_rules 1/1" in text and "q903: ClientError" in text


def test_select_skips_drafts_and_uncertain():
    items = items_from(ANSWERABLE + DECLINE)
    draft = G.parse([{"id": "q904", "stage": "draft", "persona": "frontline", "type": "dispute", "question": "Q?"}])[0]
    unsure = [items[0].model_copy(update={"id": "q905", "uncertain": True, "question": "Other?"})]
    chosen, notes = R.select(items + draft + unsure, "unsplit", None)
    assert [i.id for i in chosen] == ["q901", "q902"]
    assert notes == ["skipped 1 draft(s): q904", "skipped 1 uncertain: q905"]
    assert R.select(items, "dev", None)[0] == []                          # nothing is split yet
    assert [i.id for i in R.select(items, None, ["q902"])[0]] == ["q902"]
    with pytest.raises(SystemExit):
        R.select(items, None, ["q999"])


# ---------------------------------------------------------------- end to end, fake Bedrock

needs_index = pytest.mark.skipif(not (ROOT / "index" / "embeddings.npy").exists(), reason="index not built yet")


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def converse(self, **request):
        self.requests.append(request)
        return copy.deepcopy(self.replies.pop(0))


def text_reply(body):
    return {"output": {"message": {"role": "assistant", "content": [{"text": json.dumps(body)}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 900, "outputTokens": 60}}


def tool_reply(args):
    return {"output": {"message": {"role": "assistant", "content": [
                {"toolUse": {"toolUseId": "t1", "name": "dispute_timeline", "input": args}}]}},
            "stopReason": "tool_use", "usage": {"inputTokens": 1000, "outputTokens": 40}}


def fake_embed(question):
    chunks, vecs, _, _ = rt.load()
    return vecs[[c["id"] for c in chunks].index(GOLD_CHUNK)]


@needs_index
def test_run_logs_every_answer_and_regrades_the_same(tmp_path):
    items = items_from(ANSWERABLE + DECLINE)
    client = FakeClient(
        text_reply({"answer": "It must describe the service [§ 1005.17(b)(1)].", "citations": ["1005.17(b)(1)"],
                    "abstained": False, "abstain_reason": None}),
        text_reply({"answer": "The passages don't cover network rules.", "citations": [], "abstained": True,
                    "abstain_reason": "network rules are not in the corpus"}))
    db = R.connect(tmp_path / "runs.db")
    lines = []
    run_id, summary = R.run(items, DEFAULT, db=db, split="unsplit", client=client, embed=fake_embed,
                            progress=lines.append, note="test")
    o = summary["overall"]
    assert (o["n"], o["hit_at_5"], o["abstain_recall"], o["tool_correct"]) == (2, (1, 1), (1, 1), (2, 2))
    assert len(lines) == 2 and lines[0].startswith("  [1/2] q901: correct_answer")

    head = db.execute("SELECT config_hash, prompt_version, as_of, grader_version, n_items, finished_at, note FROM runs").fetchone()
    assert head[1:5] == ("p2", "2026-10-23", R.GRADER_VERSION, 2) and head[5] and head[6] == "test"
    rows = db.execute("SELECT qid, abstained, hit_at_5, tool_outcome, response_json FROM results ORDER BY qid").fetchall()
    assert [r[:4] for r in rows] == [("q901", 0, 1, "correct_no_call"), ("q902", 1, 0, "correct_no_call")]
    assert json.loads(rows[0][4])["config_hash"] == head[0]

    again, changed, version = R.regrade(db, run_id)
    assert changed == [] and version == R.GRADER_VERSION and json.loads(json.dumps(again)) == json.loads(json.dumps(summary))
    assert run_id in R.list_runs(db)

    # a run logged by an older grader: the re-grade says what changed, question by question
    old = json.loads(db.execute("SELECT metrics_json FROM results WHERE qid = 'q901'").fetchone()[0])
    old["invented"], old["gone_field"] = 3, 1
    del old["cited_too_broad"]
    db.execute("UPDATE results SET metrics_json = ? WHERE qid = 'q901'", (json.dumps(old),))
    db.execute("UPDATE runs SET grader_version = 'g0'")
    _, changed, version = R.regrade(db, run_id)
    assert version == "g0" and changed == [("q901", ["invented: 3 -> 0", "new fields: cited_too_broad",
                                                      "dropped fields: gone_field"])]


@needs_index
def test_run_with_the_tool_scores_the_calculation(tmp_path):
    item = items_from(with_tool())[0]
    dates = [d.date for d in item.expected_result.deadlines]
    client = FakeClient(tool_reply(ARGS), text_reply({
        "answer": answer_with(dates) + " [§ 1005.11(c)(1)]", "citations": ["1005.11(c)(1)"],
        "abstained": False, "abstain_reason": None}))
    _, summary = R.run([item], DEFAULT, db=R.connect(tmp_path / "runs.db"), client=client, embed=fake_embed,
                       progress=lambda _: None)
    o = summary["overall"]
    assert o["tool_correct"] == (1, 1) and o["calc_exact"] == (1, 1) and summary["tool_outcomes"] == {"correct_call": 1}


@needs_index
def test_a_failure_after_the_first_question_is_recorded(tmp_path):
    items = items_from(ANSWERABLE + DECLINE)

    class Failing(FakeClient):
        def converse(self, **request):
            if not self.replies:
                raise RuntimeError("throttled")
            return super().converse(**request)

    client = Failing(text_reply({"answer": "x [§ 1005.17(b)(1)]", "citations": ["1005.17(b)(1)"],
                                 "abstained": False, "abstain_reason": None}))
    db = R.connect(tmp_path / "runs.db")
    _, summary = R.run(items, DEFAULT, db=db, client=client, embed=fake_embed, progress=lambda _: None)
    assert summary["errors"] == [("q902", "RuntimeError: throttled")] and summary["overall"]["n"] == 1
    assert db.execute("SELECT error FROM results WHERE qid = 'q902'").fetchone()[0] == "RuntimeError: throttled"
    assert R.regrade(db, R.list_runs(db).split()[0])[1] == []


@needs_index
def test_a_failure_on_the_first_question_stops_the_run(tmp_path):
    class Broken:
        def converse(self, **request):
            raise RuntimeError("no credentials")
    with pytest.raises(RuntimeError):
        R.run(items_from(ANSWERABLE), DEFAULT, db=R.connect(tmp_path / "runs.db"), client=Broken(),
              embed=fake_embed, progress=lambda _: None)


# ---------------------------------------------------------------- command line (no model calls)

def test_test_split_needs_final(tmp_path, capsys):
    assert R.main(["--split", "test", "--db", str(tmp_path / "runs.db")]) == 2
    assert "--final" in capsys.readouterr().out


def test_runs_and_regrade_on_an_empty_database(tmp_path, capsys):
    db = str(tmp_path / "runs.db")
    assert R.main(["--runs", "--db", db]) == 0 and "No runs logged yet." in capsys.readouterr().out
    with pytest.raises(SystemExit):
        R.main(["--regrade", "r20990101T000000Z", "--db", db])


def reg_tier_item(reg, tier):
    text = ANSWERABLE.replace('type = "reg_e_account"', 'type = "unauthorized"')
    text += ('\n[question.expected_tool_call]\nname = "dispute_timeline"\n'
             f'args = {{ regulation = "{reg}", event = "unauthorized", notice_date = 2026-12-01 }}\n'
             f'\n[question.expected_result]\nliability_tiers = ["{tier}"]\n')
    return G.parse(tomllib.loads(text)["question"])[0][0]


def test_reg_e_tier_amount_must_appear_in_the_answer():
    item = reg_tier_item("E", "max_50")
    assert R.grade(item, resp(answer="At most $50 [§ 1005.17(b)(1)]."), IX)["calc_exact"] is True
    m = R.grade(item, resp(answer="Nothing [§ 1005.17(b)(1)]."), IX)
    assert m["calc_exact"] is False and m["calc_missing"] == ["tier max_50"]


def test_reg_z_tier_is_a_cap_so_its_amount_is_not_required():
    item = reg_tier_item("Z", "max_50")
    m = R.grade(item, resp(answer="He owes $0 [§ 1005.17(b)(1)]."), IX)
    assert (m["calc_checks"], m["calc_missing"], m["calc_exact"]) == (0, [], None)


@needs_index
def test_a_p3_run_logs_its_prompt_and_sends_it(tmp_path):
    from app.prompts import SYSTEM_PROMPT_P3
    client = FakeClient(text_reply({"answer": "It must describe the service [§ 1005.17(b)(1)].",
                                    "citations": ["1005.17(b)(1)"], "abstained": False, "abstain_reason": None}))
    db = R.connect(tmp_path / "runs.db")
    config = DEFAULT.with_(prompt="p3")
    run_id, _ = R.run(items_from(ANSWERABLE), config, db=db, split="unsplit", client=client, embed=fake_embed,
                      progress=lambda _: None)
    head = db.execute("SELECT prompt_version, config_json, config_hash FROM runs").fetchone()
    assert head[0] == "p3" and json.loads(head[1])["prompt"] == "p3"
    assert client.requests[0]["system"] == [{"text": SYSTEM_PROMPT_P3}]
    resp = json.loads(db.execute("SELECT response_json FROM results").fetchone()[0])
    assert resp["prompt_version"] == "p3" and resp["config_hash"] == head[2]
