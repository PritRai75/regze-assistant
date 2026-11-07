"""Tests for eval/report.py: the error tags and their order, the outside-law keywords, the slices and
the author's review. No AWS call: answers are made up, graded by eval.run.grade and judged by hand."""
import json
import tomllib

import pytest

from app.tools import dispute_timeline
from eval import golden as G
from eval import judge as J
from eval import report as P
from eval import run as R

IX = R.index_view()
BY_ID = {i.id: i for i in G.load()}
PLAIN = BY_ID["q006"]          # Z, frontline, one gold ID (1026.10(b)(2)(ii)), two key facts, no tool
TWO_GOLD = BY_ID["q005"]       # Z, compliance, gold 1026.9(c)(2)(i) and 1026.55(b)(3)
TOOL = BY_ID["q004"]           # E, a dispute_timeline call with two deadlines
NETWORK = BY_ID["q008"]        # decline, network_rules
STATE = BY_ID["q024"]          # decline, state_law
OTHER_CHUNK = "1026.4(a)"      # holds no gold ID of these items
NOT_IN_FORCE = "1026.2(a)(15)(i)"


def resp(retrieved=("1026.10(b)", OTHER_CHUNK), cites=("1026.10(b)(2)(ii)",), **over):
    """A made-up app.schema.Response as a dict."""
    r = {"answer": "An answer.", "citations": list(cites), "tool_calls": [], "abstained": False,
         "abstain_reason": None, "retrieved_ids": list(retrieved), "regulations_as_of": "2026-10-23",
         "disclaimer": "x", "raw_answer": "An answer.", "raw_citations": list(cites), "invalid_citations": [],
         "flags": [], "prompt_version": "p2", "config_hash": "abc", "model_id": "m",
         "usage": {"inputTokens": 1000, "outputTokens": 100}, "model_calls": 1, "latency_ms": 4000}
    r.update(over)
    return r


def tool_resp(args=None, answer=None, **over):
    args = args or dict(TOOL.expected_tool_call.args)
    dates = " and ".join(d.date.strftime("%B %d, %Y") for d in TOOL.expected_result.deadlines)
    return resp(retrieved=("1005.11(c)", OTHER_CHUNK), cites=("1005.11(c)(1)",),
                tool_calls=[{"name": "dispute_timeline", "args": args, "result": dispute_timeline(**args)}],
                answer=answer or f"Investigate by {dates}.", **over)


def judged(label="faithful", covered=None, n=2, unsupported=()):
    covered = n if covered is None else covered
    return P.Verdict("judged", label, covered, n, list(unsupported))


def tags(item, r, v=None):
    return [t for t, _ in P.tags_for(item, R.grade(item, r, IX), v or judged(n=len(item.key_facts or [])))]


def test_the_items_used_here_are_what_the_tests_assume():
    assert (PLAIN.regulation, PLAIN.persona, PLAIN.gold_ids, len(PLAIN.key_facts)) == ("Z", "frontline", ["1026.10(b)(2)(ii)"], 2)
    assert TWO_GOLD.gold_ids == ["1026.9(c)(2)(i)", "1026.55(b)(3)"] and TOOL.expected_tool_call.args["regulation"] == "E"
    assert (NETWORK.abstain_category, STATE.abstain_category) == ("network_rules", "state_law")
    assert OTHER_CHUNK in IX.chunk_provisions and IX.status[NOT_IN_FORCE] != "in_force"
    assert NOT_IN_FORCE in IX.chunk_provisions["1026.2(a)(15)#2"]


def test_tags_are_the_spec_taxonomy_plus_incomplete():
    spec = {"retrieval_miss", "partial_retrieval", "wrong_reg", "status_error", "tool_misuse", "calculation_error",
            "memory_leak", "unsupported_claim", "wrong_citation", "missed_abstention", "over_abstention"}
    assert set(P.TAGS) == spec | {"incomplete"} and len(P.TAGS) == 12 and P.PASS not in P.TAGS


# ---------------------------------------------------------------- outside-law keywords

@pytest.mark.parametrize("claim", [
    "Visa's zero liability policy means she owes nothing", "Under the network rules the bank must refund",
    "State law may give the customer more time", "Under California law the deadline is longer",
    "The Fair Credit Reporting Act governs the dispute with the bureau", "FCRA requires the bureau to investigate",
    "the Act requires a written notice", "Regulation CC sets the hold period", "Reg F covers the collector",
    "a court could hold the issuer liable", "the issuer's own terms waive the fee", "the cardholder agreement allows it",
    "the statute of limitations is three years", "she can file a chargeback with Mastercard",
])
def test_outside_law_finds(claim):
    assert P.outside_law(claim)


@pytest.mark.parametrize("claim", [
    "She discovered the loss on Friday", "The creditor must state that the rate may increase",
    "the consumer acted within 60 days", "Regulation E caps liability at $50", "Under Regulation Z the issuer must",
    "Reg Z requires an acknowledgement", "the institution must act on the notice", "a courtesy notice is not required",
    "the prepaid account agreement must disclose fees", "subject to the terms and conditions of the account",
    "the transfer was made through a debit card network",
])
def test_outside_law_leaves(claim):
    assert not P.outside_law(claim)


# Every unsupported claim the judge (j2) found in dev run r20261106T081540Z (vector k5), word for word,
# with whether it names something outside Z and E. Session 21 set the keyword list from these.
REAL_CLAIMS = [
    ("q004", False, "an unauthorized ACH debit after cancellation qualifies as an 'error'"),
    ("q005", False, "This is required by §§ 1026.9(c) or (g)"),
    ("q006", False, "That question is governed by § 1026.10(b)"),
    ("q007", False, "The debit card transactions that drained the checking account balance are governed by Regulation E, "
                    "which covers unauthorized electronic fund transfers from a consumer's bank account"),
    ("q007", False, "The overdraft credit line is a separate open-end credit feature"),
    ("q007", False, "Unauthorized use of a credit card or credit feature is governed by Regulation Z"),
    ("q008", False, "Under federal law (Regulation E), a cardholder's liability for unauthorized use of a debit card is "
                    "governed by § 1005.6(b), not Regulation Z's credit card rules"),
    ("q009", False, "Regulation Z does not set a deadline for the credit bureaus themselves to correct a report"),
    ("q009", True, "The timeline for the bureaus to act on that notification is governed by the Fair Credit Reporting "
                   "Act and its implementing regulations"),
    ("q010", False, "The applicable rules for that situation are in Subpart B of Regulation E (the remittance transfer "
                    "provisions, around § 1005.31 et seq.)"),
    ("q016", False, "The prohibited practices listed in § 1026.56(j) apply '[n]otwithstanding a consumer's affirmative "
                    "consent,' meaning the consent requirement is a threshold condition that must be satisfied before "
                    "any fee is permissible"),
    ("q024", True, "This question turns on which state's usury or interest-rate laws apply to a credit card account — a "
                   "matter of state law and potentially federal banking statutes, not Regulation Z (12 CFR Part 1026)."),
    ("q028", False, "A checking account is a deposit account, not an open-end credit plan"),
    ("q028", False, "Regulation Z's account-opening disclosure rules (§ 1026.5(b)(1)(i), § 1026.6(b)) do not apply to it "
                    "[a checking account]"),
    ("q028", False, "Regulation E (12 CFR Part 1005) governs electronic fund transfers associated with checking accounts"),
    ("q028", False, "providing a compliance sign-off or clearance determination is outside what I can do under these regulations"),
    ("q044", True, "The federal rules that directly regulate those practices (such as restrictions on calling hours and "
                   "workplace contacts) fall outside Regulation Z and Regulation E entirely"),
    ("q052", False, "Her call on Tuesday, 10 November 2026 is three business days before Sunday, 15 November 2026 "
                    "(counting Wednesday, Thursday, and Friday)"),
]


@pytest.mark.parametrize("qid,want,claim", REAL_CLAIMS)
def test_outside_law_on_a_real_run(qid, want, claim):
    assert P.outside_law(claim) is want


@pytest.mark.parametrize("claim,want", [
    ("under federal law (Regulation Z) the cap is $50", False), ("federal regulations, Reg E, apply", False),
    ("the federal law in 12 CFR Part 1005 applies", False), ("federal law may give the customer more", True),
    ("federal regulations on debt collection apply", True),
])
def test_federal_law_unless_z_or_e_follows(claim, want):
    assert P.outside_law(claim) is want


# ---------------------------------------------------------------- answerable questions: each tag and the order

def test_a_clean_answer_passes():
    assert tags(PLAIN, resp()) == []
    assert tags(TOOL, tool_resp()) == []


def test_retrieval_miss_comes_before_the_decline_it_causes():
    r = resp(retrieved=(OTHER_CHUNK,), cites=(), abstained=True)
    assert tags(PLAIN, r, judged(covered=0)) == ["retrieval_miss", "over_abstention"]


def test_over_abstention_records_its_cause_and_hides_its_consequences():
    m = R.grade(TOOL, resp(retrieved=("1005.11(c)",), cites=(), abstained=True, flags=["no_valid_citation"]), IX)
    got = P.tags_for(TOOL, m, P.Verdict("skipped", kf_total=None))
    assert got == [("over_abstention", "declined (cause: no_valid_citation)")]   # no missed call, no key facts


def test_wrong_reg_and_status_error():
    assert tags(PLAIN, resp(cites=("1026.10(b)(2)(ii)", "1005.6(b)(1)")))[0] == "wrong_reg"
    assert tags(PLAIN, resp(retrieved=("1026.10(b)", "1026.2(a)(15)#2"), cites=("1026.10(b)(2)(ii)", NOT_IN_FORCE))) \
        == ["status_error"]


def test_tool_misuse_and_calculation_error():
    wrong = dict(TOOL.expected_tool_call.args, notice_date="2026-11-06")
    got = tags(TOOL, tool_resp(args=wrong, answer="Investigate by November 23, 2026."))
    assert got == ["tool_misuse", "calculation_error"]
    assert tags(TOOL, tool_resp(answer="Investigate by November 20, 2026.")) == ["calculation_error"]
    m = R.grade(TOOL, tool_resp(args=wrong), IX)
    assert P.tags_for(TOOL, m, judged())[0][1] == "wrong_args (notice_date)"


def test_judge_claims_split_into_memory_leak_and_unsupported_claim():
    v = judged("partially_faithful", unsupported=["Visa's zero liability covers her", "the fee is $40"])
    assert tags(PLAIN, resp(), v) == ["memory_leak", "unsupported_claim"]
    assert tags(PLAIN, resp(), judged("unfaithful", unsupported=["the bank must reply in 5 days"])) == ["unsupported_claim"]


def test_wrong_citation_but_not_for_a_citation_form():
    assert tags(PLAIN, resp(cites=("1026.10(b)(2)(ii)", "1026.11(a)(1)"))) == ["wrong_citation"]   # real ID, not given
    assert tags(PLAIN, resp(cites=("1026.10(b)(2)(ii)", "1026.99(z)"))) == ["wrong_citation"]      # not in the index
    assert tags(PLAIN, resp(cites=("1026.10(b)(2)(ii)", "§ 1026.10"))) == []                       # a whole section


def test_missing_key_facts_partial_retrieval_or_incomplete():
    both = ("1026.9(c)(2)#1", "1026.55(b)(3)")
    assert tags(TWO_GOLD, resp(retrieved=both, cites=("1026.55(b)(3)",)), judged(covered=1)) == ["incomplete"]
    assert tags(TWO_GOLD, resp(retrieved=("1026.55(b)(3)", OTHER_CHUNK), cites=("1026.55(b)(3)",)),
                judged(covered=1)) == ["partial_retrieval"]
    assert tags(TWO_GOLD, resp(retrieved=both, cites=("1026.55(b)(3)",)), judged(covered=2)) == []


def test_no_key_fact_tag_after_a_retrieval_miss():
    assert tags(PLAIN, resp(retrieved=(OTHER_CHUNK,), cites=()), judged(covered=0)) == ["retrieval_miss"]


def test_an_unneeded_call_is_checked_last():
    call = {"name": "dispute_timeline", "args": {"regulation": "Z"}, "result": {"error": "x"}}
    assert tags(PLAIN, resp(tool_calls=[call])) == ["tool_misuse"]
    got = P.tags_for(PLAIN, R.grade(PLAIN, resp(tool_calls=[call]), IX), judged(covered=1))
    assert [t for t, _ in got] == ["incomplete", "tool_misuse"] and "no call was needed" in got[1][1]


def test_without_a_verdict_the_judge_tags_cannot_fire():
    assert tags(PLAIN, resp(), P.Verdict("missing")) == []
    assert tags(PLAIN, resp(), P.Verdict("error")) == []


# ---------------------------------------------------------------- should-decline questions

def test_declines():
    no = resp(retrieved=("1005.6", OTHER_CHUNK), cites=(), abstained=True)
    assert tags(NETWORK, no) == []
    # decision 79: a correct decline that names an outside law is a memory_leak; any other
    # unsupported claim in it is an unsupported_claim (Prit's (a), after the first real report)
    assert tags(NETWORK, no, judged("partially_faithful", unsupported=["Visa's zero liability covers her"])) \
        == ["memory_leak"]
    assert tags(NETWORK, no, judged("partially_faithful", unsupported=["governed by Subpart B of Regulation E"])) \
        == ["unsupported_claim"]
    yes = resp(retrieved=("1005.6", OTHER_CHUNK), cites=("1005.6(b)(1)",))
    assert tags(NETWORK, yes, judged("unfaithful", unsupported=["the bank must refund in 5 days", "Visa's zero liability"])) \
        == ["missed_abstention", "memory_leak", "unsupported_claim"]
    m = R.grade(STATE, yes, IX)
    assert P.tags_for(STATE, m, judged())[0] == ("missed_abstention", "answered; should decline (state_law)")


# ---------------------------------------------------------------- a whole run

RUN = "r20990101T000000Z"


def add_verdict(db, qid, *, status="judged", label="faithful", covered=2, total=2, unsupported=(), version="j2", run=RUN):
    claims = [{"claim": "ok", "supported": True, "source": "passage 1"}] + [
        {"claim": c, "supported": False, "source": None} for c in unsupported]
    vj = None if status != "judged" else json.dumps({
        "claims": claims, "faithfulness": label, "rationale": "why",
        "key_facts": [{"n": n, "covered": n <= covered} for n in range(1, total + 1)]})
    db.execute("INSERT INTO judgements (run_id, qid, judge_version, status, faithfulness, kf_covered, kf_total, "
               "verdict_json) VALUES (?,?,?,?,?,?,?,?)",
               (run, qid, version, status, label if status == "judged" else None,
                covered if status == "judged" else None, total if status == "judged" else None, vj))
    db.commit()


@pytest.fixture
def db(tmp_path):
    d = J.connect(tmp_path / "runs.db")
    d.execute("INSERT INTO runs (run_id, split, git_commit, git_dirty, config_json, prompt_version, grader_version, "
              "n_items, today_default) VALUES (?,?,?,?,?,?,?,?,?)",
              (RUN, "dev", "437f410abc", 0, json.dumps({"search_mode": "hybrid", "k": 5, "tool_on": True}), "p2",
               "g4", 6, "2026-11-03"))
    answers = {
        "q006": resp(),                                                                  # passes
        "q005": resp(retrieved=("1026.55(b)(3)", OTHER_CHUNK), cites=("1026.55(b)(3)",)),   # partial_retrieval
        "q004": tool_resp(answer="Investigate by November 20, 2026."),                    # calculation_error
        "q016": resp(retrieved=(OTHER_CHUNK,), cites=(), abstained=True, flags=["no_valid_citation"]),
        "q008": resp(retrieved=("1005.6",), cites=(), abstained=True),                     # memory_leak (decline)
        "q024": resp(retrieved=(OTHER_CHUNK,), cites=(), abstained=True),                  # passes
    }
    for qid, r in answers.items():
        item = BY_ID[qid]
        R.log_result(d, RUN, item, r, None, R.grade(item, r, IX))
    R.log_result(d, RUN, BY_ID["q049"], None, "ThrottlingException: slow down", None)
    add_verdict(d, "q006")
    add_verdict(d, "q005", label="partially_faithful", covered=1, unsupported=["the fee is $40"])
    add_verdict(d, "q004", covered=1)
    add_verdict(d, "q016", status="skipped")
    add_verdict(d, "q008", label="partially_faithful", unsupported=["Visa's zero liability covers her"])
    # q024: no verdict yet
    return d


def test_build_counts_slices_judge_and_tags(db):
    rep = P.build(db, RUN, ix=IX)
    o = rep["overall"]
    assert (o["n"], o["answerable"], o["declines"]) == (6, 4, 2)
    assert o["faithful"] == (2, 4)                 # judged: q006, q005, q004, q008; q016 skipped; q024 none
    assert o["key_facts"] == (2 + 1 + 1 + 0, 2 + 2 + 2 + 3)   # q016's fallback covers 0 of its 3
    assert o["failed"] == (4, 6) and o["abstain_correct"] == (5, 6)
    assert {f["id"]: f["primary"] for f in rep["failures"]} == {
        "q004": "calculation_error", "q005": "unsupported_claim", "q008": "memory_leak", "q016": "retrieval_miss"}
    assert rep["tag_counts"]["memory_leak"] == 1
    assert rep["declines_by_category"] == {"network_rules": (1, 1, 1), "state_law": (1, 1, 0)}
    assert (rep["by_persona"]["frontline"]["n"], rep["by_persona"]["compliance"]["n"]) == (3, 3)
    assert rep["no_verdict"] == [("q024", "missing")] and rep["not_run"][0][0] == "q049"
    assert rep["run_name"] == "hybrid k5"


def test_q005_tags_in_order(db):
    f = next(f for f in P.build(db, RUN, ix=IX)["failures"] if f["id"] == "q005")
    assert [t for t, _ in f["tags"]] == ["unsupported_claim", "partial_retrieval"]


def test_type_slices_follow_the_golden_order(db):
    rep = P.build(db, RUN, ix=IX)
    assert list(rep["by_type"]) == [t for t in G.TYPES if t in {"payments_disclosures", "fees_terms", "deadline",
                                                                 "decline"}]      # q049 (dispute) failed to run
    assert list(rep["by_persona"]) == ["frontline", "compliance"]


def test_an_earlier_rubric_reads_its_own_verdicts(db):
    add_verdict(db, "q006", label="partially_faithful", unsupported=["x"], version="j1")
    rep = P.build(db, RUN, version="j1", ix=IX)
    assert rep["overall"]["judged"] == 1 and "this report uses j1" in P.text(rep)


def test_text_and_markdown(db):
    rep = P.build(db, RUN, ix=IX)
    t, md = P.text(rep), P.markdown(rep)
    for out in (t, md):
        assert "upper bound" in out and "lower bound" in out and "q024 (missing)" in out
        assert f"Graded again with {R.GRADER_VERSION} (the run was logged with g4)." in out
        assert "too few to compare" in out
    assert "| all | 6 |" in md and "| q008 | frontline | decline | memory_leak |" in md
    assert "4 of 6 questions failed" in t


def test_show(db):
    out = P.show(db, RUN, "q005")
    assert TWO_GOLD.question.strip()[:40] in out and "unsupported_claim, partial_retrieval" in out
    assert "partially_faithful" in out
    assert "Failed to run" in P.show(db, RUN, "q049")
    with pytest.raises(SystemExit):
        P.show(db, RUN, "q001")


# ---------------------------------------------------------------- the author's review

def write(path, rep, fill=None):
    text = P.review_sheet(rep)
    for qid, (tag, note) in (fill or {}).items():
        text = text.replace(f'id = "{qid}"', f'id = "{qid}"\n#x', 1)
        head, rest = text.split(f'id = "{qid}"\n#x', 1)
        rest = rest.replace('tag = ""', f'tag = "{tag}"', 1).replace('note = ""', f'note = "{note}"', 1)
        text = head + f'id = "{qid}"' + rest
    path.write_text(text, encoding="utf-8")
    return path


def test_review_sheet_lists_every_failure_with_its_automatic_tag(db, tmp_path):
    rep = P.build(db, RUN, ix=IX)
    sheet = P.review_sheet(rep)
    assert "For declines: missed_abstention, then memory_leak, then unsupported_claim." in sheet
    blocks = tomllib.loads(sheet)["review"]
    assert [(b["id"], b["auto"], b["tag"]) for b in blocks] == [
        ("q004", "calculation_error", ""), ("q005", "unsupported_claim", ""),
        ("q008", "memory_leak", ""), ("q016", "retrieval_miss", "")]


def test_review_in_progress_then_complete(db, tmp_path):
    rep = P.build(db, RUN, ix=IX)
    path = write(tmp_path / "r.toml", rep, {"q004": ("calculation_error", "")})
    assert P.read_review(path, rep) == (None, "review in progress: 1 of 4 reviewed (r.toml)")
    path = write(tmp_path / "r.toml", rep, {"q004": ("calculation_error", ""), "q005": ("incomplete", "fee is in passage 2"),
                                            "q008": ("pass", "judge misread"), "q016": ("retrieval_miss", "")})
    review, state = P.read_review(path, rep)
    assert state == "reviewed by the author (r.toml)"
    done = P.build(db, RUN, review=review, ix=IX)
    assert done["review"]["counts"] == {"calculation_error": 1, "incomplete": 1, "retrieval_miss": 1}
    assert done["review"]["passed"] == ["q008"]
    assert done["review"]["changed"] == [("q005", "unsupported_claim", "incomplete", "fee is in passage 2"),
                                         ("q008", "memory_leak", "pass", "judge misread")]
    t = P.text(done)
    assert "after the review 3" in t and "q005: unsupported_claim -> incomplete (fee is in passage 2)" in t


def test_review_refuses_a_sheet_that_no_longer_matches(db, tmp_path):
    rep = P.build(db, RUN, ix=IX)
    path = write(tmp_path / "r.toml", rep)
    add_verdict(db, "q024", label="partially_faithful", unsupported=["state law gives more"])   # a new failure
    with pytest.raises(SystemExit, match="missing: q024"):
        P.read_review(path, P.build(db, RUN, ix=IX))
    bad = tmp_path / "bad.toml"
    bad.write_text(P.review_sheet(rep).replace('tag = ""', 'tag = "hallucination"', 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown tag 'hallucination'"):
        P.read_review(bad, rep)


def test_command_line_review_sheet_and_markdown(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(P, "REVIEW_DIR", tmp_path / "review")
    dbfile = tmp_path / "runs.db"
    assert P.main([RUN, "--review-sheet", "--db", str(dbfile)]) == 0
    sheet = tmp_path / "review" / f"{RUN}.toml"
    assert sheet.exists() and "4 failures to review" in capsys.readouterr().out
    assert P.main([RUN, "--review-sheet", "--db", str(dbfile)]) == 2          # never overwritten
    md = tmp_path / "new folder" / "report.md"
    assert P.main([RUN, "--db", str(dbfile), "--markdown", str(md)]) == 0
    out = capsys.readouterr().out
    assert "review in progress: 0 of 4 reviewed" in out and md.read_text(encoding="utf-8").startswith("# Evaluation report")
    assert P.main([RUN, "--db", str(tmp_path / "none.db")]) == 2


# ---------------------------------------------------------------- two runs side by side (--compare)

RUN_B = "r20990102T000000Z"


def add_run(db, run_id, config, commit="a919b13abc", dirty=0, prompt="p2"):
    db.execute("INSERT INTO runs (run_id, split, git_commit, git_dirty, config_json, prompt_version, grader_version, "
               "n_items, today_default) VALUES (?,?,?,?,?,?,?,?,?)",
               (run_id, "dev", commit, dirty, json.dumps(config), prompt, "g5", 6, "2026-11-03"))


@pytest.fixture
def two(db):
    """Run A is the `db` fixture's run (hybrid k5). Run B (vector k5) changes one thing per question."""
    add_run(db, RUN_B, {"search_mode": "vector", "k": 5, "tool_on": True})
    answers = {
        "q006": resp(retrieved=(OTHER_CHUNK, "1026.10(b)")),                                 # rank 1 -> 2
        "q005": resp(retrieved=("1026.55(b)(3)", OTHER_CHUNK), cites=("1026.55(b)(3)",)),     # judge better
        "q004": tool_resp(),                                                                  # calculation fixed
        "q016": resp(retrieved=(OTHER_CHUNK,), cites=(), abstained=True, flags=["no_valid_citation"]),  # same
        "q008": resp(retrieved=("1005.6",), cites=("1005.6(b)(1)",)),                         # now answered
        "q024": resp(retrieved=(OTHER_CHUNK,), cites=(), abstained=True),                     # same, now judged
        "q049": resp(retrieved=("1005.11(c)",), cites=("1005.11(c)(1)",)),                    # failed in A
        "q052": resp(),                                                                       # only in B
    }
    for qid, r in answers.items():
        item = BY_ID[qid]
        R.log_result(db, RUN_B, item, r, None, R.grade(item, r, IX))
    add_verdict(db, "q006", run=RUN_B)
    add_verdict(db, "q005", run=RUN_B, label="faithful", covered=2)
    add_verdict(db, "q004", run=RUN_B, covered=2)
    add_verdict(db, "q016", run=RUN_B, status="skipped")
    add_verdict(db, "q008", run=RUN_B, label="partially_faithful", unsupported=["Visa's zero liability covers her"])
    add_verdict(db, "q024", run=RUN_B, covered=3, total=3)
    add_verdict(db, "q049", run=RUN_B)
    add_verdict(db, "q052", run=RUN_B)
    return db


@pytest.mark.parametrize("b,c,p", [(0, 0, 1.0), (1, 1, 1.0), (3, 1, 0.625), (5, 0, 0.0625), (6, 0, 0.03125),
                                   (0, 6, 0.03125), (10, 2, 158 / 4096)])
def test_mcnemar_exact(b, c, p):
    # Reference values: statsmodels.stats.contingency_tables.mcnemar(exact=True), checked in session 22.
    assert P.mcnemar_p(b, c) == pytest.approx(p)


def test_compare_pairs_the_questions_both_runs_answered(two):
    c = P.compare(two, RUN, RUN_B, ix=IX)
    assert c["common"] == ["q004", "q005", "q006", "q008", "q016", "q024"]
    assert c["failed"] == [("q049", "A")] and c["only_b"] == ["q052"] and c["only_a"] == [] and c["relabelled"] == []
    assert c["settings"] == [("search_mode", "hybrid", "vector")] and c["dirty"] == []
    assert (c["a"]["name"], c["b"]["name"]) == ("hybrid k5", "vector k5")
    assert c["no_verdict"] == {"A": ["q024"], "B": []}


def test_compare_counts_each_measure_question_by_question(two):
    blk = P.compare(two, RUN, RUN_B, ix=IX)["slices"]["all"]
    got = {d["key"]: (d["n"], d["a"], d["b"], d["both"], d["b_only"], d["a_only"], d["neither"]) for d in blk["paired"]}
    assert got["hit_at_5"] == (4, 3, 3, 3, 0, 0, 1)            # answerable q004 q005 q006 q016; q016 misses in both
    assert got["declined"] == (2, 2, 1, 1, 0, 1, 0)            # q008 answered in B
    assert got["abstain_ok"] == (6, 5, 4, 4, 0, 1, 1)          # q016 over-abstains in both
    assert got["calc"] == (1, 0, 1, 0, 1, 0, 0)                # q004
    assert got["faithful"] == (4, 2, 3, 2, 1, 0, 1)            # judged in both: q006 q005 q004 q008
    assert got["no_leak"] == (2, 1, 1, 1, 0, 0, 1)             # q008 leaks in both
    assert got["answered"] == (4, 3, 3, 3, 0, 0, 1)            # q016 declined (guard) in both
    assert got["tool_ok"] == (6, 6, 6, 6, 0, 0, 0)
    assert got["passed"] == (6, 2, 4, 2, 2, 0, 2)              # both: q006 q024; B fixes q004 q005; q008 q016 fail in both
    ordered = {d["key"]: (d["n"], d["b_better"], d["b_worse"], d["same"]) for d in blk["ordered"]}
    assert ordered["rank"] == (4, 0, 1, 3)                     # q006 rank 1 -> 2
    assert ordered["key_facts"] == (4, 2, 0, 2)                # q005 1 -> 2, q004 1 -> 2
    assert {d["key"]: d["p"] for d in blk["ordered"]}["key_facts"] == 0.5


def test_compare_slices_categories_tags_and_changes(two):
    c = P.compare(two, RUN, RUN_B, ix=IX)
    assert list(c["slices"]) == ["all", "frontline", "compliance"]
    assert c["declines_by_category"] == {"network_rules": ((1, 1, 1), (0, 1, 1)), "state_law": ((1, 1, 0), (1, 1, 0))}
    ta, tb = c["tag_counts"]
    assert ta["calculation_error"] == 1 and tb["calculation_error"] == 0 and tb["missed_abstention"] == 1
    ch = {q: x for q, _, _, x in c["changes"]}
    assert set(ch) == {"q004", "q005", "q006", "q008", "q024"}
    assert ch["q006"] == ["first gold rank 1 → 2"]
    assert "calculation False → True" in ch["q004"] and "key facts 1 → 2 of 2" in ch["q004"]
    assert "declined → answered" in ch["q008"] and ch["q008"][-1] == "tags memory_leak → missed_abstention, memory_leak"
    assert ch["q024"] == ["judge missing → faithful"]


def test_compare_text_and_markdown(two):
    c = P.compare(two, RUN, RUN_B, ix=IX)
    t = P.compare_text(c)
    assert "Settings that differ: search_mode hybrid → vector" in t and "WARNING: more than one" not in t
    assert "couldn't check" in t                    # made-up commits aren't in this repo
    assert "Left out (failed to run): q049 (in A)" in t and "Left out (in one run only): A none; B q052" in t
    assert "Questions that changed: 5 of 6" in t and "McNemar" in t and "upper bound" in t
    md = P.compare_markdown(c)
    assert md.startswith("# Run comparison") and "| hit@5 | 3/4 | 3/4 | 3 | 0 | 0 | 1 | - |" in md and "## Frontline (3)" in md


def test_compare_warns_about_more_than_one_setting_and_dirty_runs(db):
    add_run(db, RUN_B, {"search_mode": "vector", "k": 3, "tool_on": True}, dirty=1, prompt="p3")
    R.log_result(db, RUN_B, PLAIN, resp(), None, R.grade(PLAIN, resp(), IX))
    c = P.compare(db, RUN, RUN_B, ix=IX)
    assert c["settings"] == [("k", 5, 3), ("search_mode", "hybrid", "vector"), ("prompt_version", "p2", "p3")]
    t = P.compare_text(c)
    assert "WARNING: more than one setting differs" in t and "WARNING: run B was made with uncommitted changes" in t


def test_compare_leaves_out_a_question_relabelled_between_runs(db):
    add_run(db, RUN_B, {"search_mode": "hybrid", "k": 5, "tool_on": True})
    edited = PLAIN.model_copy(update={"key_facts": list(PLAIN.key_facts) + ["one more"]})
    R.log_result(db, RUN_B, edited, resp(), None, R.grade(edited, resp(), IX))
    R.log_result(db, RUN_B, TWO_GOLD, resp(), None, R.grade(TWO_GOLD, resp(), IX))
    c = P.compare(db, RUN, RUN_B, ix=IX)
    assert c["relabelled"] == ["q006"] and c["common"] == ["q005"]
    assert "none (a repeat of the same configuration)" in P.compare_text(c)


def test_compare_refuses_the_same_run_or_nothing_in_common(db):
    with pytest.raises(SystemExit, match="two different runs"):
        P.compare(db, RUN, RUN, ix=IX)
    add_run(db, RUN_B, {"search_mode": "hybrid", "k": 5, "tool_on": True})
    R.log_result(db, RUN_B, BY_ID["q052"], resp(), None, R.grade(BY_ID["q052"], resp(), IX))
    with pytest.raises(SystemExit, match="no question in common"):
        P.compare(db, RUN, RUN_B, ix=IX)


def test_app_changed(monkeypatch):
    assert P.app_changed("abc", "abc") is False and P.app_changed(None, "abc") is None
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: type("Done", (), {"returncode": 1})())
    assert P.app_changed("abc", "def") is True
    monkeypatch.setattr(P.subprocess, "run", lambda *a, **k: type("Done", (), {"returncode": 128})())
    assert P.app_changed("abc", "def") is None


def test_command_line_compare(two, tmp_path, capsys):
    md = tmp_path / "out" / "cmp.md"
    assert P.main([RUN, "--compare", RUN_B, "--db", str(tmp_path / "runs.db"), "--markdown", str(md)]) == 0
    out = capsys.readouterr().out
    assert "A: r20990101T000000Z (hybrid k5)" in out and md.read_text(encoding="utf-8").startswith("# Run comparison")
    with pytest.raises(SystemExit):
        P.main([RUN, "--compare", RUN_B, "--show", "q006", "--db", str(tmp_path / "runs.db")])


def test_compare_reads_the_prompt_once_and_names_it(db):
    # From session 22 config_json also holds "prompt": the comparison mustn't count it twice.
    add_run(db, RUN_B, {"search_mode": "hybrid", "k": 5, "tool_on": True, "prompt": "p3"}, prompt="p3")
    R.log_result(db, RUN_B, PLAIN, resp(), None, R.grade(PLAIN, resp(), IX))
    c = P.compare(db, RUN, RUN_B, ix=IX)
    assert c["settings"] == [("prompt_version", "p2", "p3")]
    assert c["b"]["name"] == "hybrid k5 prompt p3" and c["a"]["name"] == "hybrid k5"
    assert "WARNING: more than one" not in P.compare_text(c)


# ---------------------------------------------------------------- the tool off (experiment 3)

def test_a_missed_call_is_no_tool_misuse_when_the_run_had_no_tool():
    # Both dates missing and no call: with the tool on that's tool_misuse then calculation_error; with
    # the tool off only the calculation is wrong.
    r = resp(retrieved=("1005.11(c)", OTHER_CHUNK), cites=("1005.11(c)(1)",), answer="We will look into it.")
    m = R.grade(TOOL, r, IX)
    assert m["tool_outcome"] == "missed_call"
    v = judged(n=len(TOOL.key_facts))
    assert [t for t, _ in P.tags_for(TOOL, m, v)][:2] == ["tool_misuse", "calculation_error"]
    assert [t for t, _ in P.tags_for(TOOL, m, v, tool_on=False)] == ["calculation_error"]
    right = tool_resp()
    no_tool = resp(retrieved=right["retrieved_ids"], cites=right["citations"], answer=right["answer"])
    assert P.tags_for(TOOL, R.grade(TOOL, no_tool, IX), v, tool_on=False) == []


def test_load_rows_reads_whether_the_tool_was_on(db):
    add_run(db, RUN_B, {"search_mode": "hybrid", "k": 5, "tool_on": False, "prompt": "p4"}, prompt="p4")
    r = resp(retrieved=("1005.11(c)", OTHER_CHUNK), cites=("1005.11(c)(1)",), answer="We will look into it.")
    R.log_result(db, RUN_B, TOOL, r, None, R.grade(TOOL, r, IX))
    _, rows = P.load_rows(db, RUN_B, ix=IX)
    assert [t for t, _ in rows[0].tags] == ["calculation_error"]
    assert P.compare(db, RUN, RUN_B, ix=IX)["b"]["name"] == "hybrid k5 no tool prompt p4"


def test_a_setting_missing_from_an_older_log_counts_as_its_default(db):
    # Run A's log (made before session 22) has no "interpretations" key: it used the default (True).
    add_run(db, RUN_B, {"search_mode": "hybrid", "k": 5, "tool_on": True, "interpretations": True})
    R.log_result(db, RUN_B, PLAIN, resp(), None, R.grade(PLAIN, resp(), IX))
    assert P.compare(db, RUN, RUN_B, ix=IX)["settings"] == []
    db.execute("UPDATE runs SET config_json = ? WHERE run_id = ?",
               (json.dumps({"search_mode": "hybrid", "k": 5, "tool_on": True, "interpretations": False}), RUN_B))
    c = P.compare(db, RUN, RUN_B, ix=IX)
    assert c["settings"] == [("interpretations", True, False)] and c["b"]["name"] == "hybrid k5 rules only"
