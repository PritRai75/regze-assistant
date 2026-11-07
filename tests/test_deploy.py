"""Tests for the deployment files: template.yaml, samconfig.toml and deploy/stage.py. They read the files
only; nothing is deployed and no AWS call is made."""
import json
import tomllib
from pathlib import Path

import pytest
import yaml

from app.config import DEFAULT
from deploy import stage as st

ROOT = Path(__file__).resolve().parents[1]


class CfnLoader(yaml.SafeLoader):
    """Reads CloudFormation's short tags (!Ref, !Sub, !GetAtt) as {"Ref": ...}-style dicts."""


for _tag, _key in (("!Ref", "Ref"), ("!Sub", "Fn::Sub"), ("!GetAtt", "Fn::GetAtt")):
    CfnLoader.add_constructor(_tag, lambda loader, node, key=_key: {key: loader.construct_scalar(node)})

TEMPLATE = yaml.load((ROOT / "template.yaml").read_text(encoding="utf-8"), Loader=CfnLoader)
RES = TEMPLATE["Resources"]
FUNCTION = RES["AskFunction"]["Properties"]
STATEMENTS = {s["Sid"]: s for p in RES["AskFunctionRole"]["Properties"]["Policies"]
              for s in p["PolicyDocument"]["Statement"]}
SAMCONFIG = tomllib.loads((ROOT / "samconfig.toml").read_text(encoding="utf-8"))
META = json.loads((ROOT / "index" / "meta.json").read_text(encoding="utf-8"))


def default(name):
    return TEMPLATE["Parameters"][name]["Default"]


# ---------------------------------------------------------------- the template matches the app

def test_generator_model_is_the_configs_global_profile():
    assert DEFAULT.generator_model == "global." + default("GeneratorModel")


def test_embedding_model_and_region_match_the_index():
    assert default("EmbeddingModel") == META["model_id"]
    assert SAMCONFIG["default"]["global"]["parameters"]["region"] == META["region"]


def test_function_settings():
    assert FUNCTION["Handler"] == "app.handler.lambda_handler" and (ROOT / "app" / "handler.py").exists()
    assert FUNCTION["Runtime"] == "python3.13"
    assert FUNCTION["CodeUri"].rstrip("/") == ".aws-sam/stage"
    assert FUNCTION["Timeout"] <= 29                     # API Gateway's integration timeout
    assert FUNCTION["Role"] == {"Fn::GetAtt": "AskFunctionRole.Arn"}
    assert FUNCTION["LoggingConfig"] == {"LogGroup": {"Ref": "AskFunctionLogs"}}
    assert FUNCTION["Events"]["Ask"]["Properties"] == {"RestApiId": {"Ref": "CardRulesApi"}, "Path": "/ask",
                                                       "Method": "post"}


# ---------------------------------------------------------------- least privilege

def test_role_allows_only_invoke_model_and_its_own_logs():
    actions = {a for s in STATEMENTS.values() for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])}
    assert actions == {"bedrock:InvokeModel", "logs:CreateLogStream", "logs:PutLogEvents"}
    assert all(s["Effect"] == "Allow" for s in STATEMENTS.values())
    for s in STATEMENTS.values():
        resource = s["Resource"]
        assert resource != "*" and "*" not in json.dumps(resource)
    assert STATEMENTS["WriteOwnLogs"]["Resource"] == {"Fn::GetAtt": "AskFunctionLogs.Arn"}
    assert "ManagedPolicyArns" not in RES["AskFunctionRole"]["Properties"]      # no AWSLambdaBasicExecutionRole


def test_bedrock_statements_follow_the_global_profile_policy():
    profile = "arn:${AWS::Partition}:bedrock:${AWS::Region}:${AWS::AccountId}:inference-profile/global.${GeneratorModel}"
    assert STATEMENTS["EmbedQuestions"]["Resource"] == {
        "Fn::Sub": "arn:${AWS::Partition}:bedrock:${AWS::Region}::foundation-model/${EmbeddingModel}"}
    assert "Condition" not in STATEMENTS["EmbedQuestions"]
    s = STATEMENTS["GeneratorProfileInRegion"]
    assert s["Resource"] == {"Fn::Sub": profile}
    assert s["Condition"] == {"StringEquals": {"aws:RequestedRegion": {"Ref": "AWS::Region"}}}
    s = STATEMENTS["GeneratorModelInRegion"]
    assert s["Resource"] == {"Fn::Sub": "arn:${AWS::Partition}:bedrock:${AWS::Region}::foundation-model/${GeneratorModel}"}
    assert s["Condition"] == {"StringEquals": {"aws:RequestedRegion": {"Ref": "AWS::Region"},
                                               "bedrock:InferenceProfileArn": {"Fn::Sub": profile}}}
    s = STATEMENTS["GeneratorModelGlobal"]
    assert s["Resource"] == {"Fn::Sub": "arn:${AWS::Partition}:bedrock:::foundation-model/${GeneratorModel}"}
    assert s["Condition"] == {"StringEquals": {"aws:RequestedRegion": "unspecified",
                                               "bedrock:InferenceProfileArn": {"Fn::Sub": profile}}}


def test_judge_model_is_not_allowed():
    text = (ROOT / "template.yaml").read_text(encoding="utf-8")
    assert "4-5" not in text and "judge" not in text.lower()


# ---------------------------------------------------------------- the API

def test_api_needs_a_key_and_has_a_throttle_and_quota():
    api = RES["CardRulesApi"]["Properties"]
    assert api["Auth"]["ApiKeyRequired"] is True
    plan = api["Auth"]["UsagePlan"]
    assert plan["CreateUsagePlan"] == "PER_API"
    assert 0 < plan["Throttle"]["RateLimit"] <= 1 and 0 < plan["Throttle"]["BurstLimit"] <= 2
    assert plan["Quota"] == {"Limit": {"Ref": "DailyQuota"}, "Period": "DAY"}
    assert 0 < default("DailyQuota") <= 100
    assert TEMPLATE["Outputs"]["ApiKeyId"]["Value"] == {"Ref": "CardRulesApiApiKey"}    # SAM's name for PER_API


def test_log_retention_is_set():
    assert 1 <= RES["AskFunctionLogs"]["Properties"]["RetentionInDays"] <= 30


def test_samconfig_asks_before_changing_anything():
    deploy = SAMCONFIG["default"]["deploy"]["parameters"]
    assert deploy["confirm_changeset"] is True and deploy["capabilities"] == "CAPABILITY_IAM"


# ---------------------------------------------------------------- staging

def test_stage_copies_exactly_what_the_function_needs(tmp_path):
    copied = st.stage(dest=tmp_path / "stage")
    rels = [r for r, _, _ in copied]
    assert "app/handler.py" in rels and "app/answer.py" in rels
    assert set(rels) == set(st.app_files()) | set(st.FILES)
    assert not any(r.startswith(("eval/", "tests/", "corpus/", "ingest/", "scratch/")) for r in rels)
    for rel, size, _ in copied:
        assert (tmp_path / "stage" / rel).read_bytes() == (ROOT / rel).read_bytes()          # app/__init__.py is empty
    assert sum(s for _, s, _ in copied) < 15e6


def test_stage_starts_from_an_empty_folder(tmp_path):
    dest = tmp_path / "stage"
    (dest / "old").mkdir(parents=True)
    (dest / "old" / "leftover.txt").write_text("x")
    st.stage(dest=dest)
    assert not (dest / "old").exists()


@pytest.mark.parametrize("line", ["botocore[crt]==1.0", "boto3[crt]", "awscrt==0.20", "botocore [ crt ]"])
def test_crt_requirements_are_refused(line):
    with pytest.raises(ValueError, match="CRT"):
        st.check_requirements(f"numpy==2.5.4\n{line}\n")


def test_the_real_requirements_have_no_crt():
    st.check_requirements((ROOT / "requirements.txt").read_text(encoding="utf-8"))
    st.check_requirements("# boto3[crt] in a comment is fine\nboto3==1.43.123\n")


# ---------------------------------------------------------------- smoke test (fake API, no network)

from deploy import smoke  # noqa: E402

GOOD_ANSWER = {"answer": "x", "citations": ["1005.6(b)(1)"], "abstained": False, "model_calls": 2,
               "latency_ms": 5200, "config_hash": "b2bef7bd2dd0",
               "tool_calls": [{"name": "dispute_timeline", "args": {}, "result": {}}]}


def fake_api(replies, seen):
    def poster(url, body, key=None):
        seen.append((url, body, key))
        return replies[len(seen) - 1]
    return poster


def test_smoke_passes_on_expected_replies(capsys):
    seen = []
    replies = [(403, {"message": "Forbidden"}, 0.2), (200, GOOD_ANSWER, 9.5),
               (200, {**GOOD_ANSWER, "tool_calls": []}, 4.1),
               (200, {**GOOD_ANSWER, "citations": [], "abstained": True, "tool_calls": []}, 3.0),
               (400, {"error": "unknown field(s): questions"}, 0.3)]
    env = {"REGZE_API_URL": "https://example.test/v1/ask", "REGZE_API_KEY": "SECRET-KEY-VALUE"}
    assert smoke.main([], env=env, poster=fake_api(replies, seen)) == 0
    out = capsys.readouterr().out
    assert "5 of 5 checks passed" in out and "SECRET-KEY-VALUE" not in out
    assert seen[0][2] is None and all(k == "SECRET-KEY-VALUE" for _, _, k in seen[1:])


def test_smoke_fails_on_infrastructure_problems_and_notes_model_behaviour(capsys):
    seen = []
    replies = [(200, GOOD_ANSWER, 1.0),                                   # the API took no key: fail
               (200, {**GOOD_ANSWER, "tool_calls": []}, 9.5),             # tool never ran: fail
               (504, {"message": "Endpoint request timed out"}, 29.1),   # over the limit: fail
               (200, {**GOOD_ANSWER, "abstained": False}, 3.0),           # answered a decline: note only
               (400, {"error": "x"}, 0.3)]
    env = {"REGZE_API_URL": "u", "REGZE_API_KEY": "k"}
    assert smoke.main([], env=env, poster=fake_api(replies, seen)) == 1
    out = capsys.readouterr().out
    assert "2 of 5 checks passed" in out
    assert "status 200, expected 403" in out and "no dispute_timeline call" in out and "status 504" in out
    assert "PASS  out of scope" in out and "note: abstained False, expected True" in out


def test_smoke_needs_both_variables():
    with pytest.raises(SystemExit, match="REGZE_API_URL and REGZE_API_KEY"):
        smoke.main([], env={"REGZE_API_URL": "u"}, poster=None)


def test_smoke_questions_are_not_golden_questions():
    import re
    import tomllib as tl
    golden = tl.loads((ROOT / "eval" / "golden.toml").read_text(encoding="utf-8"))["question"]

    def runs(t, n=4):
        w = re.findall(r"[a-z0-9]+", t.lower())
        return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    golden_runs = set().union(*(runs(q["question"]) for q in golden))
    for call in smoke.CALLS:
        assert not runs(call["body"].get("question", "")) & golden_runs, call["name"]
