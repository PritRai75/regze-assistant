"""Tests for app/handler.py, the Lambda entry point. No AWS call: answer() runs with a fake Bedrock
client and a ready-made question vector, as in tests/test_answer.py."""
import base64
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from app import answer as an
from app import handler as h
from app import retrieve as rt
from app.config import DEFAULT

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "index" / "embeddings.npy").exists(), reason="index not built yet")

Q = "Under Reg E 6(b), what is the most a customer can lose?"
GOOD = {"answer": "At most $50 if reported within 2 business days [§ 1005.6(b)(1)].",
        "citations": ["1005.6(b)(1)", "1026.99(z)"], "abstained": False, "abstain_reason": None}


class Context:
    aws_request_id = "req-123"


class FakeClient:
    def __init__(self, *replies):
        self.replies = list(replies)

    def converse(self, **request):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def text(body):
    return {"output": {"message": {"role": "assistant", "content": [{"text": json.dumps(body)}]}},
            "stopReason": "end_turn", "usage": {"inputTokens": 1000, "outputTokens": 50}}


class ClientError(Exception):
    """Shaped like botocore's ClientError: the code is in .response["Error"]["Code"]."""
    def __init__(self, code):
        super().__init__(f"An error occurred ({code}) when calling the Converse operation: secret detail")
        self.response = {"Error": {"Code": code, "Message": "secret detail"}}


@pytest.fixture
def fake(monkeypatch):
    """Route handler.answer through a fake client; returns a list that records each call's (question, today)."""
    chunks, vecs, _, _ = rt.load()
    vec = vecs[[c["id"] for c in chunks].index("1005.6(b)-cmt")]
    calls, script = [], []

    def fake_answer(question, today):
        calls.append((question, today))
        return an.answer(question, today, config=DEFAULT, client=FakeClient(*script), query_vec=vec)
    monkeypatch.setattr(h, "answer", fake_answer)
    monkeypatch.setitem(h._state, "cold", False)
    return calls, script


def event(body, b64=False):
    raw = body if isinstance(body, str) else json.dumps(body)
    if b64:
        return {"body": base64.b64encode(raw.encode()).decode(), "isBase64Encoded": True}
    return {"body": raw, "isBase64Encoded": False}


def call(ev):
    out = h.lambda_handler(ev, Context())
    return out["statusCode"], json.loads(out["body"]), out


# ---------------------------------------------------------------- a good request

def test_answers_with_public_fields_only(fake):
    calls, script = fake
    script.append(text(GOOD))
    status, body, out = call(event({"question": Q, "today": "2026-11-03"}))
    assert status == 200
    assert out["headers"]["Content-Type"] == "application/json"
    assert list(body) == list(h.PUBLIC_FIELDS)
    for hidden in ("raw_answer", "raw_citations", "invalid_citations"):
        assert hidden not in body
    assert calls == [(Q, date(2026, 11, 3))]
    assert body["citations"] == ["1005.6(b)(1)"]                 # the guard removed 1026.99(z)
    assert "invalid_citations" in body["flags"]
    assert body["config_hash"] == "b2bef7bd2dd0"                 # the configuration the eval measured
    assert body["regulations_as_of"] == "2026-10-23"
    assert body["disclaimer"].startswith("General information")


def test_every_public_field_is_a_response_field():
    from app.schema import Response
    assert set(h.PUBLIC_FIELDS) < set(Response.model_fields)
    assert set(Response.model_fields) - set(h.PUBLIC_FIELDS) == {"raw_answer", "raw_citations", "invalid_citations"}


def test_today_defaults_to_the_utc_date(fake):
    calls, script = fake
    script.append(text(GOOD))
    before = datetime.now(timezone.utc).date()
    status, _, _ = call(event({"question": Q}))
    assert status == 200 and calls[0][1] in (before, datetime.now(timezone.utc).date())


def test_base64_body_is_read(fake):
    calls, script = fake
    script.append(text(GOOD))
    status, _, _ = call(event({"question": Q}, b64=True))
    assert status == 200 and calls[0][0] == Q


# ---------------------------------------------------------------- bad requests (no model call)

@pytest.mark.parametrize("body, status, words", [
    ("not json", 400, "JSON object"),
    ("[1, 2]", 400, "JSON object"),
    ("", 400, "JSON object"),
    ({"questions": Q}, 400, "unknown field(s): questions"),
    ({}, 400, "question must be"),
    ({"question": "   "}, 400, "question must be"),
    ({"question": 42}, 400, "question must be"),
    ({"question": "x" * 2001}, 400, "over 2000 characters"),
    ({"question": Q, "today": "3 Nov 2026"}, 400, "YYYY-MM-DD"),
    ({"question": Q, "today": 20261103}, 400, "YYYY-MM-DD"),
    ({"question": "x" * 9000}, 413, "over 8192 bytes"),
])
def test_bad_requests(fake, body, status, words):
    calls, _ = fake
    got, out, _ = call(event(body))
    assert (got, words in out["error"], out["request_id"]) == (status, True, "req-123")
    assert calls == []


def test_question_at_the_cap_is_accepted(fake):
    calls, script = fake
    script.append(text(GOOD))
    status, _, _ = call(event({"question": "x" * 2000}))
    assert status == 200 and len(calls) == 1


def test_bad_base64_is_a_400(fake):
    status, body, _ = call({"body": "%%%not-base64", "isBase64Encoded": True})
    assert status == 400 and "decoded" in body["error"]


# ---------------------------------------------------------------- Bedrock errors

@pytest.mark.parametrize("code, status", [("ThrottlingException", 503), ("ServiceUnavailableException", 503),
                                          ("AccessDeniedException", 502), ("ValidationException", 502)])
def test_bedrock_errors(fake, code, status):
    _, script = fake
    script.append(ClientError(code))
    got, body, _ = call(event({"question": Q}))
    assert got == status
    assert "secret detail" not in body["error"] and body["request_id"] == "req-123"


def test_unexpected_error_is_a_500(fake):
    _, script = fake
    script.append(RuntimeError("boom with details"))
    status, body, _ = call(event({"question": Q}))
    assert (status, body) == (500, {"error": "Internal error.", "request_id": "req-123"})


# ---------------------------------------------------------------- logs

def log_lines_from(out):
    return [json.loads(x) for x in out.splitlines() if x.startswith("{")]


def test_log_has_numbers_but_never_the_question_or_answer(fake, capsys):
    _, script = fake
    script.append(text(GOOD))
    question = "My customer Jane Roe at 12 Elm St lost her card; what is her liability under Reg E 6(b)?"
    call(event({"question": question}))
    out = capsys.readouterr().out
    assert "Jane" not in out and "Elm" not in out and "At most $50" not in out
    [line] = log_lines_from(out)
    assert line["status"] == 200 and line["question_chars"] == len(question)
    assert line["input_tokens"] == 1000 and line["model_calls"] == 1 and line["cold"] is False
    assert line["config_hash"] == "b2bef7bd2dd0" and "invalid_citations" in line["flags"]


def test_error_log_has_the_code_not_the_message(fake, capsys):
    _, script = fake
    script.append(ClientError("ThrottlingException"))
    call(event({"question": "Jane Roe's card"}))
    out = capsys.readouterr().out
    assert "secret detail" not in out and "Jane" not in out
    assert log_lines_from(out)[0]["error"] == "ThrottlingException"


def test_first_request_is_logged_cold_with_init_time(fake, capsys, monkeypatch):
    _, script = fake
    script += [text(GOOD), text(GOOD)]
    monkeypatch.setitem(h._state, "cold", True)
    monkeypatch.setitem(h._state, "init_ms", 1234)
    call(event({"question": Q}))
    call(event({"question": Q}))
    first, second = log_lines_from(capsys.readouterr().out)
    assert (first["cold"], first["init_ms"]) == (True, 1234)
    assert second["cold"] is False and "init_ms" not in second


# ---------------------------------------------------------------- init

def test_warm_loads_the_index_and_makes_both_clients():
    ms = h.warm()
    assert isinstance(ms, int) and ms >= 0
    assert rt.load.cache_info().currsize == 1
    assert rt._client.cache_info().currsize == 1 and an._client.cache_info().currsize == 1


def test_nothing_is_warmed_at_import_outside_lambda():
    import os
    assert "AWS_LAMBDA_FUNCTION_NAME" not in os.environ and h._state["init_ms"] is None
