"""AWS Lambda entry point behind API Gateway (spec step 8). One route: POST /ask.

Request body (JSON):
    {"question": "Customer's debit card was stolen Friday and she called today. Most she can lose?",
     "today": "2026-11-03"}                      # optional; default: today's date in UTC

Response: 200 with the answer (the Response fields in PUBLIC_FIELDS), or an error status with
{"error": "...", "request_id": "..."}:
    400  the body isn't a JSON object, has an unknown field, the question is missing, empty or over
         the length cap, or "today" isn't a YYYY-MM-DD date
    413  the body is over MAX_BODY_BYTES
    503  Bedrock is throttling (too many requests); try again shortly
    502  Bedrock returned another error
    500  anything else
API Gateway itself answers 403 when the API key is missing or wrong, 429 when the usage plan's
throttle or daily quota is hit, and 504 when the whole call takes over 29 seconds.

The configuration is app.config.DEFAULT, the one the eval measured (its config_hash is in every
response). The API leaves out the eval-only fields: the model's answer text and citation strings
from before the citation guard (raw_answer, raw_citations) and the IDs the guard removed
(invalid_citations). A caller sees only what passed the guard; the "invalid_citations" flag still
says when something was removed.

Logs: one JSON line per request with sizes, timings, token counts and flags. Never the question or
the answer text (spec: no personal data in logs).

Cold start: when this module is imported inside Lambda, warm() loads the index, builds the BM25
statistics and creates both Bedrock clients during the init phase, so the first request doesn't pay
for them. The first request after a cold start logs "cold": true with "init_ms".
"""
import base64
import json
import os
import time
from datetime import date, datetime, timezone

from app import retrieve as rt
from app.answer import _client, answer
from app.config import DEFAULT

MAX_BODY_BYTES = 8192               # the question cap is 2,000 characters; this leaves room for JSON escaping
ALLOWED_FIELDS = {"question", "today"}
PUBLIC_FIELDS = ("answer", "citations", "tool_calls", "abstained", "abstain_reason", "retrieved_ids",
                 "regulations_as_of", "disclaimer", "flags", "prompt_version", "config_hash", "model_id",
                 "usage", "model_calls", "latency_ms")
THROTTLED = {"ThrottlingException", "ServiceQuotaExceededException", "TooManyRequestsException",
             "ServiceUnavailableException", "ModelNotReadyException"}

_state = {"cold": True, "init_ms": None}


def warm():
    """Load the index and create the Bedrock clients now, not on the first request. Returns milliseconds."""
    started = time.perf_counter()
    rt.load()
    rt._client()                    # Titan, for the question embedding
    _client(DEFAULT.read_timeout)   # Sonnet, for the answer
    return round((time.perf_counter() - started) * 1000)


if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):      # only inside Lambda; tests call warm() themselves
    _state["init_ms"] = warm()


class BadRequest(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def read_request(event):
    """(question, today) from an API Gateway proxy event. Raises BadRequest."""
    body = event.get("body") or ""
    if event.get("isBase64Encoded"):
        try:
            body = base64.b64decode(body).decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            raise BadRequest(400, "the body could not be decoded") from None
    if len(body.encode("utf-8")) > MAX_BODY_BYTES:
        raise BadRequest(413, f"the body is over {MAX_BODY_BYTES} bytes")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise BadRequest(400, 'the body must be a JSON object such as {"question": "..."}') from None
    if not isinstance(data, dict):
        raise BadRequest(400, 'the body must be a JSON object such as {"question": "..."}')
    unknown = sorted(set(data) - ALLOWED_FIELDS)
    if unknown:
        raise BadRequest(400, f"unknown field(s): {', '.join(unknown)}; allowed: question, today")
    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise BadRequest(400, "question must be a non-empty string")
    if len(question.strip()) > DEFAULT.max_question_chars:
        raise BadRequest(400, f"question is over {DEFAULT.max_question_chars} characters")
    today = data.get("today")
    if today is None:
        return question, datetime.now(timezone.utc).date()
    if not isinstance(today, str):
        raise BadRequest(400, "today must be a date written YYYY-MM-DD")
    try:
        return question, date.fromisoformat(today)
    except ValueError:
        raise BadRequest(400, "today must be a date written YYYY-MM-DD") from None


def reply(status, body):
    return {"statusCode": status,
            "headers": {"Content-Type": "application/json", "Cache-Control": "no-store"},
            "body": json.dumps(body, ensure_ascii=False)}


def error_code(exc):
    """The AWS error code of a botocore ClientError, else None."""
    response = getattr(exc, "response", None)
    return response.get("Error", {}).get("Code") if isinstance(response, dict) else None


def log(**fields):
    print(json.dumps(fields, sort_keys=True))       # Lambda sends stdout to CloudWatch Logs


def lambda_handler(event, context):
    started = time.perf_counter()
    request_id = getattr(context, "aws_request_id", None)
    cold, _state["cold"] = _state["cold"], False
    entry = {"event": "ask", "request_id": request_id, "cold": cold}
    if cold:
        entry["init_ms"] = _state["init_ms"]
    try:
        question, today = read_request(event)
        entry["question_chars"] = len(question.strip())
        resp = answer(question, today)
    except BadRequest as e:
        log(**entry, status=e.status, error="bad_request", total_ms=_ms(started))
        return reply(e.status, {"error": str(e), "request_id": request_id})
    except Exception as e:                          # Bedrock and anything unexpected; text never logged
        code = error_code(e)
        if code in THROTTLED:
            status, message = 503, "The model service is busy. Try again in a few seconds."
        elif code:
            status, message = 502, "The model service returned an error."
        else:
            status, message = 500, "Internal error."
        log(**entry, status=status, error=code or type(e).__name__, total_ms=_ms(started))
        return reply(status, {"error": message, "request_id": request_id})

    body = resp.model_dump(mode="json", include=set(PUBLIC_FIELDS))
    log(**entry, status=200, total_ms=_ms(started), answer_ms=resp.latency_ms, model_calls=resp.model_calls,
        input_tokens=resp.usage.get("inputTokens", 0), output_tokens=resp.usage.get("outputTokens", 0),
        tool_calls=len(resp.tool_calls), abstained=resp.abstained, flags=resp.flags,
        config_hash=resp.config_hash)
    return reply(200, {k: body[k] for k in PUBLIC_FIELDS})


def _ms(started):
    return round((time.perf_counter() - started) * 1000)
