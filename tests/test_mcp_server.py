"""Tests for the MCP server (mcp_server/server.py). No AWS call: a small fake API runs on localhost, and
the server is reached the two ways an MCP app reaches it (in-process, and as a subprocess over stdio)."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anyio
import pytest
from mcp import Client
from mcp.client.stdio import StdioServerParameters

from mcp_server import server as ms
from mcp_server import try_it

ANSWER = {"answer": "At most $50 if reported within 2 business days [§ 1005.6(b)(1)].",
          "citations": ["1005.6(b)(1)"], "abstained": False, "abstain_reason": None,
          "tool_calls": [{"name": "dispute_timeline", "args": {"regulation": "E", "event": "unauthorized"},
                          "result": {}}],
          "regulations_as_of": "2026-10-23", "disclaimer": "General information, not legal advice.",
          "config_hash": "b2bef7bd2dd0"}
KEY = "TEST-KEY-1234567890"


class FakeApi(BaseHTTPRequestHandler):
    """Answers like the deployed API: needs the key, then replies with whatever `script` says."""
    script = []          # list of (status, body)
    seen = []            # request bodies and headers

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeApi.seen.append({"body": body, "key": self.headers.get("x-api-key"),
                             "agent": self.headers.get("User-Agent")})
        status, reply = (403, {"message": "Forbidden"}) if self.headers.get("x-api-key") != KEY \
            else FakeApi.script.pop(0)
        data = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


@pytest.fixture
def api(monkeypatch):
    httpd = HTTPServer(("127.0.0.1", 0), FakeApi)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{httpd.server_port}/v1/ask"
    FakeApi.script, FakeApi.seen = [], []
    monkeypatch.setenv(ms.URL_VAR, url)
    monkeypatch.setenv(ms.KEY_VAR, KEY)
    yield url
    httpd.shutdown()


async def call(arguments):
    async with Client(ms.server) as client:
        return await client.call_tool("ask_card_rules", arguments)


def text_of(result):
    return "\n".join(b.text for b in result.content)


# ---------------------------------------------------------------- the tool as an app sees it

def test_one_tool_with_a_required_question_and_optional_today():
    async def listing():
        async with Client(ms.server) as client:
            return (await client.list_tools()).tools
    [tool] = anyio.run(listing)
    assert tool.name == "ask_card_rules"
    schema = tool.input_schema
    assert schema["required"] == ["question"]
    assert set(schema["properties"]) == {"question", "today"}
    assert "YYYY-MM-DD" in json.dumps(schema["properties"]["today"])
    assert "Regulation Z" in tool.description and "Regulation E" in tool.description
    assert tool.annotations.read_only_hint is True


def test_answer_text_has_answer_citations_date_and_config(api):
    FakeApi.script.append((200, ANSWER))
    result = anyio.run(call, {"question": "Card stolen; most she can lose?", "today": "2026-11-03"})
    out = text_of(result)
    assert not result.is_error
    assert ANSWER["answer"] in out and "Citations (12 CFR): 1005.6(b)(1)" in out
    assert "Deadline calculation used:" in out and "as of 2026-10-23" in out and "b2bef7bd2dd0" in out
    assert FakeApi.seen[0]["body"] == {"question": "Card stolen; most she can lose?", "today": "2026-11-03"}
    assert FakeApi.seen[0]["key"] == KEY and FakeApi.seen[0]["agent"] == "regze-assistant-mcp"


def test_today_is_left_out_when_not_given(api):
    FakeApi.script.append((200, ANSWER))
    anyio.run(call, {"question": "q"})
    assert FakeApi.seen[0]["body"] == {"question": "q"}


def test_a_decline_says_so(api):
    FakeApi.script.append((200, {**ANSWER, "abstained": True, "abstain_reason": "card network rules",
                                 "citations": [], "tool_calls": [], "answer": "Outside Regulations Z and E."}))
    out = text_of(anyio.run(call, {"question": "chargeback reason code?"}))
    assert out.startswith("DECLINED: the regulations don't answer this (card network rules).")
    assert "Citations" not in out


# ---------------------------------------------------------------- failures reach the model as tool errors

@pytest.mark.parametrize("status, body, words", [
    (400, {"error": "question is over 2000 characters"}, "rejected: question is over 2000 characters"),
    (429, {"message": "Limit Exceeded"}, "daily quota"),
    (503, {"error": "busy"}, "busy"),
    (504, {"message": "Endpoint request timed out"}, "29 seconds"),
    (500, {"error": "Internal error."}, "HTTP 500"),
])
def test_api_errors(api, status, body, words):
    FakeApi.script.append((status, body))
    result = anyio.run(call, {"question": "q"})
    assert result.is_error and words in text_of(result)


def test_wrong_key_is_refused_and_never_shown(api, monkeypatch):
    monkeypatch.setenv(ms.KEY_VAR, "WRONG-KEY-VALUE")
    result = anyio.run(call, {"question": "q"})
    assert result.is_error and "API key was refused" in text_of(result)
    assert "WRONG-KEY-VALUE" not in text_of(result)


def test_missing_settings(monkeypatch):
    monkeypatch.delenv(ms.URL_VAR, raising=False)
    monkeypatch.setenv(ms.KEY_VAR, KEY)
    result = anyio.run(call, {"question": "q"})
    assert result.is_error and "Set REGZE_API_URL and REGZE_API_KEY" in text_of(result)


def test_unreachable_api(monkeypatch):
    monkeypatch.setenv(ms.URL_VAR, "http://127.0.0.1:9/v1/ask")      # nothing listens on port 9
    monkeypatch.setenv(ms.KEY_VAR, KEY)
    result = anyio.run(call, {"question": "q"})
    assert result.is_error and "Could not reach" in text_of(result)


# ---------------------------------------------------------------- end to end, over stdio

def test_stdio_subprocess_end_to_end(api, capsys):
    """Starts server.py as a subprocess, as an MCP app would, through try_it.run."""
    FakeApi.script.append((200, ANSWER))
    result = anyio.run(try_it.run, "Card stolen; most she can lose?", "2026-11-03", sys.executable)
    out = capsys.readouterr().out
    assert not result.is_error
    assert "Tools offered: ask_card_rules" in out and "Citations (12 CFR): 1005.6(b)(1)" in out
    assert KEY not in out
    assert FakeApi.seen[0]["body"]["today"] == "2026-11-03"
