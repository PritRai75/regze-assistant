"""MCP server with one tool, ask_card_rules, that asks the deployed RegZE Assistant API (spec step 8).

MCP (Model Context Protocol) is a standard way for an AI app (Claude Desktop, Claude Code, an IDE) to
call tools that run on your own machine. The app starts this file as a small background program and
talks to it over stdin/stdout ("stdio"). When the app's model decides a question is about card or
payment rules, it calls ask_card_rules; this program sends the question to the API over HTTPS and
hands back the answer, its citations and the "as of" date.

The program holds no model and no index: the deployed API does all the work, with the same
configuration the eval measured. The API key is read from an environment variable on your machine and
never written to the repo or printed (spec).

Run it (an MCP app normally does this for you):
    export REGZE_API_URL=https://....execute-api.ap-south-1.amazonaws.com/v1/ask
    export REGZE_API_KEY=...
    python -m mcp_server.server           # from the repo root; waits for an MCP client on stdin
Try it without an AI app:  python -m mcp_server.try_it "your question"
"""
import json
import os
import urllib.error
import urllib.request
from typing import Annotated

import anyio
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations
from pydantic import Field

TIMEOUT = 35            # seconds; API Gateway gives up at 29
URL_VAR, KEY_VAR = "REGZE_API_URL", "REGZE_API_KEY"

server = MCPServer(
    name="regze-assistant",
    instructions=("Answers questions about US Regulation Z (credit cards, truth in lending) and Regulation E "
                  "(debit cards and electronic fund transfers) from the official text, with paragraph citations "
                  "and a deterministic deadline calculator. General information, not legal advice."),
)

TOOL_DESCRIPTION = """Answer a question about US card and payment rules from the text of Regulation Z \
(12 CFR 1026: credit cards and other open-end credit) and Regulation E (12 CFR 1005: debit cards, ATM and \
other electronic fund transfers), as of the date the answer states.

Use it for: liability for unauthorized card use or transfers, error and billing dispute deadlines and \
investigations, fee and rate notices, statement and disclosure rules. For questions with dates (when was the \
loss noticed, when was the bank told, when was the statement sent), include the dates; the service works \
out deadlines and liability tiers with a calculator rather than by estimate.

It declines questions the two regulations don't answer (card network rules, state law, an issuer's own \
account terms, other laws, rules that a court or Congress has overturned) and says so; a decline is a \
correct answer, not a failure. Every claim cites the paragraph it rests on, e.g. "1005.6(b)(1)" (a rule \
paragraph) or "1026.13(c)(2)-1" (an official interpretation). Quote those citations to the user."""


def call_api(question, today=None, *, url, key, timeout=TIMEOUT):
    """POST the question; returns (HTTP status, parsed JSON body or {})."""
    body = {"question": question} if today is None else {"question": question, "today": today}
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "x-api-key": key,
                                          "User-Agent": "regze-assistant-mcp"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except (urllib.error.URLError, TimeoutError) as e:
        raise ToolError(f"Could not reach the RegZE Assistant API ({getattr(e, 'reason', e)}).") from None
    try:
        data = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        data = {}
    return status, data if isinstance(data, dict) else {}


# What each failure means for the person asking, without exposing the key or the API's internals.
ERRORS = {
    400: "The question was rejected: {detail}",
    403: "The API key was refused. Check REGZE_API_KEY.",
    429: "The daily quota or rate limit for this API key was reached. Try again later.",
    502: "The model service returned an error. Try again later.",
    503: "The model service is busy. Try again in a few seconds.",
    504: "The answer took longer than the API allows (29 seconds). Try again, or shorten the question.",
}


def format_answer(data):
    """The API's JSON as text for the model to read."""
    lines = []
    if data.get("abstained"):
        lines.append("DECLINED: the regulations don't answer this" +
                     (f" ({data['abstain_reason']})." if data.get("abstain_reason") else "."))
    lines.append(data.get("answer", "").strip())
    if data.get("citations"):
        lines.append("Citations (12 CFR): " + ", ".join(data["citations"]))
    for call in data.get("tool_calls", []):
        lines.append("Deadline calculation used: " + json.dumps(call.get("args", {}), sort_keys=True))
    lines.append(f"Regulations as of {data.get('regulations_as_of', '?')}. {data.get('disclaimer', '')}".strip())
    lines.append(f"(RegZE Assistant, configuration {data.get('config_hash', '?')})")
    return "\n".join(line for line in lines if line)


def ask(question, today=None, env=os.environ, poster=call_api):
    """The tool's work, without MCP: returns the answer text or raises ToolError."""
    url, key = env.get(URL_VAR), env.get(KEY_VAR)
    if not url or not key:
        raise ToolError(f"Set {URL_VAR} and {KEY_VAR} in the environment that starts this MCP server.")
    status, data = poster(question, today, url=url, key=key)
    if status == 200:
        return format_answer(data)
    message = ERRORS.get(status, f"The API returned HTTP {status}.")
    raise ToolError(message.format(detail=data.get("error", "bad request")))


@server.tool(name="ask_card_rules", title="Ask the card rules assistant", description=TOOL_DESCRIPTION,
             annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True, idempotentHint=False))
async def ask_card_rules(
        question: Annotated[str, Field(description="The question in plain words, with any dates that matter "
                                                   "(up to 2,000 characters).")],
        today: Annotated[str | None, Field(description="Optional: the date the question is asked, YYYY-MM-DD. "
                                                       "Deadlines count from the dates in the question; default "
                                                       "today's date (UTC).")] = None) -> str:
    return await anyio.to_thread.run_sync(ask, question, today)


def main():
    server.run()        # stdio


if __name__ == "__main__":
    main()
