"""Try the MCP server without an AI app: start it the way an MCP app would (a subprocess over stdio),
list its tools, ask one question through ask_card_rules, and print what the model would receive.

From the repo root, with REGZE_API_URL and REGZE_API_KEY set (as for deploy.smoke):
    python -m mcp_server.try_it "A customer's debit card was stolen on 3 March 2026 ..." [--today 2026-03-05]
One question costs a few US cents and counts against the API key's daily quota.
"""
import argparse
import os
import sys
from pathlib import Path

import anyio
from mcp import Client
from mcp.client.stdio import StdioServerParameters

SERVER = Path(__file__).resolve().parent / "server.py"


async def run(question, today=None, command=sys.executable, server=SERVER):
    # The SDK starts the server with only a few safe variables (PATH, HOME and the like), not this
    # whole environment, so the URL and key are passed on by name, as an MCP app's config does.
    env = {k: os.environ[k] for k in ("REGZE_API_URL", "REGZE_API_KEY") if k in os.environ}
    params = StdioServerParameters(command=command, args=[str(server)], env=env)
    async with Client(params) as client:
        tools = (await client.list_tools()).tools
        print("Tools offered:", ", ".join(t.name for t in tools))
        args = {"question": question} if today is None else {"question": question, "today": today}
        result = await client.call_tool("ask_card_rules", args)
        print("Error:" if result.is_error else "Result:")
        for block in result.content:
            print(getattr(block, "text", block))
        return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Ask one question through the MCP server.")
    ap.add_argument("question", nargs="+")
    ap.add_argument("--today", help="YYYY-MM-DD (default: the API uses today's UTC date)")
    args = ap.parse_args(argv)
    result = anyio.run(run, " ".join(args.question), args.today)
    return 1 if result.is_error else 0


if __name__ == "__main__":
    sys.exit(main())
