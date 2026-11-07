"""Smoke test of the deployed API (spec step 8): a few real calls, each checked, with round-trip times.

From the repo root, after `sam deploy`, with the two environment variables set (the key is never printed):
    export REGZE_API_URL=...      # the stack output AskUrl
    export REGZE_API_KEY=...      # the API key's value
    python -m deploy.smoke

Calls, in this order:
  1. no API key                    -> 403 from API Gateway (the function is never reached)
  2. a date question (tool)        -> 200, dispute_timeline called, citations
     The first request after a deploy starts a new function instance, so this one is the cold start
     (spec t4-time: the slowest case, a tool question, against the 29-second limit).
  3. a plain rule question         -> 200, with citations
  4. an out-of-scope question      -> 200 (it should be declined: a "note" if not)
  5. a misspelt field              -> 400 from the function
The questions are made up for this check; none comes from the golden set. Each answer costs a few US
cents of Bedrock usage (3 model questions) and counts against the key's daily quota (4 requests).

A check tests the deployment (status codes, the tool running inside Lambda, citations passing the
guard). Whether a question was declined is model behaviour, which the eval measures, so a surprise there
is printed as a note, not a failure. Prints one line per call and a summary; exits 1 if any check fails.
Lambda-side timings (cold or warm, init time) are in the function's log: `aws logs tail LOG_GROUP --since 15m`.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

TIMEOUT = 35            # seconds; API Gateway gives up at 29

CALLS = [
    {"name": "no key", "key": False, "body": {"question": "What does Regulation E cover?"},
     "expect": {"status": 403}},
    {"name": "tool question (cold)", "key": True,
     "body": {"question": "A customer told us on Thursday, 10 July 2025 that her debit card had been missing "
                          "since Monday, 7 July 2025, when she noticed it was gone. Someone used it for $900 of "
                          "purchases in between. What is the most she can be held liable for, and by what date "
                          "should she have told us to keep her liability at the lowest tier?",
              "today": "2025-07-10"},
     "expect": {"status": 200, "tool": True, "citations": True}, "soft": {"abstained": False}},
    {"name": "rule question", "key": True,
     "body": {"question": "Under Regulation E, what must a bank tell a consumer about electronic fund "
                          "transfers when the account is first set up, and by when?", "today": "2026-11-05"},
     "expect": {"status": 200, "citations": True}, "soft": {"abstained": False}},
    {"name": "out of scope", "key": True,
     "body": {"question": "What chargeback reason code should we use for a duplicate charge under the card "
                          "network's rules?", "today": "2026-11-05"},
     "expect": {"status": 200}, "soft": {"abstained": True}},
    {"name": "bad field", "key": True, "body": {"questions": "x"}, "expect": {"status": 400}},
]


def post(url, body, key=None, timeout=TIMEOUT):
    """(status, parsed JSON body or {}, seconds)."""
    headers = {"Content-Type": "application/json"}
    if key:
        headers["x-api-key"] = key
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    seconds = time.perf_counter() - started
    try:
        data = json.loads(raw or b"{}")
    except json.JSONDecodeError:
        data = {"_text": raw[:200].decode("utf-8", "replace")}
    return status, data if isinstance(data, dict) else {"_value": data}, seconds


def check(expect, status, data):
    """List of failed expectations (empty: pass). These test the deployment: the status, the tool path
    running inside Lambda (it reads data/us_federal_holidays.json), citations passing the guard."""
    fails = []
    if status != expect["status"]:
        fails.append(f"status {status}, expected {expect['status']}")
        return fails
    if expect.get("tool") and not any(c.get("name") == "dispute_timeline" for c in data.get("tool_calls", [])):
        fails.append("no dispute_timeline call")
    if expect.get("citations") and not data.get("citations"):
        fails.append("no citations")
    return fails


def notes(soft, status, data):
    """Model behaviour worth a look but not a deployment failure (the eval measures it): whether the
    answer was declined as expected."""
    if status != 200 or "abstained" not in soft or data.get("abstained") is soft["abstained"]:
        return []
    return [f"note: abstained {data.get('abstained')}, expected {soft['abstained']}"]


def summary_line(call, status, data, seconds, fails, note=()):
    parts = [f"{'PASS' if not fails else 'FAIL'}  {call['name']:22} {status}  {seconds:5.1f} s"]
    if status == 200:
        parts.append(f"abstained={data.get('abstained')} citations={len(data.get('citations', []))} "
                     f"tool_calls={len(data.get('tool_calls', []))} model_calls={data.get('model_calls')} "
                     f"in_lambda={data.get('latency_ms', 0) / 1000:.1f} s config={data.get('config_hash')}")
    elif data.get("error") or data.get("message"):
        parts.append(f"error={data.get('error') or data.get('message')!r}")
    if fails or note:
        parts.append("; ".join([*fails, *note]))
    return "  ".join(parts)


def main(argv=None, env=os.environ, poster=post):
    ap = argparse.ArgumentParser(description="Call the deployed API a few times and check the replies.")
    ap.add_argument("--show", action="store_true", help="also print each answer's text and citations")
    args = ap.parse_args(argv)
    url, key = env.get("REGZE_API_URL"), env.get("REGZE_API_KEY")
    if not url or not key:
        sys.exit("Set REGZE_API_URL and REGZE_API_KEY first (see the docstring).")
    failed = 0
    for call in CALLS:
        status, data, seconds = poster(url, call["body"], key if call["key"] else None)
        fails = check(call["expect"], status, data)
        failed += bool(fails)
        print(summary_line(call, status, data, seconds, fails, notes(call.get("soft", {}), status, data)))
        if args.show and status == 200:
            print("      " + data.get("answer", "").replace("\n", " "))
            print("      citations: " + ", ".join(data.get("citations", [])))
    print(f"\n{len(CALLS) - failed} of {len(CALLS)} checks passed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
