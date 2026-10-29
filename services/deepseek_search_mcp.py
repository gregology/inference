"""MCP server that gives Home Assistant's LLM agents DeepSeek web search.

Home Assistant's MCP integration speaks SSE, so this serves the SSE transport.
DeepSeek bills per successful request, so only the --allow'ed client
addresses (Home Assistant, loopback) may connect.

Also serves GetDateTime, which keeps Home Assistant's prompt cacheable.

Reads DEEPSEEK_API_KEY from the environment.
"""

import argparse
import ipaddress
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime

import anyio
import uvicorn
from mcp.server.mcpserver import MCPServer

API = "https://api.deepseek.com/v1/tools"

# Answers are read aloud by a small model on the O6, which processes prompts
# at ~30 tok/s: every result costs seconds before it can start speaking.
MAX_RESULTS = 5
SUMMARY_CHARS = 400

server = MCPServer(
    "deepseek-search",
    instructions="Web search and page fetch for questions about current events or facts you are unsure of.",
)


def _post(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{API}/{path}",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        # Hand the model a readable failure instead of a stack trace.
        return {"error": f"DeepSeek {path} returned HTTP {e.code}"}


def _site_name(name: str) -> str:
    # Some results carry the site name hex-encoded (6e686c652e636f6d = nhle.com).
    if len(name) % 2 == 0 and re.fullmatch(r"[0-9a-f]{6,}", name):
        try:
            decoded = bytes.fromhex(name).decode("ascii")
            if decoded.isprintable():
                return decoded
        except (ValueError, UnicodeDecodeError):
            pass
    return name


def format_results(data: dict) -> str:
    if "error" in data:
        return data["error"]
    results = data.get("results") or []
    if not results:
        return "No results."
    lines = []
    for i, r in enumerate(results, 1):
        date = (r.get("date") or "")[:10]
        meta = ", ".join(x for x in (date, _site_name(r.get("site_name") or "")) if x)
        text = (r.get("summary") or r.get("snippet") or "").strip().replace("\n", " ")
        if len(text) > SUMMARY_CHARS:
            text = text[:SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"
        lines.append(f"{i}. {r.get('title', '').strip()}" + (f" ({meta})" if meta else ""))
        lines.append(f"   {text}")
        lines.append(f"   {r.get('url', '')}")
    return "\n".join(lines)


@server.tool()
async def web_search(query: str, count: int = 3) -> str:
    """Search the web. Returns the top results with their date, a short summary and URL.

    Use for anything current, recent or that you are not sure about.
    query: what to search for, up to 128 characters. Supports site:, after:YYYY-MM-DD and "quoted phrases".
    count: number of results, 1 to 5.
    """
    body = {"query": query[:128], "count": max(1, min(count, MAX_RESULTS))}
    return format_results(await anyio.to_thread.run_sync(_post, "search", body))


@server.tool()
async def fetch_page(url: str, max_chars: int = 2000) -> str:
    """Fetch a web page as Markdown, truncated to max_chars (up to 6000).

    Use only when the search summaries don't answer the question.
    """
    data = await anyio.to_thread.run_sync(_post, "fetch", {"url": url})
    if "error" in data:
        return data["error"]
    content = (data.get("content") or "").strip()
    limit = max(200, min(max_chars, 6000))
    if len(content) > limit:
        content = content[:limit] + "\n[truncated]"
    return f"# {data.get('title', '').strip()}\n\n{content}"


# Home Assistant puts the time, to the second, at the end of the system prompt
# unless an agent has a tool named *GetDateTime. Qwen3.5's recurrent layers
# can't resume a cached prompt from that point, so the whole system prompt
# was re-processed every question (~550 tokens, ~17 s).
@server.tool(name="GetDateTime")
async def get_date_time() -> str:
    """Get the current date, time and day of the week."""
    return datetime.now().astimezone().strftime("%A %Y-%m-%d %H:%M %Z")


def allow_only(app, networks):
    """Reject HTTP clients outside the allowed networks with a 403."""
    async def wrapped(scope, receive, send):
        if scope["type"] == "http":
            host = (scope.get("client") or ("",))[0]
            try:
                allowed = any(ipaddress.ip_address(host) in net for net in networks)
            except ValueError:
                allowed = False
            if not allowed:
                await send({"type": "http.response.start", "status": 403,
                            "headers": [(b"content-type", b"text/plain")]})
                await send({"type": "http.response.body", "body": b"forbidden\n"})
                return
        await app(scope, receive, send)
    return wrapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10500)
    parser.add_argument("--allow", action="append", default=[],
                        help="client address or network allowed to connect (repeatable; default: loopback only)")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY is not set", file=sys.stderr)
        return 1
    networks = [ipaddress.ip_network(a, strict=False) for a in (args.allow or ["127.0.0.1", "::1"])]
    app = allow_only(server.sse_app(host=args.host), networks)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
