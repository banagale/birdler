#!/usr/bin/env python3
"""
Web research script (single pass, no curl) to:
- Run web searches (DuckDuckGo lite + Bing HTML) for Codex/Claude CLI logs/history/realtime exfil topics.
- Fetch top results, verify HTTP 200, and check for expected keywords.
- Emit a markdown report with sources and brief notes at repo root.

Usage:
  poetry run python tools/web_research_codex.py --out web-research-codex-logs.md --days 60
"""
from __future__ import annotations

import argparse
import html
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import requests


DDG_LITE = "https://duckduckgo.com/lite/"
BING = "https://www.bing.com/search"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15"
)


QUERIES = [
    "codex cli conversation logs",
    "codex cli history path",
    "codex cli ndjson events",
    "codex cli sink logging",
    "claude code cli logs",
    "claude cli conversation history file",
    "claude code ndjson events logs",
]


KEYWORDS = re.compile(r"(?i)log|history|json|ndjson|event|stdout|sink|append|session|conversation")


@dataclass
class Hit:
    engine: str
    query: str
    title: str
    url: str
    status: int | None = None
    matched: bool = False
    notes: str = ""


def ddg_search(session: requests.Session, query: str, max_hits: int = 8) -> List[Hit]:
    params = {"q": query}
    r = session.get(DDG_LITE, params=params, timeout=20)
    r.raise_for_status()
    html_text = r.text
    hits: List[Hit] = []
    # DuckDuckGo lite lists results as <a href="...">Title</a> inside <td class="result-link">
    for m in re.finditer(r"<a\s+href=\"(https?://[^\"]+)\"[^>]*>(.*?)</a>", html_text, flags=re.I):
        url = html.unescape(m.group(1))
        title = re.sub("<.*?>", "", html.unescape(m.group(2)))
        hits.append(Hit(engine="ddg", query=query, title=title.strip(), url=url))
        if len(hits) >= max_hits:
            break
    return hits


def bing_search(session: requests.Session, query: str, max_hits: int = 8) -> List[Hit]:
    params = {"q": query}
    r = session.get(BING, params=params, timeout=20)
    r.raise_for_status()
    html_text = r.text
    hits: List[Hit] = []
    # Rough extract: result links often have <li class="b_algo">...<a href="...">Title</a>
    for m in re.finditer(r"<li\s+class=\"b_algo\"[\s\S]*?<a\s+href=\"(https?://[^\"]+)\"[^>]*>(.*?)</a>", html_text, flags=re.I):
        url = html.unescape(m.group(1))
        title = re.sub("<.*?>", "", html.unescape(m.group(2)))
        hits.append(Hit(engine="bing", query=query, title=title.strip(), url=url))
        if len(hits) >= max_hits:
            break
    return hits


def fetch_and_check(session: requests.Session, hit: Hit, timeout: int = 25) -> Hit:
    try:
        r = session.get(hit.url, timeout=timeout, allow_redirects=True)
        hit.status = r.status_code
        text = r.text if r.status_code == 200 else ""
        hit.matched = bool(KEYWORDS.search(text))
        # Short note: where the keyword matched
        if hit.matched:
            snip = KEYWORDS.search(text)
            if snip:
                lo = max(0, snip.start() - 40)
                hi = min(len(text), snip.end() + 40)
                hit.notes = re.sub(r"\s+", " ", text[lo:hi])
    except Exception as e:
        hit.status = None
        hit.matched = False
        hit.notes = f"error: {e}"
    return hit


def unique_urls(hits: Iterable[Hit]) -> List[Hit]:
    seen = set()
    out: List[Hit] = []
    for h in hits:
        if h.url in seen:
            continue
        seen.add(h.url)
        out.append(h)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("web-research-codex-logs.md"))
    ap.add_argument("--days", type=int, default=60, help="Intended recency window (best-effort)")
    ap.add_argument("--max-per-query", type=int, default=6)
    args = ap.parse_args()

    sess = requests.Session()
    sess.headers.update({"User-Agent": UA})

    all_hits: List[Hit] = []
    for q in QUERIES:
        try:
            all_hits.extend(ddg_search(sess, q, max_hits=args.max_per_query))
        except Exception:
            pass
        try:
            all_hits.extend(bing_search(sess, q, max_hits=2))  # fewer from Bing to limit volume
        except Exception:
            pass

    hits = unique_urls(all_hits)

    # Also search GitHub API for repo/docs likely to contain logging/history details
    try:
        gh_repos = [
            "Daivison06/open_codex_cli",
            # Add more known/community repos of interest here if needed
        ]
        api = "https://api.github.com/repos/{repo}/contents/{path}"
        for repo in gh_repos:
            for path in ("README.md", "README.MD", "readme.md"):
                url = api.format(repo=repo, path=path)
                r = sess.get(url, timeout=20, headers={"Accept": "application/vnd.github.raw"})
                if r.status_code == 200 and isinstance(r.text, str):
                    text = r.text
                    title = f"GitHub README: {repo}/{path}"
                    fake_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{path}"
                    matched = bool(KEYWORDS.search(text))
                    all_hits.append(Hit(engine="github", query="repo-readme", title=title, url=fake_url, status=200, matched=matched, notes="readme"))
    except Exception:
        pass

    # Fetch and check content
    checked: List[Hit] = []
    for h in hits:
        checked.append(fetch_and_check(sess, h))
        time.sleep(0.2)
    # Append pre-checked GitHub README items
    for h in all_hits:
        if h.engine == "github":
            checked.append(h)

    # Write report
    now = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
    lines: List[str] = []
    lines.append("# Codex/Claude CLI Realtime Conversation Research (web)")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("Notes: Each source was fetched and checked for HTTP 200 and presence of relevant keywords (log/history/json/ndjson/events/stdout/sink/session/conversation). This is a best‑effort pass.")
    lines.append("")

    for h in checked:
        status = h.status if h.status is not None else "error"
        match = "yes" if h.matched else "no"
        lines.append(f"- Source: {h.engine}; Query: “{h.query}”")
        lines.append(f"  - URL: {h.url}")
        lines.append(f"  - Title: {h.title}")
        lines.append(f"  - HTTP: {status}; Relevant keywords present: {match}")
        if h.notes:
            sn = h.notes
            if len(sn) > 200:
                sn = sn[:200] + "…"
            lines.append(f"  - Snippet: {sn}")
        lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out} with {len(checked)} sources.")


if __name__ == "__main__":
    main()
