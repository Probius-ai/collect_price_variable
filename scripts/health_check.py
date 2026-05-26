#!/usr/bin/env python3
"""FastAPI health check with Discord alert — cron-friendly.

Runs every 5 minutes via cron (see `scripts/health_check_wrapper.sh`
+ `crontab -l`). Stdlib only — no project deps, no API keys.

Webhook URL is read from `HEALTH_CHECK_DISCORD_WEBHOOK_URL` (loaded
from `.env` by the wrapper). The repo-tracked copy of this script
NEVER hard-codes the URL — that would leak the credential into git
history. Empty / unset → Discord notification is silently skipped
(stdout output still happens, exit code unchanged).

Exit codes:
    0 — all endpoints healthy
    1 — at least one endpoint failed (so cron can flag the failure
        via its own MAILTO if configured)
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request


# Endpoints to probe. Add more as services come online.
TARGETS = [
    "http://localhost:8000/health",
]
TIMEOUT = 5  # seconds — per request
# When set, also verify the response JSON has `{"status": EXPECT_STATUS}`.
# `None` → just check HTTP 200.
EXPECT_STATUS = "ok"


# Repo's `.env` provides this via `scripts/health_check_wrapper.sh`.
# Empty / unset → no Discord push (stdout + exit code still work).
DISCORD_WEBHOOK_URL = os.environ.get("HEALTH_CHECK_DISCORD_WEBHOOK_URL", "").strip()


def check(url: str) -> dict:
    """Probe one endpoint. Returns a status dict — never raises."""
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as resp:
            latency = round((time.perf_counter() - start) * 1000, 1)
            body = resp.read().decode("utf-8", "ignore")
            ok = resp.status == 200
            if ok and EXPECT_STATUS:
                # Tolerate non-JSON bodies — we only fail when the body
                # IS valid JSON but the status field doesn't match.
                try:
                    if json.loads(body).get("status") != EXPECT_STATUS:
                        ok = False
                except json.JSONDecodeError:
                    pass
            return {"url": url, "ok": ok, "status": resp.status, "latency_ms": latency}
    except urllib.error.HTTPError as e:
        return {"url": url, "ok": False, "status": e.code, "error": "HTTPError"}
    except Exception as e:
        return {"url": url, "ok": False, "status": None, "error": str(e)}


def notify_discord(results: list) -> None:
    """Post a summary message to Discord. Failures are logged to stderr
    but never raised — a webhook outage shouldn't crash the cron job
    (cron would email a non-zero exit, which we reserve for ENDPOINT
    failures, not notifier failures)."""
    if not DISCORD_WEBHOOK_URL:
        return
    failures = [r for r in results if not r["ok"]]

    if failures:
        lines = [
            f":rotating_light: **Health Check 경고** "
            f"({len(failures)}/{len(results)} 실패)"
        ]
        for r in results:
            if r["ok"]:
                lines.append(f"✅ `{r['url']}` ({r['latency_ms']}ms)")
            else:
                detail = r.get("error") or f"HTTP {r['status']}"
                lines.append(f"❌ `{r['url']}` — {detail}")
    else:
        lines = [
            f":white_check_mark: **Health Check 정상** "
            f"({len(results)}/{len(results)} 통과)"
        ]
        for r in results:
            lines.append(f"✅ `{r['url']}` ({r['latency_ms']}ms)")

    payload = json.dumps({"content": "\n".join(lines)}).encode("utf-8")
    req = urllib.request.Request(
        DISCORD_WEBHOOK_URL,
        data=payload,
        # Discord rejects requests with the default `Python-urllib/X.Y`
        # User-Agent — it's on their suspicious-client list. Use a
        # generic UA that mirrors what `curl` sends.
        headers={
            "Content-Type": "application/json",
            "User-Agent": "kpx-health-check/1.0 (+cron)",
        },
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=TIMEOUT)
    except Exception as e:
        # Don't echo the webhook URL — it lives in `e` for some
        # urllib errors. Strip it before printing.
        msg = str(e).replace(DISCORD_WEBHOOK_URL, "<REDACTED>")
        print(f"⚠️  Discord 알림 실패: {msg}", file=sys.stderr)


def main() -> None:
    results = [check(url) for url in TARGETS]
    for r in results:
        if r["ok"]:
            print(f"✅ {r['url']} ({r['latency_ms']}ms)")
        else:
            detail = r.get("error") or f"HTTP {r['status']}"
            print(f"❌ {r['url']} — {detail}")

    notify_discord(results)
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
