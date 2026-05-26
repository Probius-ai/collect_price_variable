"""Discord webhook notifier — used by the MLOps smoke test for run-status pings.

Designed so the integration is **silent and non-load-bearing**:
  * Webhook URL unset / empty → `send_discord_message()` no-ops cleanly.
  * Network error / Discord 5xx → logged at WARNING, never raised. We
    don't want a webhook outage to crash a 30-minute training run.
  * URL itself is treated as a credential and never echoed in logs or
    error messages (we route it through `src.utils.redact`).
"""

from __future__ import annotations

import json
from typing import Any

import requests

from src.config.settings import get_settings
from src.utils.logging import get_logger
from src.utils.redact import redact_secrets_in_text, secret_variants


log = get_logger("discord")

# Discord's per-message hard limits — content >2000 chars or embed
# description >4096 chars trigger 400. Truncate proactively rather than
# letting Discord reject the call.
_MAX_CONTENT_CHARS = 1900
_MAX_EMBED_DESC_CHARS = 4000


def _redact(text: str) -> str:
    """Strip the webhook URL (and its url-encoded variants) from a string.

    Belt-and-braces in case the URL leaks via an exception message or a
    `requests` traceback.
    """
    url = (get_settings().mlops_discord_webhook_url or "").strip()
    if not url:
        return text
    return redact_secrets_in_text(text, secret_variants(url))


def send_discord_message(
    content: str | None = None,
    *,
    embeds: list[dict[str, Any]] | None = None,
    timeout: float = 5.0,
) -> bool:
    """POST a message to the configured Discord webhook.

    Returns True if Discord accepted the message (2xx), False on any
    error (no webhook configured, network failure, 4xx/5xx). NEVER
    raises — callers can wrap a smoke test in it without worrying about
    a webhook outage taking down the run.

    Either `content` (≤1900 chars) or `embeds` (≤10 entries) must be
    supplied. Both works too.
    """
    settings = get_settings()
    webhook_url = (settings.mlops_discord_webhook_url or "").strip()
    if not webhook_url:
        return False

    if not content and not embeds:
        log.warning("send_discord_message called with no content or embeds")
        return False

    payload: dict[str, Any] = {}
    if content:
        if len(content) > _MAX_CONTENT_CHARS:
            content = content[:_MAX_CONTENT_CHARS - 3] + "..."
        payload["content"] = content
    if embeds:
        # Truncate every embed description that's over the limit
        clean: list[dict[str, Any]] = []
        for e in embeds[:10]:
            e = dict(e)
            desc = e.get("description")
            if isinstance(desc, str) and len(desc) > _MAX_EMBED_DESC_CHARS:
                e["description"] = desc[:_MAX_EMBED_DESC_CHARS - 3] + "..."
            clean.append(e)
        payload["embeds"] = clean

    try:
        r = requests.post(
            webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.RequestException as exc:
        log.warning("Discord webhook POST failed: %s", _redact(str(exc)))
        return False

    if not r.ok:
        log.warning(
            "Discord webhook returned %s: %s",
            r.status_code, _redact(r.text[:200]),
        )
        return False
    return True


def discord_enabled() -> bool:
    """Cheap predicate to skip building a heavy message body when the
    integration isn't configured."""
    return bool((get_settings().mlops_discord_webhook_url or "").strip())
