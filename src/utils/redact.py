"""Shared secret-redaction helpers used across collectors.

These were originally defined inside ``kma_mid_temperature.py`` for the
data.go.kr serviceKey leak fix. They're factored out here so the EIA STEO
collector (and any future API collector) can apply the same multi-encoding
discipline:

  * The RAW value (what the user put in .env)
  * The URL-ENCODED form (what ``requests.get(params=...)`` actually
    transmits when the value contains ``/`` ``+`` ``=`` etc.)
  * The URL-DECODED form (in case the user supplied an already-encoded
    value and the server normalises it before echoing back)

Without all three forms, an upstream connection error surfaced by urllib3
— ``HTTPSConnectionPool: ... url: /v2/steo?api_key=AbCd%2FEf%2B...`` —
would slip the URL-encoded key past a simple ``str.replace(raw_key, ...)``.
"""

from __future__ import annotations

from urllib.parse import quote, unquote


# Minimum length before a value is considered a credible secret worth
# substring-masking. Below this we risk wiping legitimate substrings
# (e.g. a dev-mode placeholder set to "x" or "dev").
_MIN_SECRET_LEN = 12


def secret_variants(value: str | None) -> list[str]:
    """Return every form of `value` worth scrubbing from a piece of text.

    Always emits at minimum the value itself, plus its URL-encoded form
    and its URL-decoded form when those differ. Filters out short values
    (< 12 chars) on both the input and any derived form, since masking
    short substrings causes too many false positives.
    """
    if not value or len(value) < _MIN_SECRET_LEN:
        return []
    out: set[str] = {value}
    try:
        encoded = quote(value, safe="")
        if encoded:
            out.add(encoded)
    except Exception:
        pass
    try:
        decoded = unquote(value)
        if decoded:
            out.add(decoded)
    except Exception:
        pass
    return [v for v in out if len(v) >= _MIN_SECRET_LEN]


def redact_secrets_in_text(text: str, secrets: list[str]) -> str:
    """Substring-mask every occurrence of any known secret in ``text``.

    Sort by length descending so longer, more specific variants are
    masked first — otherwise a shorter substring that overlaps with a
    longer one could leave partial leakage.
    """
    if not text or not secrets:
        return text
    for secret in sorted(set(secrets), key=len, reverse=True):
        if secret and secret in text:
            text = text.replace(secret, "<REDACTED>")
    return text
