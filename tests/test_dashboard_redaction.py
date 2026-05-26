"""Dashboard preview redaction canary.

The dashboard's Page 9 ("Round 8-9 collectors") shows the first 4 KB of a
raw API envelope so operators can sanity-check what was persisted. Even
though every collector is supposed to redact at write time, the preview is
a separate trust boundary: a future collector added without redaction, a
data.go.kr error envelope that echoes the key in its message body, or a
stale on-disk file from before the redaction fix would all leak straight
into the rendered page.

This test pins the display-time redactor's behaviour so that bug never
regresses silently.
"""

from __future__ import annotations


import pytest


@pytest.fixture
def redactor():
    # Import inside the fixture so a syntax error in dashboard.py fails the
    # test loudly rather than at module-collection time.
    from dashboard import _redact_preview
    return _redact_preview


def test_redacts_data_go_kr_servicekey_in_query_string(redactor):
    url = (
        "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        "?serviceKey=ABCdef123%2F%2BkeyVALUE%3D&base_date=20260526&nx=60&ny=127"
    )
    out = redactor(url)
    assert "ABCdef123" not in out
    assert "keyVALUE" not in out
    assert "<REDACTED>" in out
    # Non-credential params survive
    assert "base_date=20260526" in out
    assert "nx=60" in out


def test_redacts_servicekey_inside_json_body(redactor):
    body = '{"serviceKey":"ABCdef123%2FkeyVALUE","regId":"11B10101"}'
    out = redactor(body)
    assert "ABCdef123" not in out
    assert "<REDACTED>" in out
    assert '"regId":"11B10101"' in out


def test_redacts_eia_api_key_query_param(redactor):
    url = "https://api.eia.gov/v2/steo/data/?api_key=SECRETkey789ABC&frequency=monthly"
    out = redactor(url)
    assert "SECRETkey789ABC" not in out
    assert "<REDACTED>" in out
    assert "frequency=monthly" in out


def test_redacts_camelcase_apikey_in_json(redactor):
    body = '{"apiKey": "live_xyz_supersecret", "other": "ok"}'
    out = redactor(body)
    assert "live_xyz_supersecret" not in out
    assert "<REDACTED>" in out
    assert '"other": "ok"' in out


def test_redacts_authorization_bearer_header(redactor):
    body = 'Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature'
    out = redactor(body)
    assert "eyJhbGciOiJIUzI1NiJ9.payload.signature" not in out
    assert "<REDACTED>" in out


def test_redacts_x_api_key_and_subscription_key(redactor):
    body = (
        '{"x-api-key":"FOO_BAR_BAZ_1234567890","subscription-key":"AAA-BBB-CCC-DDD"}'
    )
    out = redactor(body)
    assert "FOO_BAR_BAZ_1234567890" not in out
    assert "AAA-BBB-CCC-DDD" not in out


def test_redacts_live_env_var_values_even_outside_key_patterns(
    redactor, monkeypatch
):
    """Belt-and-braces: if a key fragment appears in an upstream error
    string (e.g. ``"Provided key 'ABCDEF...' was not registered."``), the
    key=value regex won't match it. But we still know what the live key
    value IS, so substring-mask any occurrence.
    """
    fake_key = "VERYsecretLIVEkeyVALUE12345"
    monkeypatch.setenv("KMA_PUBLIC_API_KEY", fake_key)
    text = (
        '<errMsg>SERVICE ERROR</errMsg>'
        f'<returnAuthMsg>Provided key {fake_key} was not registered.</returnAuthMsg>'
    )
    out = redactor(text)
    assert fake_key not in out
    assert "<REDACTED>" in out


def test_does_not_mask_short_env_var_values(redactor, monkeypatch):
    """An env var with a 4-char value like 'dev' should not trigger
    substring masking — too many false positives on real JSON content.
    """
    monkeypatch.setenv("KMA_PUBLIC_API_KEY", "dev")  # too short to mask
    body = '{"baseDate":"20260526","developer":"alice"}'
    out = redactor(body)
    assert "alice" in out
    assert "developer" in out  # `dev` substring must not be masked


# ---------------------------------------------------------------------------
# Gap-closing canaries — patterns Codex called out as still leaky.
# ---------------------------------------------------------------------------


def test_redacts_naked_key_token_auth_password(redactor):
    """Bare credential-shaped key names without the `api_` prefix."""
    cases = [
        ("https://example.com/?key=NAKEDkey1234567890abcd", "NAKEDkey1234567890abcd"),
        ("https://example.com/?token=NAKEDtoken1234567890abcd", "NAKEDtoken1234567890abcd"),
        ("https://example.com/?auth=NAKEDauth1234567890abcd", "NAKEDauth1234567890abcd"),
        ('{"secret_key":"NAKEDsecret1234567890abcd"}', "NAKEDsecret1234567890abcd"),
        ('{"client_secret":"NAKEDclient1234567890abcd"}', "NAKEDclient1234567890abcd"),
        ('{"private_key":"-----BEGIN-PRIVATE-KEY-DATA-----"}', "BEGIN-PRIVATE-KEY-DATA"),
        ('aws_access_key_id=AKIAIOSFODNN7EXAMPLE', "AKIAIOSFODNN7EXAMPLE"),
        ('{"password":"hunter2hunter2hunter2"}', "hunter2hunter2hunter2"),
        ('pwd=hunter2hunter2hunter2', "hunter2hunter2hunter2"),
    ]
    for body, sensitive in cases:
        out = redactor(body)
        assert sensitive not in out, f"leaked {sensitive!r} from {body!r} → {out!r}"
        assert "<REDACTED>" in out


def test_redacts_cookie_and_session_headers(redactor):
    """Cookie / session ID leaks — common when an upstream proxy echoes
    request headers back in an error envelope.
    """
    cases = [
        ('Cookie: JSESSIONID=ABCDEFGH1234567890XYZ', "ABCDEFGH1234567890XYZ"),
        ('Set-Cookie: PHPSESSID=AAABBBCCCDDDEEEFFF111', "AAABBBCCCDDDEEEFFF111"),
        ('{"sessionId":"sess_abc123def456ghi789"}', "sess_abc123def456ghi789"),
        ('{"sso_token":"sso_VALUE_1234567890abcdef"}', "sso_VALUE_1234567890abcdef"),
    ]
    for body, sensitive in cases:
        out = redactor(body)
        assert sensitive not in out, f"leaked {sensitive!r} from {body!r} → {out!r}"


def test_redacts_naked_jwt_without_bearer_prefix(redactor):
    """JWTs echoed in error messages without a `Bearer ` scheme word."""
    jwt = (
        "eyJhbGciOiJIUzI1NiJ9"
        ".eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4ifQ"
        ".SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    )
    body = f'{{"error":"Invalid token: {jwt}"}}'
    out = redactor(body)
    assert jwt not in out
    assert "<REDACTED>" in out
    # Sibling fields are untouched
    assert '"error"' in out


def test_redacts_authorization_basic_and_token_schemes(redactor):
    """Single-token schemes: Bearer, Basic, Token, Digest."""
    cases = [
        ("Authorization: Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
         "QWxhZGRpbjpvcGVuIHNlc2FtZQ=="),
        ("Authorization: Token 1234567890abcdef1234567890abcdef",
         "1234567890abcdef1234567890abcdef"),
        ("Authorization: Digest abcdef1234567890longerThan16Chars",
         "abcdef1234567890longerThan16Chars"),
    ]
    for body, sensitive in cases:
        out = redactor(body)
        assert sensitive not in out, f"leaked {sensitive!r} from {body!r}"


def test_redacts_parameterized_authorization_headers(redactor):
    """Parameterised Authorization headers — HMAC, AWS SigV4, OAuth1, Digest
    with multiple internal parameters. Each one packs the credential
    inside fields whose names (``signature``, ``response``, ``Signature``,
    ``oauth_signature``) don't match the generic key keyword set.

    The fix: wholesale masking of everything after the scheme word. The
    keyId / Credential identifier is masked along with the signature
    because separating "public identifier" from "secret" requires knowing
    the scheme — and false positives in a preview are cheap, leaks aren't.
    """
    cases = [
        # HMAC parameterised
        (
            "Authorization: HMAC keyId=AKID, signature=longSecretValueXYZ",
            ["longSecretValueXYZ"],
        ),
        # AWS Signature V4
        (
            "Authorization: AWS4-HMAC-SHA256 "
            "Credential=AKIAIOSFODNN7EXAMPLE/20260526/us-east-1/s3/aws4_request, "
            "SignedHeaders=host;x-amz-date, "
            "Signature=fe5f80f77d5fa3beca038a248ff027d0445342fe2855ddc963176630326f1024",
            [
                "AKIAIOSFODNN7EXAMPLE",
                "fe5f80f77d5fa3beca038a248ff027d0445342fe2855ddc963176630326f1024",
            ],
        ),
        # OAuth1 quoted parameters
        (
            'Authorization: OAuth oauth_consumer_key="ckey123ABC456", '
            'oauth_token="tkn789DEF012", '
            'oauth_signature="sigGHI345jkl678MNO"',
            ["ckey123ABC456", "tkn789DEF012", "sigGHI345jkl678MNO"],
        ),
        # Digest with response= parameter
        (
            'Authorization: Digest username="alice", realm="example", '
            'nonce="abc123def456", '
            'response="6629fae49393a05397450978507c4ef1"',
            ["6629fae49393a05397450978507c4ef1"],
        ),
        # AWS classic
        (
            "Authorization: AWS AKIDEXAMPLE:longBase64SignaturePlusMore==",
            ["AKIDEXAMPLE", "longBase64SignaturePlusMore=="],
        ),
    ]
    for body, sensitive_values in cases:
        out = redactor(body)
        for sensitive in sensitive_values:
            assert sensitive not in out, (
                f"leaked {sensitive!r} from {body!r}\n→ {out!r}"
            )
        assert "<REDACTED>" in out


def test_redacts_json_quoted_authorization_header(redactor):
    """The HTTP regex stops at `"` so it doesn't mangle JSON structure;
    the JSON regex handles the quoted form cleanly and preserves the
    surrounding object's structure.
    """
    body = (
        '{"Authorization":"Bearer eyJhbGciOiJIUzI1NiJ9.payload.signaturePart",'
        '"Content-Type":"application/json"}'
    )
    out = redactor(body)
    assert "eyJhbGciOiJIUzI1NiJ9.payload.signaturePart" not in out
    assert '"Authorization":"<REDACTED>"' in out
    # Surrounding JSON object is intact — Content-Type field survives
    assert '"Content-Type":"application/json"' in out


def test_redacts_json_quoted_parameterized_authorization(redactor):
    """JSON-quoted HMAC/AWS-style header — the entire value between the
    JSON quotes is wholesale-masked.
    """
    body = (
        '{"Authorization": "AWS4-HMAC-SHA256 '
        'Credential=AKIASECRET123/20260526/us-east-1/s3/aws4_request, '
        'Signature=fe5f80f77d5fa3beca038a248ff027d0445342fe2855ddc"}'
    )
    out = redactor(body)
    assert "AKIASECRET123" not in out
    assert "fe5f80f77d5fa3beca038a248ff027d0445342fe2855ddc" not in out
    # JSON structure preserved
    assert out.endswith('"}')


def test_authorization_header_inside_json_string_value_does_not_leak(redactor):
    """An error message that ECHOES a raw Authorization header inside a
    JSON string value (e.g. ``{"error":"Provided Authorization: Bearer abc
    is invalid"}``) must not leak the token, even though this mangles the
    JSON structure of the description field. The leak is the bug — the
    mangling is acceptable in a preview.
    """
    body = (
        '{"error":"Provided Authorization: Bearer SENSITIVEtoken1234567 '
        'is invalid"}'
    )
    out = redactor(body)
    assert "SENSITIVEtoken1234567" not in out
    assert "<REDACTED>" in out


def test_auto_discovers_credential_env_vars_not_in_hardcoded_list(
    redactor, monkeypatch
):
    """A future collector adding e.g. WEATHER_API_KEY or JKM_AUTH_TOKEN
    must be masked even though we never wrote its name into the dashboard.
    """
    monkeypatch.setenv("WEATHER_API_KEY", "SECRETweather1234567890ABCD")
    monkeypatch.setenv("JKM_AUTH_TOKEN", "SECRETjkm1234567890ABCDEFGH")
    body = (
        'Some upstream error string mentions SECRETweather1234567890ABCD '
        'and also SECRETjkm1234567890ABCDEFGH in a free-text field.'
    )
    out = redactor(body)
    assert "SECRETweather1234567890ABCD" not in out
    assert "SECRETjkm1234567890ABCDEFGH" not in out
    assert out.count("<REDACTED>") >= 2


def test_substring_masks_both_raw_and_urlencoded_forms_of_env_value(
    redactor, monkeypatch
):
    """data.go.kr issues the same service key in raw AND URL-encoded form.
    If our env holds the raw value but the on-disk envelope carries the
    encoded form (or vice versa), substring matching must catch both.
    """
    raw_key = "ABC/DEF+GHI=JKL=longSecretValueXYZ"
    encoded_key = "ABC%2FDEF%2BGHI%3DJKL%3DlongSecretValueXYZ"
    monkeypatch.setenv("KMA_PUBLIC_API_KEY", raw_key)
    # On-disk envelope carries the encoded form
    body_with_encoded = f"http://x/api?serviceKey={encoded_key}&base_date=20260526"
    out1 = redactor(body_with_encoded)
    assert encoded_key not in out1
    assert raw_key not in out1
    # And the raw form is also masked when env holds raw
    body_with_raw = f"error: provided key '{raw_key}' is invalid"
    out2 = redactor(body_with_raw)
    assert raw_key not in out2


def test_does_not_mask_env_vars_without_credential_shaped_names(
    redactor, monkeypatch
):
    """A long but non-credential env var (e.g. DATABASE_URL) is NOT
    auto-masked just because of its length — only credential-shaped names
    qualify. This prevents accidental masking of, say, a publicly-known
    base URL in the preview.
    """
    monkeypatch.setenv("DATABASE_URL", "postgresql://user@host/dbnameXYZ1234567")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://api.example.com/v2/long-path-here")
    body = "Connecting to postgresql://user@host/dbnameXYZ1234567 ok"
    out = redactor(body)
    # DATABASE_URL is NOT credential-shaped → not auto-masked
    assert "postgresql://user@host/dbnameXYZ1234567" in out


def test_value_capture_stops_at_xml_and_json_array_terminators(redactor):
    """Values living inside `<element>VALUE</element>` or `["VALUE"]` shouldn't
    spill into the closing markup."""
    xml = '<returnAuthMsg>SERVICE ERROR with key=ABC123longvalue</returnAuthMsg>'
    out = redactor(xml)
    assert "ABC123longvalue" not in out
    assert "</returnAuthMsg>" in out  # closing tag is preserved (not consumed)


def test_idempotent_redaction(redactor):
    """Running the redactor twice produces the same output as once.
    Prevents accidental double-substitution patterns from compounding."""
    body = (
        '{"serviceKey":"ABCkey1234567890","api_key":"DEFkey0987654321",'
        '"data":[1,2,3]}'
    )
    once = redactor(body)
    twice = redactor(once)
    assert once == twice
    assert "ABCkey1234567890" not in once
    assert "DEFkey0987654321" not in once


def test_preserves_non_credential_payload(redactor):
    body = (
        '{"response":{"header":{"resultCode":"00","resultMsg":"NORMAL_SERVICE"},'
        '"body":{"items":{"item":[{"baseDate":"20260526","fcstValue":"21"}]}}}}'
    )
    out = redactor(body)
    # No credentials → identical output
    assert out == body


def test_handles_empty_and_none(redactor):
    assert redactor("") == ""
    assert redactor(None) is None  # pragma: no cover — defensive
