"""Phase 4 step 1b — call the Gemini API, rotate across N keys, force JSON-Schema output.

Replaces ollama.py as the backend (see docs/decisions/phase4-gemini-backend.md);
same chat_json() signature so callers switch backends with one import line.

Keys rotate on EVERY call, not just on failure — this spreads load across each
key's own free-tier quota. A transient error (429, 5xx) retries with the next
key in rotation, up to len(KEYS) times; non-transient errors (bad auth, bad
args) raise immediately since retrying would just repeat the same failure.

Gemini's response_schema is an OpenAPI subset, not full JSON Schema:
`additionalProperties` is rejected with a 400, but `minLength`/`enum`/
`description`/`required` (including nested) all pass through — so
`_to_gemini_schema()` only strips `additionalProperties`.

thinking_level=MINIMAL: this is structured extraction, not open reasoning —
default thinking burns hundreds of wasted "thinking" tokens per call.
"""

from __future__ import annotations

import itertools
import json
import os
import time

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

load_dotenv()

# Measured on the full 75-topic dev set (docs/phase4-tong-ket.md 6.5):
# gemini-3.6-flash extracts more (16.9 vs 13.0 values/patient) and especially
# more NEGATIONS (92% vs 58% recall) — but Lite is 2.8x faster with its own
# quota, deciding Phase 8's feasibility (27,045 calls). Lite is a deliberate
# trade-off: its lower negation recall weakens `satisfied` conclusions for
# exclusion criteria in Phase 8, a cost that must be measured, not forgotten.
MODEL = "gemini-3.5-flash-lite"
_RETRYABLE_CODES = {429, 500, 502, 503, 504}


class GeminiError(RuntimeError):
    pass


def _load_keys() -> list[str]:
    keys = []
    i = 1
    while True:
        k = os.environ.get(f"GEMINI_API_KEY_{i}")
        if not k:
            break
        keys.append(k)
        i += 1
    return keys


KEYS = _load_keys()
_cycle = itertools.cycle(KEYS) if KEYS else None


def _retry_delay_seconds(e: errors.APIError, default: float = 2.0, cap: float = 20.0) -> float:
    """Read Gemini's own `retryDelay` hint from a 429/5xx instead of guessing.

    If keys share a Google project, RPM quota is shared too — retrying
    instantly with the next key just fails again within the same window.
    Waiting for the server's actual retryDelay is what makes the retry work.
    """
    try:
        for d in e.details.get("error", {}).get("details", []):
            if str(d.get("@type", "")).endswith("RetryInfo"):
                return min(float(str(d["retryDelay"]).rstrip("s")), cap)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    return default


def _to_gemini_schema(node):
    """Strip `additionalProperties` from the schema tree — see module docstring."""
    if isinstance(node, dict):
        return {k: _to_gemini_schema(v) for k, v in node.items()
                if k != "additionalProperties"}
    if isinstance(node, list):
        return [_to_gemini_schema(v) for v in node]
    return node


def chat_json(model: str, system: str, user: str, schema: dict) -> tuple[dict | None, dict]:
    """One structured call. Returns (parsed_data | None, measurement_info).

    Same convention as ollama.chat_json: invalid JSON returns None rather than
    raising — that's a measured outcome (schema-valid rate), not a system
    error that should fail the whole benchmark run.
    """
    if not KEYS:
        raise GeminiError(
            "Khong co GEMINI_API_KEY_1/2/3 nao trong .env. Xem .env.example.")

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=_to_gemini_schema(schema),
        temperature=0,
        thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL),
    )

    last_err: Exception | None = None
    for _ in range(len(KEYS)):
        key = next(_cycle)
        client = genai.Client(api_key=key)  # kept alive — GC mid-call raises "client has been closed"
        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=model, contents=user, config=config)
        except errors.APIError as e:
            last_err = e
            if e.code in _RETRYABLE_CODES:
                time.sleep(_retry_delay_seconds(e))
                continue  # the cycle already advanced to the next key
            raise GeminiError(f"{model}: {e}") from e
        except httpx.TransportError as e:
            # Network-layer failure (DNS, connection, timeout) — happens
            # BEFORE reaching Google's server, so it's not errors.APIError and
            # the SDK doesn't wrap it; left uncaught it crashes the whole
            # process. Treated as transient like 429/5xx: no retryDelay to
            # read (no response at all), so wait a fixed interval and retry.
            last_err = e
            time.sleep(2.0)
            continue
        elapsed = time.time() - t0

        usage = resp.usage_metadata
        meta = {
            "seconds": round(elapsed, 2),
            "prompt_tokens": usage.prompt_token_count if usage else None,
            "output_tokens": usage.candidates_token_count if usage else None,
            "raw": resp.text,
        }
        try:
            data = json.loads(resp.text)
        except (json.JSONDecodeError, TypeError):
            return None, meta
        return (data if isinstance(data, dict) else None), meta

    raise GeminiError(f"{model}: het {len(KEYS)} key deu loi tam thoi, "
                       f"loi cuoi: {last_err}")


def self_test() -> bool:
    """Check that each key authenticates and the filtered schema still works.

    Run by hand once before the first extraction batch:
        python -m src.extraction.gemini --self-test
    Not called automatically from extract.py — this is a real, billed API
    call, unlike Ollama's cheap local is_up() check.
    """
    probe_schema = {"type": "object",
                     "properties": {"ok": {"type": "boolean"}},
                     "required": ["ok"], "additionalProperties": False}
    all_ok = True
    for i in range(1, len(KEYS) + 1):
        try:
            data, meta = chat_json(MODEL, "Reply with ok=true.",
                                    "Confirm you are working.", probe_schema)
            ok = isinstance(data, dict) and data.get("ok") is True
            print(f"key {i}: {'OK' if ok else 'JSON khong nhu mong doi'} "
                  f"({meta['seconds']}s) {data}")
            all_ok &= ok
        except GeminiError as e:
            print(f"key {i}: LOI {e}")
            all_ok = False
    return all_ok


if __name__ == "__main__":
    import sys

    if "--self-test" in sys.argv:
        sys.exit(0 if self_test() else 1)
    print(f"{len(KEYS)} GEMINI_API_KEY_* da nap tu .env. "
          f"Dung: python -m src.extraction.gemini --self-test")
