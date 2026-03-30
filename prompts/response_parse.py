"""
Parse model chat-completion text into a top-level JSON array of findings.

Shared by `tests/test_model_request.py` and `validator.py` so parsing rules stay
identical.
"""

from __future__ import annotations

import json


def parse_findings_json_array(content: str) -> list | None:
    """
    Extract a JSON array from the model reply.

    Prefer a message that is only JSON. If the model prepends chain-of-thought,
    accept the first top-level JSON array that consumes the rest of the string
    (aside from trailing whitespace).

    Returns:
        The parsed list on success, or None if no valid array was found.
    """
    text = content.strip()
    try:
        val = json.loads(text)
        if isinstance(val, list):
            return val
    except json.JSONDecodeError:
        pass

    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "[":
            continue
        try:
            val, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(val, list):
            continue
        if text[end:].strip():
            continue
        return val
    return None
