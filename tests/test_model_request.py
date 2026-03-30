"""
Test script – send a single vision inference request to the local model
using prompts/system_prompt.py (system + user message) and print the model’s
JSON array of finding objects only (no report wrapper).

Usage
-----
    python tests/test_model_request.py --image path/to/image.jpg
    python tests/test_model_request.py --image-url https://example.com/xray.png
    python tests/test_model_request.py --image x.png --age 47 --sex M
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prompts.response_parse import parse_findings_json_array
from prompts.system_prompt import build_chutes_messages

from model_utils import DEFAULT_HOST, DEFAULT_MODEL, DEFAULT_PORT, image_to_data_url, post_json


def main() -> None:
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", metavar="FILE")
    src.add_argument("--image-url", metavar="URL")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--age",
        type=int,
        default=None,
        help="Patient age at acquisition (optional; passed to build_user_message).",
    )
    parser.add_argument(
        "--sex",
        choices=("M", "F"),
        default=None,
        help="Patient sex (optional; passed to build_user_message).",
    )
    parser.add_argument(
        "--system-role",
        action="store_true",
        help=(
            "Send SYSTEM_PROMPT as a separate system message (some APIs require this; "
            "many local servers reject it with 400 — default is merged into the user text)."
        ),
    )
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}/v1/chat/completions"

    demographics: dict = {}
    if args.age is not None:
        demographics["age_at_acquisition"] = args.age
    if args.sex is not None:
        demographics["sex"] = args.sex

    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"ERROR: file not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        image_url = image_to_data_url(image_path)
    else:
        image_url = args.image_url

    payload = {
        "model": args.model,
        "messages": build_chutes_messages(
            image_url,
            demographics,
            merge_system_into_user=not args.system_role,
        ),
        "max_tokens": 1024,
        "temperature": 0.0,
    }

    try:
        response = post_json(url, payload)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    content = response["choices"][0]["message"]["content"]
    parsed = parse_findings_json_array(content)
    if parsed is None:
        print(
            "ERROR: could not parse a JSON array from the reply (expected only [], "
            "or prompts/system_prompt.py format).",
            file=sys.stderr,
        )
        print(content)
        sys.exit(1)

    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    main()
