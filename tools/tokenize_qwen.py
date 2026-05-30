#!/usr/bin/env python3
"""Tokenize text for the toy VM token-level inference path."""
import argparse
from collections.abc import Mapping
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Hugging Face model name or local tokenizer directory.",
    )
    parser.add_argument("--text", help="Prompt text. Defaults to stdin.")
    parser.add_argument(
        "--out",
        default="input_tokens.txt",
        help="Output text file containing whitespace-separated token ids.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use tokenizer.apply_chat_template for Qwen-Instruct style prompts.",
    )
    parser.add_argument(
        "--no-special",
        action="store_true",
        help="Pass add_special_tokens=False when not using --chat.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: install transformers first: pip install transformers", file=sys.stderr)
        return 1

    text = args.text if args.text is not None else sys.stdin.read()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.chat:
        messages = [{"role": "user", "content": text}]
        ids = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        ids = tok.encode(text, add_special_tokens=not args.no_special)

    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    elif hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif hasattr(ids, "data") and isinstance(ids.data, Mapping):
        ids = ids.data["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]

    ids = [int(i) for i in ids]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(" ".join(str(i) for i in ids))
        f.write("\n")

    print(f"wrote {len(ids)} token ids to {args.out}")
    print(" ".join(str(i) for i in ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
