#!/usr/bin/env python3
"""Detokenize generated token ids from the toy VM inference path."""
import argparse
import sys


def read_ids(path: str) -> list[int]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return [int(x) for x in text.split()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-0.5B-Instruct",
        help="Hugging Face model name or local tokenizer directory.",
    )
    parser.add_argument(
        "--ids",
        default="output_tokens.txt",
        help="Input text file containing whitespace-separated token ids.",
    )
    parser.add_argument(
        "--skip-special",
        action="store_true",
        help="Skip special tokens during decode.",
    )
    args = parser.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: install transformers first: pip install transformers", file=sys.stderr)
        return 1

    ids = read_ids(args.ids)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = tok.decode(ids, skip_special_tokens=args.skip_special)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
