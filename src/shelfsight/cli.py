"""Command-line interface for training and benchmark-image inference."""

import argparse
import json
from pathlib import Path

from .data import download


def main():
    parser = argparse.ArgumentParser(description="ShelfSight apparel vision benchmark")
    parser.add_argument("command", choices=["download", "run", "predict"])
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "download":
        print(download(args.root))
    elif args.command == "predict":
        if args.input is None or args.output is None:
            parser.error("predict requires --input and --output")
        from .inference import predict_image

        print(json.dumps(predict_image(args.root, args.input, args.output), indent=2))
    else:
        from .pipeline import run

        print(run(args.root).to_string(index=False))


if __name__ == "__main__":
    main()
