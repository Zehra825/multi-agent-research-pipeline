#!/usr/bin/env python3
"""
CLI entry point.

Usage:
    python main.py --ticker AAPL
    python main.py --ticker MSFT --period 1y --output-dir ./output
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from core.orchestrator import ResearchPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent stock research pipeline.")
    parser.add_argument("--ticker", required=True, help="Stock ticker symbol, e.g. AAPL")
    parser.add_argument(
        "--period",
        default="6mo",
        help="Price history window (yfinance format: 1mo, 3mo, 6mo, 1y, 2y). Default: 6mo",
    )
    parser.add_argument(
        "--output-dir", default="output", help="Directory for the report + chart. Default: output"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress step-by-step trace output")
    return parser.parse_args()


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        pipeline = ResearchPipeline(verbose=not args.quiet)
        result = pipeline.run(args.ticker, output_dir=args.output_dir, period=args.period)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"\nReport written to: {result['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
