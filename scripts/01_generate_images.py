"""CLI: CSV → parquet → price trend image dataset (prepare / build / all)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import IMAGES_ROOT, OHLC_PARQUET, RAW_OHLC_CSV, WINDOW_DAYS
from data.build_images import build_window_images
from data.prepare import prepare_ohlc_parquet


def main() -> None:
    parser = argparse.ArgumentParser(description="OHLC data preparation and image generation")
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="CSV → partitioned parquet")
    p_prepare.add_argument("--raw", type=Path, default=RAW_OHLC_CSV)
    p_prepare.add_argument("--parquet", type=Path, default=OHLC_PARQUET)

    p_build = sub.add_parser("build", help="parquet → yearly image memmaps + labels")
    p_build.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    p_build.add_argument("--output", type=Path, default=IMAGES_ROOT)
    p_build.add_argument("--windows", type=int, nargs="+", default=list(WINDOW_DAYS))
    p_build.add_argument("--permno-limit", type=int, default=None)

    p_all = sub.add_parser("all", help="prepare then build")
    p_all.add_argument("--raw", type=Path, default=RAW_OHLC_CSV)
    p_all.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    p_all.add_argument("--output", type=Path, default=IMAGES_ROOT)
    p_all.add_argument("--windows", type=int, nargs="+", default=list(WINDOW_DAYS))
    p_all.add_argument("--permno-limit", type=int, default=None)

    args = parser.parse_args()

    if args.command == "prepare":
        path = prepare_ohlc_parquet(args.raw, args.parquet)
        print(f"prepared {path}")
    elif args.command == "build":
        build_window_images(
            parquet_path=args.parquet,
            output_root=args.output,
            window_days_list=tuple(args.windows),
            permno_limit=args.permno_limit,
        )
    elif args.command == "all":
        path = prepare_ohlc_parquet(args.raw, args.parquet)
        print(f"prepared {path}")
        build_window_images(
            parquet_path=args.parquet,
            output_root=args.output,
            window_days_list=tuple(args.windows),
            permno_limit=args.permno_limit,
        )


if __name__ == "__main__":
    main()
