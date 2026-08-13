"""Orchestrate CSV → parquet → image dataset generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import IMAGES_ROOT, OHLC_PARQUET, RAW_OHLC_CSV, WINDOW_DAYS
from data.build_images import build_window_images
from data.prepare import prepare_ohlc_parquet


def run_prepare(raw_path: Path = RAW_OHLC_CSV, parquet_path: Path = OHLC_PARQUET) -> Path:
    path = prepare_ohlc_parquet(raw_path, parquet_path)
    print(f"prepared {path}")
    return path


def run_build(
    parquet_path: Path = OHLC_PARQUET,
    output_root: Path = IMAGES_ROOT,
    window_days_list: tuple[int, ...] = WINDOW_DAYS,
    permno_limit: int | None = None,
) -> None:
    build_window_images(
        parquet_path=parquet_path,
        output_root=output_root,
        window_days_list=window_days_list,
        permno_limit=permno_limit,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="OHLC data preparation and image build pipeline")
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
        run_prepare(args.raw, args.parquet)
    elif args.command == "build":
        run_build(
            parquet_path=args.parquet,
            output_root=args.output,
            window_days_list=tuple(args.windows),
            permno_limit=args.permno_limit,
        )
    elif args.command == "all":
        run_prepare(args.raw, args.parquet)
        run_build(
            parquet_path=args.parquet,
            output_root=args.output,
            window_days_list=tuple(args.windows),
            permno_limit=args.permno_limit,
        )


if __name__ == "__main__":
    main()
