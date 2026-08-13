"""One-off: render I5 / I20 / I60 for a single stock at one as_of date.

Saves PNGs under outputs/temp. Not part of the main pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import OHLC_PARQUET, WINDOW_DAYS
from data.images import try_build_window
from data.parquet_io import load_calendar, permno_list, read_stock

OUT_DIR = ROOT / "outputs" / "temp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview I5/I20/I60 for one stock at one date")
    parser.add_argument("--permno", type=int, default=None, help="PERMNO; default is first in parquet")
    parser.add_argument("--as-of", type=str, default="2010-12-31", help="decision date YYYY-MM-DD")
    parser.add_argument("--parquet", type=Path, default=OHLC_PARQUET)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.parquet.exists():
        raise FileNotFoundError(f"parquet not found: {args.parquet}")

    permno = args.permno if args.permno is not None else int(permno_list(args.parquet)[0])
    as_of = pd.Timestamp(args.as_of)
    calendar = load_calendar(args.parquet)
    calendar_last = calendar[-1]
    stock_df = read_stock(args.parquet, permno)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for window_days in WINDOW_DAYS:
        built = try_build_window(
            stock_df=stock_df,
            permno=permno,
            as_of=as_of,
            window_days=window_days,
            calendar_last=calendar_last,
        )
        if built is None:
            raise ValueError(
                f"no image for PERMNO={permno} I{window_days} as_of={as_of.date()} "
                "(as_of must be a trading day on this stock with enough history)"
            )
        image, _meta = built
        path = args.out_dir / f"PERMNO{permno}_{as_of.date()}_I{window_days}.png"
        fig, ax = plt.subplots(figsize=(max(4, window_days * 0.12), 4))
        ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(f"I{window_days}  PERMNO={permno}  {as_of.date()}  {image.shape}")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path} shape={image.shape}")


if __name__ == "__main__":
    main()
