from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"

RAW_OHLC_CSV = DATA_ROOT / "raw" / "OHLC_92_24.csv"
PROCESSED_DIR = DATA_ROOT / "processed"
OHLC_PARQUET = PROCESSED_DIR / "ohlc_daily"
IMAGES_ROOT = PROCESSED_DIR / "images"

SAMPLE_START = "1993-01-01"
SAMPLE_END = "2019-12-31"

WINDOW_DAYS = (5, 20, 60)
FUTURE_HORIZONS = (5, 20, 60)

COLS_PER_DAY = 3
IMAGE_LAYOUT: dict[int, tuple[int, int, int]] = {
    5: (25, 1, 6),
    20: (51, 1, 12),
    60: (76, 1, 19),
}

IMAGE_FREQ_DIR = {
    5: "5d_week",
    20: "20d_month",
    60: "60d_quarter",
}

PIXEL_ON = 255
PIXEL_OFF = 0

CSV_CHUNKSIZE = 500_000
