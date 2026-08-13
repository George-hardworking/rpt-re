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
# Paper: top 4/5 OHLC, bottom 1/5 volume; no gap row. Heights 32/64/96.
IMAGE_HEIGHT: dict[int, int] = {5: 32, 20: 64, 60: 96}
IMAGE_LAYOUT: dict[int, tuple[int, int]] = {
    w: (h - int(round(h / 5)), int(round(h / 5))) for w, h in IMAGE_HEIGHT.items()
}

IMAGE_FREQ_DIR = {
    5: "5d_week",
    20: "20d_month",
    60: "60d_quarter",
}

PIXEL_ON = 255
PIXEL_OFF = 0

CSV_CHUNKSIZE = 500_000

OHLC_DTYPES: dict[str, str] = {
    "PERMNO": "int64",
    "HdrCUSIP": "string",
    "Ticker": "string",
    "PERMCO": "int64",
    "DlyCalDt": "string",
    "DlyCap": "float64",
    "DlyRet": "float64",
    "DlyRetx": "float64",
    "DlyVol": "float64",
    "DlyClose": "float64",
    "DlyLow": "float64",
    "DlyHigh": "float64",
    "DlyOpen": "float64",
}

PREPARE_COLS = [
    "PERMNO",
    "DlyCalDt",
    "DlyCap",
    "DlyRet",
    "DlyVol",
    "DlyClose",
    "DlyLow",
    "DlyHigh",
    "DlyOpen",
]
