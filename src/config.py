import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_DATA_ROOT = Path("/data/kaibiao/data/projects/rpt-re")
DATA_ROOT = Path(os.environ.get("RPT_DATA_ROOT", _DEFAULT_DATA_ROOT))

MARKET_US = "us"
MARKET_CN = "cn"


def market_raw_dir(market: str) -> Path:
    return DATA_ROOT / "raw" / market


def market_processed_dir(market: str) -> Path:
    return DATA_ROOT / "processed" / market


# US defaults (post-migration: processed/us/, raw/us/)
PROCESSED_DIR = market_processed_dir(MARKET_US)
RAW_OHLC_CSV = market_raw_dir(MARKET_US) / "OHLC_92_24.csv"
OHLC_PARQUET = PROCESSED_DIR / "ohlc_daily"
OHLC_CALENDAR = PROCESSED_DIR / "ohlc_calendar.parquet"
FEATURES_PARQUET = PROCESSED_DIR / "features"
IMAGES_ROOT = PROCESSED_DIR / "images"
MODELS_ROOT = PROCESSED_DIR / "models"
OUTPUT_ROOT = PROJECT_ROOT / "output"
BACKTEST_ROOT = OUTPUT_ROOT

BACKTEST_N_GROUP = 10
BACKTEST_WEIGHT_SCHEMES = ("equal", "float", "total")
# Rebalance / annualization keyed by forecast horizon R (not image window I).
BACKTEST_PERIODS_PER_YEAR = {5: 52, 20: 12, 60: 4}

SAMPLE_START = "1993-01-01"
SAMPLE_END = "2019-12-31"
TRAIN_END = "2000-12-31"
TEST_START = "2001-01-01"

# CNN training (Jiang–Kelly–Xiu JF 2023, Section II.C)
N_ENSEMBLE = 5
TRAIN_VAL_SPLIT_SEED = 0  # fixed 70/30 split per Ix/Ry; ensemble seeds only affect optimization
ADAM_LR = 1e-5
BATCH_SIZE = 128
EARLY_STOP_PATIENCE = 2
TRAIN_VAL_FRAC = 0.7
MAX_EPOCHS = 100
# Independent (I, R, seed) processes packed onto remaining GPU memory.
GPU_MIN_FREE_GIB = 16.0
VRAM_PER_JOB_GIB = {5: 2.0, 20: 4.0, 60: 8.0}
TRAIN_JOB_RAM_GIB = 10.0
TRAIN_JOBS_PER_GPU = 4

WINDOW_DAYS = (5, 20, 60)
FUTURE_HORIZONS = (5, 20, 60)
# Author GenerateStockData.ret_len_list
LABEL_HORIZONS = (5, 20, 60, 65, 180, 250, 260)
EWMA_VOL_SPAN = 60

COLS_PER_DAY = 3
# Paper: top 4/5 OHLC, bottom 1/5 volume; no gap row. Heights 32/64/96.
IMAGE_HEIGHT: dict[int, int] = {5: 32, 20: 64, 60: 96}
IMAGE_LAYOUT: dict[int, tuple[int, int]] = {
    w: (h - int(round(h / 5)), int(round(h / 5))) for w, h in IMAGE_HEIGHT.items()
}

IMAGE_SAMPLE_FREQS = ("week", "month", "quarter")
# Paper: sample dates align with forecast horizon R, not image window I.
HORIZON_SAMPLE_FREQ = {5: "week", 20: "month", 60: "quarter"}
# Legacy diagonal bundles (window tied to default freq): I5/week, I20/month, I60/quarter.
WINDOW_DEFAULT_SAMPLE_FREQ = {5: "week", 20: "month", 60: "quarter"}

PAPER_CROSS_BUNDLES: tuple[tuple[int, str], ...] = (
    (20, "week"),
    (60, "week"),
    (5, "month"),
    (60, "month"),
    (5, "quarter"),
    (20, "quarter"),
)
PAPER_CROSS_CONFIGS: tuple[tuple[int, int], ...] = (
    (20, 5),
    (60, 5),
    (5, 20),
    (60, 20),
    (5, 60),
    (20, 60),
)
PAPER_DIAGONAL_CONFIGS: tuple[tuple[int, int], ...] = (
    (5, 5),
    (20, 20),
    (60, 60),
)

IMAGES_CHECKPOINT_PAPER_CROSS = ".checkpoint_permnos_paper_cross"


def image_bundle_dir(window_days: int, sample_freq: str) -> str:
    if sample_freq not in IMAGE_SAMPLE_FREQS:
        raise ValueError(f"unsupported sample_freq={sample_freq}")
    return f"{window_days}d_{sample_freq}"


def sample_freq_for_horizon(horizon: int) -> str:
    if horizon not in HORIZON_SAMPLE_FREQ:
        raise ValueError(f"unsupported horizon={horizon}")
    return HORIZON_SAMPLE_FREQ[horizon]


def diagonal_bundles() -> tuple[tuple[int, str], ...]:
    return tuple((w, WINDOW_DEFAULT_SAMPLE_FREQ[w]) for w in WINDOW_DAYS)


def windows_to_diagonal_bundles(window_days_list: tuple[int, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((w, WINDOW_DEFAULT_SAMPLE_FREQ[w]) for w in window_days_list)


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
