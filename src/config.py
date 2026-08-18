import os
from dataclasses import dataclass
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
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
BACKTEST_STEP04_ROOT = OUTPUT_ROOT / "04_backtest"
BACKTEST_CNN_ROOT = BACKTEST_STEP04_ROOT / "cnn_baseline"
BACKTEST_CNN_TOP500_ROOT = BACKTEST_STEP04_ROOT / "cnn_top500"
BACKTEST_CNN_TOP500_FLOAT_ROOT = BACKTEST_STEP04_ROOT / "cnn_top500_float"
BACKTEST_CN_FACTOR_ROOT = BACKTEST_STEP04_ROOT / "cn_factors"
BACKTEST_BREAKDOWN_ROOT = OUTPUT_ROOT / "06_holding_breakdown"
BACKTEST_BENCHMARK_ROOT = OUTPUT_ROOT / "05_benchmark_signals"
BACKTEST_TOP_N_CAP = 500

# Step 07: US CNN weights -> China inference / head finetune (not processed/cn/models/).
CN_TRANSFER_FROM_US_ROOT = market_processed_dir(MARKET_CN) / "transfer_from_us"
TRANSFER_MODE_DIRECT = "direct"
TRANSFER_MODE_FINETUNE = "finetune"
TRANSFER_BACKTEST_ROOT = OUTPUT_ROOT / "07_transfer_us_cn"
TRANSFER_BACKTEST_DIRECT_ROOT = TRANSFER_BACKTEST_ROOT / TRANSFER_MODE_DIRECT
TRANSFER_BACKTEST_FINETUNE_ROOT = TRANSFER_BACKTEST_ROOT / TRANSFER_MODE_FINETUNE
TRANSFER_COMPARE_ROOT = TRANSFER_BACKTEST_ROOT / "compare"


def transfer_models_root(mode: str) -> Path:
    if mode not in (TRANSFER_MODE_DIRECT, TRANSFER_MODE_FINETUNE):
        raise ValueError(f"unsupported transfer mode={mode}")
    return CN_TRANSFER_FROM_US_ROOT / mode


def transfer_backtest_output_root(mode: str) -> Path:
    if mode == TRANSFER_MODE_DIRECT:
        return TRANSFER_BACKTEST_DIRECT_ROOT
    if mode == TRANSFER_MODE_FINETUNE:
        return TRANSFER_BACKTEST_FINETUNE_ROOT
    raise ValueError(f"unsupported transfer mode={mode}")


def us_seed_checkpoint(image_days: int, horizon: int, seed: int) -> Path:
    return (
        market_processed_dir(MARKET_US)
        / "models"
        / f"I{image_days}_R{horizon}"
        / f"seed{seed}"
        / "best.pt"
    )

# Paper Table III: monthly R20 holding split into days 1–5 vs 6–20.
HOLDING_BREAKDOWN_HORIZON = 20
HOLDING_BREAKDOWN_FIRST_DAYS = 5
RET_D1_D5 = "Ret_d1_d5"
RET_D6_D20 = "Ret_d6_d20"
HOLDING_BREAKDOWN_WINDOWS: tuple[tuple[str, str], ...] = (
    ("d1_d5", RET_D1_D5),
    ("d6_d20", RET_D6_D20),
)
HOLDING_BREAKDOWN_CNN_CONFIGS: tuple[tuple[int, int], ...] = (
    (5, HOLDING_BREAKDOWN_HORIZON),
    (20, HOLDING_BREAKDOWN_HORIZON),
    (60, HOLDING_BREAKDOWN_HORIZON),
)
HOLDING_BREAKDOWN_BENCH_COLS: tuple[str, ...] = (
    "MOM",
    "REV1m_STR",
    "REV1w_WSTR",
    "TREND_HZZ",
)

# Paper trend benchmarks (Jiang–Kelly–Xiu JF 2023 Table I; HZZ from Han–Zhou–Zhu 2016).
TREND_MOM_WINDOW = 252
TREND_MOM_SKIP = 21
TREND_STR_WINDOW = 21
TREND_WSTR_WINDOW = 5
TREND_52WH_WINDOW = 252
TREND_HZZ_MA_LAGS: tuple[int, ...] = (3, 5, 10, 20, 50, 100, 200, 400)
TREND_HZZ_EMA_LAMBDA = 0.1
TREND_HZZ_MIN_NAMES = 30
TREND_SIGNALS_DAILY = "trend_signals_daily"
TREND_SIGNALS_REBALANCE = "trend_signals_rebalance"

BENCHMARK_SIGNAL_COLS: tuple[str, ...] = (
    "MOM",
    "REV1m_STR",
    "REV1w_WSTR",
    "TREND_HZZ",
    "DIST_52WH",
)
BENCHMARK_SIGNAL_DIRS: dict[str, str] = {
    "MOM": "mom",
    "REV1m_STR": "rev1m_str",
    "REV1w_WSTR": "rev1w_wstr",
    "TREND_HZZ": "trend_hzz",
    "DIST_52WH": "dist_52wh",
}

# Paper Table V–VIII characteristics (omit Bid-Ask, Price Delay).
CHAR_BETA_WINDOW = 252
CHAR_VOL_WINDOW = 21
CHAR_LIQUIDITY_WINDOW = 21
CHAR_LAG_WEEKLY_WINDOW = 5
MARKET_VW_RETURNS_FILE = "market_vw_returns.parquet"
CHARACTERISTICS_MONTH_END_FILE = "characteristics_month_end.parquet"
BENCHMARK_TABLES_ROOT = BACKTEST_BENCHMARK_ROOT / "tables_v_viii"

# Table V: 11 correlates (no LagWeeklyRet). VI/VII: +LagWeeklyRet.
TABLE_V_CHAR_COLS: tuple[str, ...] = (
    "MOM",
    "REV1m_STR",
    "REV1w_WSTR",
    "TREND_HZZ",
    "Beta",
    "Volatility",
    "DIST_52WH",
    "DollarVol",
    "ZeroTrade",
    "Size",
    "Illiquidity",
)
TABLE_VI_CHAR_COLS: tuple[str, ...] = TABLE_V_CHAR_COLS + ("LagWeeklyRet",)
TABLE_V_CHAR_DISPLAY: dict[str, str] = {
    "MOM": "MOM",
    "REV1m_STR": "STR",
    "REV1w_WSTR": "WSTR",
    "TREND_HZZ": "TREND",
    "Beta": "Beta",
    "Volatility": "Volat.",
    "DIST_52WH": "52WH",
    "DollarVol": "Dollar Volume",
    "ZeroTrade": "Zero Trade",
    "Size": "Size",
    "Illiquidity": "Illiq.",
    "LagWeeklyRet": "Lag Weekly Return",
}
LIQUIDITY_CHAR_COLS: tuple[str, ...] = (
    "Beta",
    "Volatility",
    "DollarVol",
    "ZeroTrade",
    "Illiquidity",
    "LagWeeklyRet",
)
HORIZON_DIAGONAL_IMAGE_DAYS: dict[int, int] = {5: 5, 20: 20, 60: 60}

# Rebalance folders for step-04 H1 tables (keyed by forecast horizon R).
HORIZON_BACKTEST_DIR: dict[int, str] = {
    5: "weekly",
    20: "monthly",
    60: "quarterly",
}


def benchmark_signals_dir(signal_col: str, market: str, horizon: int) -> Path:
    """Rebalance-date signal panels on DATA_ROOT (not repo outputs/)."""
    if signal_col not in BENCHMARK_SIGNAL_DIRS:
        raise ValueError(f"unsupported benchmark signal={signal_col}")
    if horizon not in HORIZON_BACKTEST_DIR:
        raise ValueError(f"unsupported horizon={horizon}")
    return (
        market_processed_dir(market)
        / TREND_SIGNALS_REBALANCE
        / BENCHMARK_SIGNAL_DIRS[signal_col]
        / HORIZON_BACKTEST_DIR[horizon]
    )


def benchmark_output_dir(signal_col: str, market: str, horizon: int) -> Path:
    """H1 backtest Excel under outputs/05_benchmark_signals/."""
    if signal_col not in BENCHMARK_SIGNAL_DIRS:
        raise ValueError(f"unsupported benchmark signal={signal_col}")
    if horizon not in HORIZON_BACKTEST_DIR:
        raise ValueError(f"unsupported horizon={horizon}")
    return (
        BACKTEST_BENCHMARK_ROOT
        / BENCHMARK_SIGNAL_DIRS[signal_col]
        / market
        / HORIZON_BACKTEST_DIR[horizon]
    )

CN_FACTOR_BACKTEST_DIR = "weekly"

BACKTEST_N_GROUP = 10
BACKTEST_WEIGHT_SCHEMES = ("equal", "float", "total")
# Eval horizons: Hk = signal at t, portfolio return at t+k rebalance periods (weekly → weeks).
EVAL_HORIZONS: tuple[int, ...] = (1, 2, 3)
# Rebalance / annualization keyed by forecast horizon R (not image window I).
BACKTEST_PERIODS_PER_YEAR = {5: 52, 20: 12, 60: 4}

SAMPLE_START = "1993-01-01"
SAMPLE_END = "2019-12-31"
TRAIN_END = "2000-12-31"
TEST_START = "2001-01-01"

# China A-share (dsf + allstk); modeling window ends 2019 like US.
CN_DSF_PATH = Path("/data/haifeng/Projects/cna/data/return/dsf.parquet.gzip")
CN_UNIV_PATH = Path("/data/haifeng/Projects/cna/data/univ/allstk/allstk.parquet.gzip")
CN_OHLC_HISTORY_START = "2000-01-01"


@dataclass(frozen=True)
class MarketSampleConfig:
    sample_start: str
    sample_end: str
    train_end: str
    test_start: str


MARKET_SAMPLE: dict[str, MarketSampleConfig] = {
    MARKET_US: MarketSampleConfig(
        sample_start=SAMPLE_START,
        sample_end=SAMPLE_END,
        train_end=TRAIN_END,
        test_start=TEST_START,
    ),
    MARKET_CN: MarketSampleConfig(
        sample_start="2007-01-01",
        sample_end="2019-12-31",
        train_end="2014-12-31",
        test_start="2015-01-01",
    ),
}


def market_sample_config(market: str) -> MarketSampleConfig:
    if market not in MARKET_SAMPLE:
        raise ValueError(f"unsupported market={market}")
    return MARKET_SAMPLE[market]

# CNN training (Jiang–Kelly–Xiu JF 2023, Section II.C)
N_ENSEMBLE = 5
TRAIN_VAL_SPLIT_SEED = 0  # fixed 70/30 split per Ix/Ry; ensemble seeds only affect optimization
ADAM_LR = 1e-5
BATCH_SIZE = 512
EARLY_STOP_PATIENCE = 2
TRAIN_VAL_FRAC = 0.7
MAX_EPOCHS = 100
# Independent (I, R, seed) processes packed onto remaining GPU memory.
GPU_MIN_FREE_GIB = 16.0
# Per-job VRAM reserve at BATCH_SIZE (scaled linearly for larger --batch-size).
VRAM_PER_JOB_GIB = {5: 2.0, 20: 4.0, 60: 8.0}
TRAIN_JOB_RAM_GIB = 10.0
# Per-GPU ceiling on concurrent train processes (VRAM scheduler may bind lower).
TRAIN_JOBS_PER_GPU_MAX = 32

WINDOW_DAYS = (5, 20, 60)
FUTURE_HORIZONS = (5, 20, 60)

TABLE_V_CNN_CONFIGS: tuple[tuple[int, int], ...] = tuple(
    (i, r) for i in WINDOW_DAYS for r in WINDOW_DAYS
)
TABLE_VI_CNN_CONFIGS: tuple[tuple[int, int], ...] = ((5, 5), (20, 5), (60, 5))
TABLE_VII_IMAGE_DAYS: tuple[int, ...] = WINDOW_DAYS
TABLE_VIII_IMAGE_DAYS = 5
TABLE_VIII_HORIZONS: tuple[int, ...] = WINDOW_DAYS


def benchmark_tables_output_dir(market: str) -> Path:
    return BENCHMARK_TABLES_ROOT / market


def characteristics_month_end_path(market: str) -> Path:
    return market_processed_dir(market) / CHARACTERISTICS_MONTH_END_FILE


def market_vw_returns_path(market: str) -> Path:
    return market_processed_dir(market) / MARKET_VW_RETURNS_FILE

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
