"""Paper Table III: split monthly R20 holding returns into days 1–5 vs 6–20.

Formation date, ranks, and weights stay the same as the monthly backtest.
Only the measured return window changes. Annualization stays monthly (12)
so mean H-L returns in the two windows add (approximately) to the full-month table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    HOLDING_BREAKDOWN_FIRST_DAYS,
    HOLDING_BREAKDOWN_HORIZON,
    RET_D1_D5,
    RET_D6_D20,
)


def attach_r20_subperiod_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Add Ret_d1_d5 and Ret_d6_d20 from PIT forward labels on the same row.

    Ret_5d  = days 1–5 after formation (t+1 through t+5).
    Ret_20d = days 1–20 after formation (t+1 through t+20).
    Days 6–20 is the residual: (1+Ret_20d)/(1+Ret_5d) − 1.
    """
    r5_col = f"Ret_{HOLDING_BREAKDOWN_FIRST_DAYS}d"
    r20_col = f"Ret_{HOLDING_BREAKDOWN_HORIZON}d"
    missing = [c for c in (r5_col, r20_col) if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing {missing} for R20 subperiod split")
    out = panel.copy()
    r5 = out[r5_col].to_numpy(dtype=np.float64)
    r20 = out[r20_col].to_numpy(dtype=np.float64)
    if np.any(np.isfinite(r5) & (r5 <= -1.0)):
        n_bad = int(np.sum(np.isfinite(r5) & (r5 <= -1.0)))
        raise ValueError(f"{n_bad} rows have {r5_col} <= -1; cannot form days 6–20 residual")
    out[RET_D1_D5] = r5
    out[RET_D6_D20] = (1.0 + r20) / (1.0 + r5) - 1.0
    return out
