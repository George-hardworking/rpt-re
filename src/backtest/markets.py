"""Market column maps for US CRSP and China A-share backtests.

US CRSP daily files in this repo have one cap field (DlyCap → MarketCap).
Both value-weight schemes use that field until a free-float series is added.
China panels must carry formation-date 流通市值 and 总市值 as separate columns.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import BACKTEST_PERIODS_PER_YEAR


@dataclass(frozen=True)
class MarketSpec:
    name: str
    id_col: str
    date_col: str
    ret_col: str
    float_cap_col: str
    total_cap_col: str
    periods_per_year: int

    def cap_col(self, scheme: str) -> str | None:
        if scheme == "equal":
            return None
        if scheme == "float":
            return self.float_cap_col
        if scheme == "total":
            return self.total_cap_col
        raise ValueError(f"unknown weight scheme: {scheme}")


def us_spec(image_days: int, horizon: int) -> MarketSpec:
    return MarketSpec(
        name="us",
        id_col="PERMNO",
        date_col="Date",
        ret_col=f"Ret_{horizon}d",
        float_cap_col="MarketCap",
        total_cap_col="MarketCap",
        periods_per_year=BACKTEST_PERIODS_PER_YEAR[image_days],
    )


CN_SPEC = MarketSpec(
    name="cn",
    id_col="SecuCode",
    date_col="date",
    ret_col="ret",
    float_cap_col="FloatCap",
    total_cap_col="TotalCap",
    periods_per_year=52,
)
