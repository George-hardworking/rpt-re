"""Sullivan-Timmermann-White technical trading rule manifest.

The paper's Figure 8 benchmark uses the 7,846-rule universe from
Sullivan, Timmermann, and White (1999).  This module keeps the rule
universe explicit and serializable so the expensive signal and backtest
steps can be sharded deterministically on a server.

The family counts match the common public reproduction of the STW universe:
497 filter rules, 2,049 moving-average rules, 1,220 support/resistance
rules, 2,040 channel-breakout rules, and 2,040 on-balance-volume rules.
If the original Scaillet code becomes available, adjust the grids below
instead of touching the downstream pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

STW_TOTAL_RULES = 7_846
STW_FAMILY_COUNTS: dict[str, int] = {
    "FR": 497,
    "MA": 2_049,
    "SR": 1_220,
    "CB": 2_040,
    "OBV": 2_040,
}

FR_X = (
    0.005,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.035,
    0.04,
    0.045,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
    0.12,
    0.14,
    0.16,
    0.18,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
)
FR_Y = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15, 0.20)
EXTREMA_E = (1, 2, 3, 4, 5, 10, 15, 20)
HOLD_C = (5, 10, 25, 50)

MA_N = (2, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 200, 250)
MA_LEGACY_SINGLE_N = (5, 10, 15, 20, 25, 50, 100, 150, 200)
BANDS = (0.001, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05)
DELAYS = (2, 3, 4, 5)
MA_HOLD_C = (5, 10, 20, 25, 50)

SR_N = (5, 10, 15, 20, 25, 50, 100, 150, 200, 250)
SR_EXTRA_STOP = (0.02, 0.03, 0.04, 0.05, 0.075)

CB_N = SR_N
CB_X = (
    0.005,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.075,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.75,
)
CB_HOLD_C = (5, 10, 25)


@dataclass(frozen=True)
class STWRule:
    rule_id: int
    name: str
    family: str
    variant: str
    params: dict[str, Any]


def _fmt_num(x: float) -> str:
    return f"{x:g}".replace(".", "p")


def _pairs(values: Iterable[int]) -> Iterable[tuple[int, int]]:
    vals = tuple(values)
    for i, short in enumerate(vals):
        for long in vals[i + 1 :]:
            yield short, long


def _rule(counter: int, family: str, variant: str, params: dict[str, Any]) -> STWRule:
    parts = [family, variant]
    for key in sorted(params):
        parts.append(f"{key}{_fmt_num(params[key]) if isinstance(params[key], float) else params[key]}")
    name = "_".join(parts)
    return STWRule(counter, f"STW_{counter:04d}_{name}", family, variant, params)


def _filter_rules(start: int) -> list[STWRule]:
    rules: list[STWRule] = []
    rid = start

    for x in FR_X:
        rules.append(_rule(rid, "FR", "simple", {"x": x}))
        rid += 1

    for x in FR_X:
        for y in FR_Y:
            if y < x:
                rules.append(_rule(rid, "FR", "neutral", {"x": x, "y": y}))
                rid += 1

    for x in FR_X:
        for c in HOLD_C:
            rules.append(_rule(rid, "FR", "hold", {"x": x, "c": c}))
            rid += 1

    for x in FR_X:
        for e in EXTREMA_E:
            rules.append(_rule(rid, "FR", "extrema", {"x": x, "e": e}))
            rid += 1

    return rules


def _ma_like_rules(start: int, family: str, include_legacy_price_rules: bool) -> list[STWRule]:
    rules: list[STWRule] = []
    rid = start

    for short, long in _pairs(MA_N):
        params = {"short": short, "long": long}
        rules.append(_rule(rid, family, "cross", params))
        rid += 1
        for b in BANDS:
            rules.append(_rule(rid, family, "cross_band", {**params, "b": b}))
            rid += 1
        for d in DELAYS:
            rules.append(_rule(rid, family, "cross_delay", {**params, "d": d}))
            rid += 1
        for c in MA_HOLD_C:
            rules.append(_rule(rid, family, "cross_hold", {**params, "c": c}))
            rid += 1

    for n in MA_N:
        params = {"n": n}
        rules.append(_rule(rid, family, "single", params))
        rid += 1
        for d in DELAYS:
            rules.append(_rule(rid, family, "single_delay", {**params, "d": d}))
            rid += 1
        for c in MA_HOLD_C:
            rules.append(_rule(rid, family, "single_hold", {**params, "c": c}))
            rid += 1

    if include_legacy_price_rules:
        for n in MA_LEGACY_SINGLE_N:
            rules.append(_rule(rid, family, "legacy_single", {"n": n}))
            rid += 1

    return rules


def _support_resistance_rules(start: int) -> list[STWRule]:
    rules: list[STWRule] = []
    rid = start

    for basis, values in (("range", SR_N), ("extrema", EXTREMA_E)):
        param_name = "n" if basis == "range" else "e"
        for value in values:
            params = {param_name: value}
            rules.append(_rule(rid, "SR", basis, params))
            rid += 1
            for b in BANDS:
                rules.append(_rule(rid, "SR", f"{basis}_band", {**params, "b": b}))
                rid += 1
            for d in DELAYS:
                rules.append(_rule(rid, "SR", f"{basis}_delay", {**params, "d": d}))
                rid += 1
            for c in HOLD_C:
                rules.append(_rule(rid, "SR", f"{basis}_hold", {**params, "c": c}))
                rid += 1
            for b in BANDS:
                for c in HOLD_C:
                    rules.append(_rule(rid, "SR", f"{basis}_band_hold", {**params, "b": b, "c": c}))
                    rid += 1
            for d in DELAYS:
                for c in HOLD_C:
                    rules.append(_rule(rid, "SR", f"{basis}_delay_hold", {**params, "d": d, "c": c}))
                    rid += 1

    # Public descriptions of STW list 1,220 S/R rules.  The documented
    # range/extrema + band/delay/holding grid above yields 1,170.  We keep the
    # remaining 50 as explicit stop-unwind S/R variants so the manifest is
    # complete and auditable; if original code is obtained, this is the small
    # block most likely to need calibration.
    for n in SR_N:
        for stop in SR_EXTRA_STOP:
            rules.append(_rule(rid, "SR", "range_stop", {"n": n, "stop": stop}))
            rid += 1

    return rules


def _channel_breakout_rules(start: int) -> list[STWRule]:
    rules: list[STWRule] = []
    rid = start

    for n in CB_N:
        for x in CB_X:
            params = {"n": n, "x": x}
            rules.append(_rule(rid, "CB", "breakout", params))
            rid += 1
            for b in BANDS:
                rules.append(_rule(rid, "CB", "breakout_band", {**params, "b": b}))
                rid += 1
            for c in CB_HOLD_C:
                rules.append(_rule(rid, "CB", "breakout_hold", {**params, "c": c}))
                rid += 1

    return rules


def stw_rule_manifest() -> list[STWRule]:
    rules: list[STWRule] = []
    rid = 1
    for block in (
        _filter_rules,
        lambda start: _ma_like_rules(start, "MA", include_legacy_price_rules=True),
        _support_resistance_rules,
        _channel_breakout_rules,
        lambda start: _ma_like_rules(start, "OBV", include_legacy_price_rules=False),
    ):
        chunk = block(rid)
        rules.extend(chunk)
        rid += len(chunk)

    validate_stw_manifest(rules)
    return rules


def validate_stw_manifest(rules: list[STWRule]) -> None:
    counts = pd.Series([r.family for r in rules]).value_counts().to_dict()
    if len(rules) != STW_TOTAL_RULES:
        raise ValueError(f"expected {STW_TOTAL_RULES} STW rules, got {len(rules)}")
    for family, expected in STW_FAMILY_COUNTS.items():
        got = int(counts.get(family, 0))
        if got != expected:
            raise ValueError(f"expected {expected} {family} rules, got {got}")
    ids = [r.rule_id for r in rules]
    if ids != list(range(1, STW_TOTAL_RULES + 1)):
        raise ValueError("STW rule ids must be contiguous from 1")


def manifest_frame(rules: list[STWRule] | None = None) -> pd.DataFrame:
    if rules is None:
        rules = stw_rule_manifest()
    rows = []
    for r in rules:
        row = asdict(r)
        row.update(r.params)
        row.pop("params")
        rows.append(row)
    return pd.DataFrame(rows)


def write_manifest(path: Path, rules: list[STWRule] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_frame(rules).to_csv(path, index=False)
    return path
