"""
detect_trend_structure
----------------------
Classifies market structure over the last N candles using
swing-point analysis: Higher Highs / Higher Lows (uptrend)
vs Lower Highs / Lower Lows (downtrend).

Used by app.py to produce a transparent, model-independent
"Trend structure" label shown alongside the XGBoost direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# ── pivot detection ────────────────────────────────────────────────────────────

def _find_pivots(highs: list[float], lows: list[float], order: int = 3):
    """
    Return lists of (index, price) for swing highs and swing lows.
    A swing high at i means highs[i] is the highest in [i-order, i+order].
    A swing low  at i means lows[i]  is the lowest  in [i-order, i+order].
    Only pivots fully surrounded by `order` bars on each side are returned.
    """
    n = len(highs)
    swing_highs: list[tuple[int, float]] = []
    swing_lows:  list[tuple[int, float]] = []

    for i in range(order, n - order):
        window_h = highs[i - order : i + order + 1]
        window_l = lows[i - order : i + order + 1]

        if highs[i] == max(window_h):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(window_l):
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


# ── structure scoring ──────────────────────────────────────────────────────────

@dataclass
class TrendStructureResult:
    label: str          # "Uptrend", "Downtrend", "Ranging", "Insufficient data"
    hh_count: int       # number of Higher Highs found
    hl_count: int       # number of Higher Lows found
    lh_count: int       # number of Lower Highs found
    ll_count: int       # number of Lower Lows found
    swing_highs: list[tuple[int, float]]
    swing_lows:  list[tuple[int, float]]
    strength: float     # 0.0–1.0  (how dominant the detected structure is)
    combined_direction: str  # UI-side merge with model direction

    def to_dict(self) -> dict:
        return {
            "label":              self.label,
            "hh_count":           self.hh_count,
            "hl_count":           self.hl_count,
            "lh_count":           self.lh_count,
            "ll_count":           self.ll_count,
            "strength":           round(self.strength, 3),
            "combined_direction": self.combined_direction,
        }


def detect_trend_structure(
    candles: list[dict],
    order: int = 3,
    model_direction: str = "Neutral",
) -> TrendStructureResult:
    """
    Analyse HH/HL vs LH/LL structure from a list of OHLC candle dicts
    (each dict must have keys: open, high, low, close).

    Parameters
    ----------
    candles : list[dict]
        Raw OHLC candles, oldest first. Typically load_last_n_candles(tf, 48).
    order : int
        Number of bars each side used to confirm a pivot (default 3).
    model_direction : str
        The raw label from XGBoost ("Bullish", "Neutral", etc.) used only
        for the UI-side combination rule.

    Returns
    -------
    TrendStructureResult
    """
    if len(candles) < 2 * order + 2:
        return TrendStructureResult(
            label="Insufficient data",
            hh_count=0, hl_count=0, lh_count=0, ll_count=0,
            swing_highs=[], swing_lows=[],
            strength=0.0,
            combined_direction=model_direction,
        )

    highs  = [c["high"]  for c in candles]
    lows   = [c["low"]   for c in candles]

    swing_highs, swing_lows = _find_pivots(highs, lows, order=order)

    # ── count HH / HL / LH / LL from consecutive pivot pairs ─────────────────
    hh = hl = lh = ll = 0

    for (i0, p0), (i1, p1) in zip(swing_highs, swing_highs[1:]):
        if p1 > p0:
            hh += 1
        elif p1 < p0:
            lh += 1

    for (i0, p0), (i1, p1) in zip(swing_lows, swing_lows[1:]):
        if p1 > p0:
            hl += 1
        elif p1 < p0:
            ll += 1

    bull_score = hh + hl       # bullish structure evidence
    bear_score = lh + ll       # bearish structure evidence
    total      = bull_score + bear_score

    if total == 0:
        label    = "Ranging"
        strength = 0.0
    elif bull_score > bear_score:
        label    = "Uptrend"
        strength = bull_score / total
    elif bear_score > bull_score:
        label    = "Downtrend"
        strength = bear_score / total
    else:
        label    = "Ranging"
        strength = 0.5

    # ── UI-side combination rule ───────────────────────────────────────────────
    # Model label stays visible; combined_direction is the enriched signal.
    combined = model_direction  # default: trust model

    if model_direction == "Neutral":
        if label == "Downtrend" and strength >= 0.60:
            combined = "Leaning Bearish"
        elif label == "Uptrend" and strength >= 0.60:
            combined = "Leaning Bullish"
        elif label == "Downtrend":
            combined = "Slightly Bearish"
        elif label == "Uptrend":
            combined = "Slightly Bullish"

    elif model_direction in ("SideBear", "Bearish"):
        if label == "Uptrend" and strength >= 0.65:
            combined = "Conflicted (Bear model / Bull structure)"
        # else keep model label

    elif model_direction in ("SideBull", "Bullish"):
        if label == "Downtrend" and strength >= 0.65:
            combined = "Conflicted (Bull model / Bear structure)"
        # else keep model label

    return TrendStructureResult(
        label=label,
        hh_count=hh, hl_count=hl, lh_count=lh, ll_count=ll,
        swing_highs=swing_highs, swing_lows=swing_lows,
        strength=strength,
        combined_direction=combined,
    )