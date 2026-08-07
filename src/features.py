"""技术指标计算层：纯 pandas/numpy 实现，无外部指标库依赖。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """为 K 线数据附加技术指标列。

    列名与原 pandas-ta 命名保持一致，signals.py 无需改动。

    Args:
        df: 原始 K 线数据，需含 open/high/low/close/volume
        params: 指标参数，来自 strategy.yaml

    Returns:
        附加指标列后的 DataFrame。
    """
    if df.empty:
        return df

    p = params or {}
    ema_fast = p.get("ema_fast", 12)
    ema_slow = p.get("ema_slow", 26)
    rsi_period = p.get("rsi_period", 14)
    bbands_period = p.get("bbands_period", 20)
    stoch_period = p.get("stoch_period", 14)
    vol_ma_period = p.get("vol_ma_period", 20)

    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out["volume"]

    # ---- EMA 快慢线 ----
    out[f"EMA_{ema_fast}"] = close.ewm(span=ema_fast, adjust=False).mean()
    out[f"EMA_{ema_slow}"] = close.ewm(span=ema_slow, adjust=False).mean()

    # ---- MACD ----
    macd = out[f"EMA_{ema_fast}"] - out[f"EMA_{ema_slow}"]
    signal = macd.ewm(span=9, adjust=False).mean()
    out["MACD_12_26_9"] = macd
    out["MACDs_12_26_9"] = signal
    out["MACDh_12_26_9"] = macd - signal

    # ---- RSI（Wilder 平滑）----
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss
    out[f"RSI_{rsi_period}"] = 100 - (100 / (1 + rs))

    # ---- KDJ（随机指标）----
    lowest_low = low.rolling(stoch_period).min()
    highest_high = high.rolling(stoch_period).max()
    stoch_range = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / stoch_range
    out[f"STOCHk_{stoch_period}_3_3"] = k
    out[f"STOCHd_{stoch_period}_3_3"] = k.rolling(3).mean()

    # ---- 布林带 ----
    bb_sma = close.rolling(bbands_period).mean()
    bb_std = close.rolling(bbands_period).std()
    out[f"BBU_{bbands_period}_2.0"] = bb_sma + 2 * bb_std
    out[f"BBM_{bbands_period}_2.0"] = bb_sma
    out[f"BBL_{bbands_period}_2.0"] = bb_sma - 2 * bb_std

    # ---- ADX（趋势强度）----
    out["ADX_14"] = _adx(high, low, close, 14)

    # ---- 量比 ----
    out["vol_ma"] = volume.rolling(vol_ma_period).mean()
    out["vol_ratio"] = volume / out["vol_ma"].replace(0, np.nan)

    return out


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算 ADX（平均趋向指数）。"""
    prev_close = close.shift(1)

    # True Range
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    # 方向移动
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    # Wilder 平滑
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr

    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx
