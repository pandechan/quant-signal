"""技术指标计算层：基于 pandas-ta 计算常用技术指标。"""

import pandas as pd
import pandas_ta as ta


def add_indicators(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """为 K 线数据附加技术指标列。

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

    # 趋势：快慢 EMA
    out.ta.ema(length=ema_fast, append=True)
    out.ta.ema(length=ema_slow, append=True)

    # 动量：MACD + RSI + KDJ
    out.ta.macd(fast=ema_fast, slow=ema_slow, append=True)
    out.ta.rsi(length=rsi_period, append=True)
    out.ta.stoch(length=stoch_period, append=True)

    # 波动：布林带
    out.ta.bbands(length=bbands_period, append=True)

    # 趋势强度：ADX
    out.ta.adx(length=14, append=True)

    # 量能比：当前成交量 / N 周期均量
    vol_col = _find_col(out, "volume")
    if vol_col:
        out["vol_ma"] = out[vol_col].rolling(vol_ma_period).mean()
        out["vol_ratio"] = out[vol_col] / out["vol_ma"]

    return out


def _find_col(df: pd.DataFrame, name: str) -> str | None:
    """大小写不敏感查找列名。"""
    for c in df.columns:
        if c.lower() == name.lower():
            return c
    return None
