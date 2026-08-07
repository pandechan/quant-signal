"""数据获取层：优先 Alpaca 实时数据，未配置时回退 yfinance。

环境变量：
  ALPACA_API_KEY     Alpaca API Key ID（配置后启用实时数据）
  ALPACA_SECRET_KEY  Alpaca Secret Key
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_klines(symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    """拉取指定股票的 K 线数据。

    优先使用 Alpaca（实时 IEX 数据），未配置 key 时回退 yfinance。

    Args:
        symbol: 股票代码，如 AAPL
        interval: K 线周期，支持 15m / 30m / 60m / 1h
        period: 回溯周期，如 5d / 30d

    Returns:
        包含 open/high/low/close/volume 列的 DataFrame，按时间升序。
    """
    if _alpaca_configured():
        try:
            df = _fetch_alpaca(symbol, interval, period)
            if not df.empty:
                return df
            logger.warning("Alpaca 返回空数据(%s)，回退 yfinance", symbol)
        except Exception as e:
            logger.warning("Alpaca 拉取失败(%s)，回退 yfinance: %s", symbol, e)
    return _fetch_yfinance(symbol, interval, period)


def _alpaca_configured() -> bool:
    return bool(os.environ.get("ALPACA_API_KEY")) and bool(
        os.environ.get("ALPACA_SECRET_KEY")
    )


def _fetch_alpaca(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """通过 Alpaca SDK 拉取实时 K 线（免费 tier 为 IEX 数据）。"""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    client = StockHistoricalDataClient(
        api_key=os.environ["ALPACA_API_KEY"],
        secret_key=os.environ["ALPACA_SECRET_KEY"],
    )

    tf = _to_timeframe(interval)
    days = _period_to_days(period)
    start = datetime.now(timezone.utc) - timedelta(days=days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=start,
    )
    bars = client.get_stock_bars(request)
    df = bars.df

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.columns = [str(c).lower() for c in df.columns]

    # 定位时间列并设为索引
    time_col = next(
        (c for c in ("timestamp", "time", "date") if c in df.columns), None
    )
    if time_col:
        df = df.set_index(time_col)

    df = df.sort_index()
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    return df[cols]


def _fetch_yfinance(symbol: str, interval: str, period: str) -> pd.DataFrame:
    """通过 yfinance 拉取 K 线（兜底数据源）。"""
    import yfinance as yf

    if interval in ("60m", "1h") and period == "5d":
        period = "30d"

    df = yf.download(
        symbol, interval=interval, period=period, progress=False, auto_adjust=True
    )
    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    return df


def _to_timeframe(interval: str):
    """将 interval 字符串转为 Alpaca TimeFrame。"""
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    if interval.endswith("m"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)
    if interval.endswith("h"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
    return TimeFrame(15, TimeFrameUnit.Minute)


def _period_to_days(period: str) -> int:
    """将 period 字符串转为天数。"""
    if period.endswith("d"):
        return int(period[:-1])
    if period.endswith("mo"):
        return int(period[:-2]) * 30
    return 5
