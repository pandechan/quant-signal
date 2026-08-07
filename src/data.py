"""数据获取层：通过 yfinance 拉取美股 K 线数据。"""

import yfinance as yf
import pandas as pd


def fetch_klines(symbol: str, interval: str = "15m", period: str = "5d") -> pd.DataFrame:
    """拉取指定股票的 K 线数据。

    Args:
        symbol: Yahoo Finance 代码，如 AAPL
        interval: K 线周期，支持 15m / 30m / 60m / 1d 等
        period: 回溯周期，15m 最多 60d，60m 最多 730d

    Returns:
        包含 open/high/low/close/volume 列的 DataFrame，按时间升序。
    """
    # 60m 周期允许拉更长历史，便于指标计算
    if interval in ("60m", "1h") and period == "5d":
        period = "30d"

    df = yf.download(
        symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=True,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    # 新版 yfinance 对单股票也返回 MultiIndex 列，拍平
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    return df
