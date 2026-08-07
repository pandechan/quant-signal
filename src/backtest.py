"""回测模块：用历史数据验证策略表现。

评估指标：总交易数、胜率、平均盈亏、盈亏比(profit factor)、
总收益率、买入持有收益、最大回撤、夏普比率。

无前视偏差：一次性计算全量指标后，逐根 K 线取"截至该行"的数据生成信号
（ewm/rolling 每行值等价于截至该行的历史计算结果）。

用法：
  python src/backtest.py              # 默认天数（15m=60天, 60m=365天）
  python src/backtest.py --days 120   # 指定天数
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import fetch_klines
from features import add_indicators
from signals import RuleBasedGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("backtest")

WARMUP = 30  # 指标预热期，跳过前几根不稳定数据


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl_pct: float = 0.0
    is_win: bool = False


@dataclass
class BacktestResult:
    symbol: str
    name: str
    interval: str
    trades: list[Trade] = field(default_factory=list)
    total_return_pct: float = 0.0
    buy_hold_return_pct: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0


def run_backtest(
    symbol: str, name: str, interval: str, params: dict, days: int | None = None
) -> BacktestResult | None:
    """对单只股票运行回测。"""
    if days is None:
        days = 60 if interval == "15m" else 365
    period = f"{days}d"

    logger.info("回测 %s(%s) %s，拉取 %d 天数据", symbol, name, interval, days)
    try:
        df = fetch_klines(symbol, interval=interval, period=period)
    except Exception as e:
        logger.error("  %s 拉取数据失败: %s", symbol, e)
        return None

    if df.empty or len(df) < 50:
        logger.warning("  %s 数据不足(%d行)，跳过", symbol, len(df) if not df.empty else 0)
        return None

    # 一次性算好全部指标（无前视偏差）
    df = add_indicators(df, params)
    df = df.dropna()
    if len(df) < WARMUP + 10:
        logger.warning("  %s 有效数据不足，跳过", symbol)
        return None

    generator = RuleBasedGenerator()
    trades: list[Trade] = []
    position: dict | None = None  # None=空仓

    for i in range(WARMUP, len(df)):
        # 取截至 i 的切片生成信号（仅看最后一行，等价于截至该时刻）
        slice_df = df.iloc[: i + 1]
        sigs = generator.generate(symbol, name, interval, slice_df, params)
        if not sigs:
            continue

        sig = sigs[0]
        price = float(df.iloc[i]["close"])
        ts = df.index[i]

        if sig.direction == "BUY" and position is None:
            position = {"entry_time": ts, "entry_price": price}
        elif sig.direction == "SELL" and position is not None:
            pnl = (price - position["entry_price"]) / position["entry_price"] * 100
            trades.append(Trade(
                entry_time=position["entry_time"],
                entry_price=position["entry_price"],
                exit_time=ts,
                exit_price=price,
                pnl_pct=pnl,
                is_win=pnl > 0,
            ))
            position = None

    # 回测结束仍持仓则用最后收盘价平仓
    if position is not None:
        last_price = float(df.iloc[-1]["close"])
        pnl = (last_price - position["entry_price"]) / position["entry_price"] * 100
        trades.append(Trade(
            entry_time=position["entry_time"],
            entry_price=position["entry_price"],
            exit_time=df.index[-1],
            exit_price=last_price,
            pnl_pct=pnl,
            is_win=pnl > 0,
        ))

    return _calc_metrics(symbol, name, interval, trades, df)


def _calc_metrics(symbol, name, interval, trades, df) -> BacktestResult:
    r = BacktestResult(symbol=symbol, name=name, interval=interval, trades=trades)
    r.total_trades = len(trades)

    first_price = float(df.iloc[0]["close"])
    last_price = float(df.iloc[-1]["close"])
    r.buy_hold_return_pct = (last_price - first_price) / first_price * 100

    if not trades:
        r.total_return_pct = 0.0
        return r

    # 策略总收益（复利累计）
    equity = 1.0
    equity_curve = [1.0]
    for t in trades:
        equity *= 1 + t.pnl_pct / 100
        equity_curve.append(equity)
    r.total_return_pct = (equity - 1) * 100

    wins = [t for t in trades if t.is_win]
    losses = [t for t in trades if not t.is_win]
    r.win_rate = len(wins) / len(trades) * 100
    r.avg_win_pct = float(np.mean([t.pnl_pct for t in wins])) if wins else 0.0
    r.avg_loss_pct = float(np.mean([t.pnl_pct for t in losses])) if losses else 0.0

    gross_profit = sum(t.pnl_pct for t in wins)
    gross_loss = abs(sum(t.pnl_pct for t in losses))
    r.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # 最大回撤
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd
    r.max_drawdown_pct = max_dd

    # 夏普比率（用每笔交易收益，按年化交易频率缩放）
    rets = [t.pnl_pct / 100 for t in trades]
    if len(rets) > 1 and np.std(rets) > 0:
        if trades[-1].exit_time and trades[0].entry_time:
            span_days = max((trades[-1].exit_time - trades[0].entry_time).days, 1)
            trades_per_year = len(trades) / span_days * 365
        else:
            trades_per_year = 50
        r.sharpe_ratio = float(
            np.mean(rets) / np.std(rets) * math.sqrt(trades_per_year)
        )

    return r


def _print_results(results: list[BacktestResult]) -> None:
    if not results:
        print("\n无回测结果\n")
        return

    print("\n" + "=" * 95)
    print(f"{'Symbol':<8} {'Intv':<5} {'Trades':<7} {'WinRate':<8} {'Return':<9} "
          f"{'BuyHold':<9} {'PF':<6} {'MaxDD':<8} {'Sharpe':<7}")
    print("-" * 95)
    for r in results:
        pf = "inf" if math.isinf(r.profit_factor) else f"{r.profit_factor:.2f}"
        print(f"{r.symbol:<8} {r.interval:<5} {r.total_trades:<7} {r.win_rate:>6.1f}%  "
              f"{r.total_return_pct:>+7.1f}%  {r.buy_hold_return_pct:>+7.1f}%  "
              f"{pf:>5}  {r.max_drawdown_pct:>+7.1f}%  {r.sharpe_ratio:>5.2f}")
    print("=" * 95)

    avg_win = np.mean([r.win_rate for r in results])
    avg_ret = np.mean([r.total_return_pct for r in results])
    avg_bh = np.mean([r.buy_hold_return_pct for r in results])
    beat = "跑赢" if avg_ret > avg_bh else "跑输"
    print(f"\n汇总: 平均胜率 {avg_win:.1f}% | 策略平均收益 {avg_ret:+.1f}% "
          f"| 买入持有平均 {avg_bh:+.1f}% | 策略{beat}基准 {abs(avg_ret - avg_bh):.1f}%\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="策略回测")
    parser.add_argument("--days", type=int, default=None,
                        help="回测天数（不指定则 15m=60天, 60m=365天）")
    args = parser.parse_args()

    with open(ROOT / "config" / "watchlist.yaml", encoding="utf-8") as f:
        wl = yaml.safe_load(f)
    with open(ROOT / "config" / "strategy.yaml", encoding="utf-8") as f:
        st = yaml.safe_load(f)
    params = st.get("strategy", {})

    results: list[BacktestResult] = []
    for item in wl.get("watchlist", []):
        r = run_backtest(
            item["symbol"], item.get("name", item["symbol"]),
            item.get("interval", "15m"), params, args.days,
        )
        if r:
            results.append(r)

    _print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
