"""回测模块：用历史数据验证策略表现。

支持：ATR 止损/止盈/trailing stop、交易成本模拟、趋势过滤。
无前视偏差：一次性计算全量指标后逐根 K 线回放。

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

WARMUP = 30


@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    exit_time: datetime | None = None
    exit_price: float | None = None
    pnl_pct: float = 0.0
    is_win: bool = False
    exit_reason: str = ""


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
    symbol: str, name: str, interval: str, params: dict,
    days: int | None = None, df_override: pd.DataFrame | None = None,
) -> BacktestResult | None:
    """对单只股票运行回测。

    df_override 传入已加载的 DataFrame；若已含指标列则跳过 add_indicators（优化器复用）。
    """
    if df_override is not None:
        df = df_override
    else:
        if days is None:
            days = 60 if interval == "15m" else 365
        logger.info("回测 %s(%s) %s，拉取 %d 天数据", symbol, name, interval, days)
        try:
            df = fetch_klines(symbol, interval=interval, period=f"{days}d")
        except Exception as e:
            logger.error("  %s 拉取数据失败: %s", symbol, e)
            return None

    if df is None or df.empty or len(df) < 50:
        logger.warning("  %s 数据不足，跳过", symbol)
        return None

    # 若未含指标列则计算（优化器传入已算好的可跳过）
    if "ATR_14" not in df.columns:
        df = add_indicators(df, params)
    df = df.dropna()
    if len(df) < WARMUP + 10:
        logger.warning("  %s 有效数据不足，跳过", symbol)
        return None

    # 交易参数
    stop_loss_atr = params.get("stop_loss_atr", 2.0)
    take_profit_atr = params.get("take_profit_atr", 3.0)
    trail_atr = params.get("trail_atr", 2.5)
    cost = params.get("commission_pct", 0.0005) + params.get("slippage_pct", 0.0005)
    atr_col = "ATR_14"

    generator = RuleBasedGenerator()
    trades: list[Trade] = []
    position: dict | None = None

    for i in range(WARMUP, len(df)):
        row = df.iloc[i]
        price = float(row["close"])
        atr_val = float(row[atr_col]) if atr_col in df.columns and pd.notna(row.get(atr_col)) else None

        # ---- 持仓中：检查止损/止盈/trailing ----
        if position is not None:
            if price > position["highest"]:
                position["highest"] = price

            exit_reason = None
            if price <= position["stop_loss"]:
                exit_reason = "止损"
            elif price >= position["take_profit"]:
                exit_reason = "止盈"
            elif atr_val and price <= position["highest"] - trail_atr * atr_val:
                exit_reason = "trailing"

            if exit_reason:
                exit_price = price * (1 - cost)
                entry_price = position["entry_price"]
                pnl = (exit_price - entry_price) / entry_price * 100
                trades.append(Trade(
                    entry_time=position["entry_time"],
                    entry_price=entry_price,
                    exit_time=df.index[i],
                    exit_price=exit_price,
                    pnl_pct=pnl,
                    is_win=pnl > 0,
                    exit_reason=exit_reason,
                ))
                position = None
            continue

        # ---- 空仓：看 BUY 信号进场 ----
        slice_df = df.iloc[: i + 1]
        sigs = generator.generate(symbol, name, interval, slice_df, params)
        if sigs and sigs[0].direction == "BUY":
            entry_price = price * (1 + cost)
            if atr_val:
                stop = entry_price - stop_loss_atr * atr_val
                target = entry_price + take_profit_atr * atr_val
            else:
                stop = entry_price * 0.97
                target = entry_price * 1.06
            position = {
                "entry_time": df.index[i],
                "entry_price": entry_price,
                "stop_loss": stop,
                "take_profit": target,
                "highest": entry_price,
            }

    # 回测结束仍持仓则用最后收盘价平仓
    if position is not None:
        last_price = float(df.iloc[-1]["close"])
        exit_price = last_price * (1 - cost)
        pnl = (exit_price - position["entry_price"]) / position["entry_price"] * 100
        trades.append(Trade(
            entry_time=position["entry_time"],
            entry_price=position["entry_price"],
            exit_time=df.index[-1],
            exit_price=exit_price,
            pnl_pct=pnl,
            is_win=pnl > 0,
            exit_reason="期末平仓",
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

    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak * 100
        if dd > max_dd:
            max_dd = dd
    r.max_drawdown_pct = max_dd

    rets = [t.pnl_pct / 100 for t in trades]
    if len(rets) > 1 and np.std(rets) > 0:
        if trades[-1].exit_time and trades[0].entry_time:
            span_days = max((trades[-1].exit_time - trades[0].entry_time).days, 1)
            trades_per_year = len(trades) / span_days * 365
        else:
            trades_per_year = 50
        r.sharpe_ratio = float(np.mean(rets) / np.std(rets) * math.sqrt(trades_per_year))

    return r


def _print_results(results: list[BacktestResult]) -> None:
    if not results:
        print("\n无回测结果\n")
        return

    print("\n" + "=" * 100)
    print(f"{'Symbol':<8} {'Intv':<5} {'Trades':<7} {'WinRate':<8} {'Return':<9} "
          f"{'BuyHold':<9} {'PF':<6} {'MaxDD':<8} {'Sharpe':<7}")
    print("-" * 100)
    for r in results:
        pf = "inf" if math.isinf(r.profit_factor) else f"{r.profit_factor:.2f}"
        print(f"{r.symbol:<8} {r.interval:<5} {r.total_trades:<7} {r.win_rate:>6.1f}%  "
              f"{r.total_return_pct:>+7.1f}%  {r.buy_hold_return_pct:>+7.1f}%  "
              f"{pf:>5}  {r.max_drawdown_pct:>+7.1f}%  {r.sharpe_ratio:>5.2f}")
    print("=" * 100)

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
