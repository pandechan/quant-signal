"""回测模块：用历史数据验证策略表现。

支持规则引擎和 ML 生成器，ATR 止损/止盈/trailing、交易成本、趋势过滤。
ML 模式通过 predict_series 预算概率序列，高效回测。

用法：
  python src/backtest.py                          # 规则引擎回测
  python src/backtest.py --generator ml           # ML 回测（需先训练或已有模型）
  python src/backtest.py --generator ml --days 120
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
from signals import RuleBasedGenerator, get_generator

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
    generator=None, start_index: int = 0,
) -> BacktestResult | None:
    """对单只股票运行回测。

    generator: 信号生成器，None 则用 RuleBasedGenerator。
    start_index: 从该行开始交易（ML 回测用，训练段提供指标历史）。
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

    if "ATR_14" not in df.columns:
        df = add_indicators(df, params)
    df = df.dropna()
    if len(df) < WARMUP + 10:
        logger.warning("  %s 有效数据不足，跳过", symbol)
        return None

    if generator is None:
        generator = RuleBasedGenerator()

    # 交易参数
    stop_loss_atr = params.get("stop_loss_atr", 3.0)
    take_profit_atr = params.get("take_profit_atr", 4.0)
    trail_atr = params.get("trail_atr", 3.0)
    cost = params.get("commission_pct", 0.0005) + params.get("slippage_pct", 0.0005)
    use_trend_filter = params.get("use_trend_filter", True)
    trend_col = f"EMA_{params.get('trend_ema', 50)}"
    buy_threshold = params.get("ml_buy_threshold", 0.6)
    atr_col = "ATR_14"

    # ML 预算概率序列
    probs = None
    if hasattr(generator, "predict_series"):
        probs = generator.predict_series(df, params)

    trades: list[Trade] = []
    position: dict | None = None
    start = max(WARMUP, start_index)

    for i in range(start, len(df)):
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
                    entry_time=position["entry_time"], entry_price=entry_price,
                    exit_time=df.index[i], exit_price=exit_price,
                    pnl_pct=pnl, is_win=pnl > 0, exit_reason=exit_reason,
                ))
                position = None
            continue

        # ---- 空仓：判断进场 ----
        should_enter = False
        if probs is not None:
            # ML 模式
            prob = probs.iloc[i]
            trend_ok = True
            if use_trend_filter and trend_col in df.columns and pd.notna(row.get(trend_col)):
                trend_ok = row["close"] > row[trend_col]
            should_enter = prob > buy_threshold and trend_ok
        else:
            # 规则模式
            slice_df = df.iloc[: i + 1]
            sigs = generator.generate(symbol, name, interval, slice_df, params)
            should_enter = bool(sigs and sigs[0].direction == "BUY")

        if should_enter:
            entry_price = price * (1 + cost)
            if atr_val:
                stop = entry_price - stop_loss_atr * atr_val
                target = entry_price + take_profit_atr * atr_val
            else:
                stop = entry_price * 0.97
                target = entry_price * 1.06
            position = {
                "entry_time": df.index[i], "entry_price": entry_price,
                "stop_loss": stop, "take_profit": target, "highest": entry_price,
            }

    # 期末平仓
    if position is not None:
        last_price = float(df.iloc[-1]["close"])
        exit_price = last_price * (1 - cost)
        pnl = (exit_price - position["entry_price"]) / position["entry_price"] * 100
        trades.append(Trade(
            entry_time=position["entry_time"], entry_price=position["entry_price"],
            exit_time=df.index[-1], exit_price=exit_price,
            pnl_pct=pnl, is_win=pnl > 0, exit_reason="期末平仓",
        ))

    return _calc_metrics(symbol, name, interval, trades, df, start_index)


def _calc_metrics(symbol, name, interval, trades, df, start_index=0) -> BacktestResult:
    r = BacktestResult(symbol=symbol, name=name, interval=interval, trades=trades)
    r.total_trades = len(trades)

    start = max(start_index, 0)
    first_price = float(df.iloc[start]["close"])
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
    parser.add_argument("--generator", default="rule", choices=["rule", "ml"],
                        help="信号生成器：rule 或 ml")
    args = parser.parse_args()

    with open(ROOT / "config" / "watchlist.yaml", encoding="utf-8") as f:
        wl = yaml.safe_load(f)
    with open(ROOT / "config" / "strategy.yaml", encoding="utf-8") as f:
        st = yaml.safe_load(f)
    params = st.get("strategy", {})

    generator = get_generator(args.generator)

    results: list[BacktestResult] = []
    for item in wl.get("watchlist", []):
        r = run_backtest(
            item["symbol"], item.get("name", item["symbol"]),
            item.get("interval", "15m"), params, args.days, generator=generator,
        )
        if r:
            results.append(r)

    _print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
