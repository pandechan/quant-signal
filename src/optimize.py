"""参数优化模块：网格搜索止损/止盈/trailing 最优组合。

优化目标：跨股票平均夏普比率最大化。
预加载数据 + 预算指标一次，回测复用，避免重复网络请求和计算。

用法：
  python src/optimize.py              # 默认天数
  python src/optimize.py --days 120   # 指定天数
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import fetch_klines
from features import add_indicators
from backtest import run_backtest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("optimize")

# 网格搜索参数空间（止损/止盈/trailing 是影响最大的交易管理参数）
PARAM_GRID = {
    "stop_loss_atr": [1.5, 2.0, 3.0],
    "take_profit_atr": [2.0, 3.0, 4.0],
    "trail_atr": [2.0, 2.5, 3.0],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="参数优化")
    parser.add_argument("--days", type=int, default=None,
                        help="回测天数（不指定则 15m=60天, 60m=365天）")
    args = parser.parse_args()

    with open(ROOT / "config" / "watchlist.yaml", encoding="utf-8") as f:
        wl = yaml.safe_load(f)
    with open(ROOT / "config" / "strategy.yaml", encoding="utf-8") as f:
        st = yaml.safe_load(f)
    base_params = st.get("strategy", {})

    watchlist = wl.get("watchlist", [])

    # ---- 预加载所有股票数据 + 预算指标（只算一次，复用）----
    logger.info("预加载 %d 只股票数据...", len(watchlist))
    data: dict[str, tuple[dict, object]] = {}
    for item in watchlist:
        symbol = item["symbol"]
        interval = item.get("interval", "15m")
        days = args.days or (60 if interval == "15m" else 365)
        try:
            df = fetch_klines(symbol, interval=interval, period=f"{days}d")
            if df is not None and not df.empty:
                df = add_indicators(df, base_params)
                data[symbol] = (item, df)
                logger.info("  %s 加载完成 (%d 行)", symbol, len(df))
        except Exception as e:
            logger.warning("  %s 加载失败: %s", symbol, e)

    if not data:
        print("无可用数据")
        return 1

    # ---- 网格搜索 ----
    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    logger.info("开始网格搜索: %d 组组合 x %d 只股票 = %d 次回测",
                len(combos), len(data), len(combos) * len(data))

    results = []
    for idx, combo in enumerate(combos):
        params = base_params.copy()
        params.update(dict(zip(keys, combo)))

        bt_results = []
        for symbol, (item, df) in data.items():
            r = run_backtest(
                symbol, item.get("name", symbol),
                item.get("interval", "15m"), params, df_override=df,
            )
            if r:
                bt_results.append(r)

        if not bt_results:
            continue

        avg_sharpe = float(np.mean([r.sharpe_ratio for r in bt_results]))
        avg_return = float(np.mean([r.total_return_pct for r in bt_results]))
        avg_win = float(np.mean([r.win_rate for r in bt_results]))
        avg_dd = float(np.mean([r.max_drawdown_pct for r in bt_results]))
        total_trades = sum(r.total_trades for r in bt_results)

        results.append({
            "combo": dict(zip(keys, combo)),
            "avg_sharpe": avg_sharpe,
            "avg_return": avg_return,
            "avg_win": avg_win,
            "avg_dd": avg_dd,
            "total_trades": total_trades,
        })

        if (idx + 1) % 9 == 0:
            logger.info("  进度 %d/%d", idx + 1, len(combos))

    # ---- 按夏普排序，输出 Top 5 ----
    results.sort(key=lambda x: x["avg_sharpe"], reverse=True)

    print("\n" + "=" * 95)
    print(f"{'Rank':<5} {'StopLoss':<9} {'TakeProfit':<11} {'Trail':<6} "
          f"{'AvgSharpe':<10} {'AvgReturn':<10} {'AvgWin':<8} {'AvgMaxDD':<9} {'Trades':<7}")
    print("-" * 95)
    for rank, r in enumerate(results[:5], 1):
        c = r["combo"]
        print(f"{rank:<5} {c['stop_loss_atr']:<9} {c['take_profit_atr']:<11} "
              f"{c['trail_atr']:<6} {r['avg_sharpe']:>+8.2f}  {r['avg_return']:>+8.1f}%  "
              f"{r['avg_win']:>6.1f}%  {r['avg_dd']:>+7.1f}%  {r['total_trades']:<7}")
    print("=" * 95)

    if results:
        best = results[0]["combo"]
        print(f"\n最优参数（夏普最高）:")
        print(f"  stop_loss_atr: {best['stop_loss_atr']}")
        print(f"  take_profit_atr: {best['take_profit_atr']}")
        print(f"  trail_atr: {best['trail_atr']}")
        print(f"\n建议写入 config/strategy.yaml 的 strategy 段。")
        print(f"当前默认值: stop=2.0 / target=3.0 / trail=2.5\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
