"""ML 训练 + 回测一体化：训练模型后在测试段回测验证。

无前视偏差：每只股票前 70% 训练，后 30% 回测。
用法：python src/ml_pipeline.py [--days 120]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ml_model import train, MLGenerator
from backtest import run_backtest, _print_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ml_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(description="ML 训练+回测一体化")
    parser.add_argument("--days", type=int, default=None,
                        help="回测天数（不指定则 15m=60天, 60m=365天）")
    args = parser.parse_args()

    with open(ROOT / "config" / "watchlist.yaml", encoding="utf-8") as f:
        wl = yaml.safe_load(f)
    with open(ROOT / "config" / "strategy.yaml", encoding="utf-8") as f:
        st = yaml.safe_load(f)
    params = st.get("strategy", {})
    watchlist = wl.get("watchlist", [])

    logger.info("=== 第一步：训练 ML 模型 ===")
    result = train(watchlist, params, args.days)
    if not result:
        logger.error("训练失败，退出")
        return 1

    model = result["model"]
    test_splits = result["test_splits"]
    data = result["data"]
    gen = MLGenerator(model_data={"model": model})

    logger.info("=== 第二步：测试段回测 ===")
    results = []
    for item in watchlist:
        symbol = item["symbol"]
        if symbol not in data:
            continue
        r = run_backtest(
            symbol, item.get("name", symbol), item.get("interval", "15m"),
            params, df_override=data[symbol], generator=gen,
            start_index=test_splits[symbol],
        )
        if r:
            results.append(r)

    _print_results(results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
