"""编排入口：加载配置 → 判断时段 → 拉数据 → 算指标 → 生信号 → 去重 → 推送。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import fetch_klines
from features import add_indicators
from signals import Signal, get_generator
from notify import send_signals
from market_calendar import is_market_open, now_et

PUSHED_FILE = ROOT / "data" / "pushed.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("quant-signal")


def load_config() -> tuple[dict, dict]:
    """加载 watchlist 与 strategy 配置。"""
    watchlist_path = ROOT / "config" / "watchlist.yaml"
    strategy_path = ROOT / "config" / "strategy.yaml"
    with open(watchlist_path, encoding="utf-8") as f:
        wl = yaml.safe_load(f)
    with open(strategy_path, encoding="utf-8") as f:
        st = yaml.safe_load(f)
    return wl, st


def load_pushed() -> dict:
    """加载已推送信号记录。"""
    if PUSHED_FILE.exists():
        try:
            return json.loads(PUSHED_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_pushed(data: dict) -> None:
    """保存已推送信号记录。"""
    PUSHED_FILE.parent.mkdir(parents=True, exist_ok=True)
    PUSHED_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def dedup(signals: list[Signal], pushed: dict, dedup_minutes: int) -> list[Signal]:
    """过滤掉去重窗口内已推送的信号。"""
    now = datetime.now(timezone.utc)
    result = []
    for s in signals:
        if s.direction == "WATCH":
            continue  # WATCH 信号不推送
        ts_str = pushed.get(s.key)
        if ts_str:
            try:
                ts = datetime.fromisoformat(ts_str)
                age = (now - ts).total_seconds() / 60
                if age < dedup_minutes:
                    logger.info("去重跳过 %s（%d分钟内已推送）", s.key, int(age))
                    continue
            except ValueError:
                pass
        result.append(s)
    return result


def scan(watchlist_cfg: dict, strategy_cfg: dict, generator_name: str = "rule") -> list[Signal]:
    """扫描全部关注股票，返回生成的信号。"""
    params = strategy_cfg.get("strategy", {})
    generator = get_generator(generator_name)
    all_signals: list[Signal] = []

    for item in watchlist_cfg.get("watchlist", []):
        symbol = item["symbol"]
        name = item.get("name", symbol)
        interval = item.get("interval", "15m")

        logger.info("扫描 %s(%s) %s", symbol, name, interval)
        try:
            df = fetch_klines(symbol, interval=interval)
            if df.empty:
                logger.warning("  %s 无数据，跳过", symbol)
                continue
            df = add_indicators(df, params)
            sigs = generator.generate(symbol, name, interval, df, params)
            all_signals.extend(sigs)
            for s in sigs:
                logger.info("  信号: %s %s 置信度=%d", s.direction, symbol, s.confidence)
        except Exception as e:
            logger.error("  %s 扫描失败: %s", symbol, e)
            continue

    return all_signals


def main() -> int:
    parser = argparse.ArgumentParser(description="美股技术分析信号扫描")
    parser.add_argument(
        "--force", action="store_true",
        help="跳过交易时段检查（用于测试）",
    )
    parser.add_argument(
        "--generator", default="rule", choices=["rule", "ml"],
        help="信号生成器：rule(规则) 或 ml(机器学习)",
    )
    args = parser.parse_args()

    # 交易时段检查
    if not args.force:
        et_now = now_et()
        if not is_market_open(et_now):
            logger.info("当前非美股交易时段(%s ET)，跳过", et_now.strftime("%H:%M"))
            return 0
        logger.info("美股交易时段，开始扫描(%s ET)", et_now.strftime("%H:%M"))

    watchlist_cfg, strategy_cfg = load_config()
    params = strategy_cfg.get("strategy", {})

    signals = scan(watchlist_cfg, strategy_cfg, args.generator)
    logger.info("共生成 %d 条信号", len(signals))

    dedup_minutes = params.get("dedup_minutes", 60)
    pushed = load_pushed()
    to_send = dedup(signals, pushed, dedup_minutes)

    logger.info("去重后待推送 %d 条", len(to_send))
    if to_send:
        ok = send_signals(to_send)
        if ok:
            # 记录已推送
            now_iso = datetime.now(timezone.utc).isoformat()
            for s in to_send:
                pushed[s.key] = now_iso
            save_pushed(pushed)
            logger.info("已更新推送记录")
        else:
            logger.error("推送失败，信号未记录去重")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
