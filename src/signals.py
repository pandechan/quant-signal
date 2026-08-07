"""信号生成层：规则引擎生成买卖信号，预留 ML 接口。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd


@dataclass
class Signal:
    """单个信号。"""

    symbol: str
    name: str
    direction: str          # BUY / SELL / WATCH
    price: float
    confidence: int          # 1-5，越高越强
    reasons: list[str] = field(default_factory=list)
    interval: str = ""

    @property
    def key(self) -> str:
        """去重用的唯一键：股票+方向+周期。"""
        return f"{self.symbol}:{self.direction}:{self.interval}"


class SignalGenerator(Protocol):
    """信号生成器统一接口，规则引擎与 ML 模型均实现此接口。"""

    def generate(self, symbol: str, name: str, interval: str,
                 df: pd.DataFrame, params: dict) -> list[Signal]:
        ...


class RuleBasedGenerator:
    """基于技术指标规则的信号生成器。

    买入条件（每满足一条 +1 置信度）：
      1. 快线在慢线上方（EMA12 > EMA26）
      2. MACD 柱状图为正
      3. RSI 未超买（< overbought）
      4. 量能放大（vol_ratio > surge_ratio）
      5. ADX > 20（趋势明确）

    卖出条件对称。置信度低于阈值则标记为 WATCH。
    """

    def generate(self, symbol: str, name: str, interval: str,
                 df: pd.DataFrame, params: dict) -> list[Signal]:
        if df.empty or len(df) < 30:
            return []

        last = df.iloc[-1]
        price = float(last["close"])

        ema_fast = _col(df, f"EMA_{params.get('ema_fast', 12)}")
        ema_slow = _col(df, f"EMA_{params.get('ema_slow', 26)}")
        macd_hist = _col(df, "MACDh_")
        rsi = _col(df, "RSI_")
        adx = _col(df, "ADX_")
        stoch_k = _col(df, "STOCHk_")
        bbands_upper = _col(df, "BBU_")
        bbands_lower = _col(df, "BBL_")

        overbought = params.get("rsi_overbought", 70)
        oversold = params.get("rsi_oversold", 30)
        vol_surge = params.get("vol_surge_ratio", 1.5)
        min_conf = params.get("min_confidence", 2)

        signals: list[Signal] = []

        # ---- 买入信号 ----
        buy_reasons = []
        if ema_fast and ema_slow and last[ema_fast] > last[ema_slow]:
            buy_reasons.append("快线在慢线上方(多头排列)")
        if macd_hist and last[macd_hist] > 0:
            buy_reasons.append("MACD柱状图为正")
        if rsi and last[rsi] < overbought:
            buy_reasons.append(f"RSI={last[rsi]:.1f}未超买")
        if "vol_ratio" in last and pd.notna(last["vol_ratio"]) and last["vol_ratio"] > vol_surge:
            buy_reasons.append(f"量比={last['vol_ratio']:.2f}放大")
        if adx and pd.notna(last[adx]) and last[adx] > 20:
            buy_reasons.append(f"ADX={last[adx]:.1f}趋势明确")
        if bbands_lower and last["close"] <= last[bbands_lower]:
            buy_reasons.append("触及布林下轨(超跌)")

        if buy_reasons:
            conf = len(buy_reasons)
            direction = "BUY" if conf >= min_conf else "WATCH"
            signals.append(Signal(
                symbol=symbol, name=name, direction=direction,
                price=price, confidence=conf, reasons=buy_reasons, interval=interval,
            ))

        # ---- 卖出信号 ----
        sell_reasons = []
        if ema_fast and ema_slow and last[ema_fast] < last[ema_slow]:
            sell_reasons.append("快线在慢线下方(空头排列)")
        if macd_hist and last[macd_hist] < 0:
            sell_reasons.append("MACD柱状图为负")
        if rsi and last[rsi] > oversold:
            sell_reasons.append(f"RSI={last[rsi]:.1f}未超卖")
        if "vol_ratio" in last and pd.notna(last["vol_ratio"]) and last["vol_ratio"] > vol_surge:
            sell_reasons.append(f"量比={last['vol_ratio']:.2f}放大")
        if adx and pd.notna(last[adx]) and last[adx] > 20:
            sell_reasons.append(f"ADX={last[adx]:.1f}趋势明确")
        if bbands_upper and last["close"] >= last[bbands_upper]:
            sell_reasons.append("触及布林上轨(超涨)")

        if sell_reasons:
            conf = len(sell_reasons)
            direction = "SELL" if conf >= min_conf else "WATCH"
            signals.append(Signal(
                symbol=symbol, name=name, direction=direction,
                price=price, confidence=conf, reasons=sell_reasons, interval=interval,
            ))

        # 同一股票只保留置信度更高的那个
        if len(signals) > 1:
            signals = [max(signals, key=lambda s: s.confidence)]

        return signals


class MLGenerator:
    """机器学习信号生成器（预留接口）。

    后续加载 models/ 下训练好的模型，输入相同 features，
    输出 Signal。切换只需在配置中指定 generator: ml。
    """

    def generate(self, symbol: str, name: str, interval: str,
                 df: pd.DataFrame, params: dict) -> list[Signal]:
        raise NotImplementedError("ML 信号生成器尚未实现，请使用 rule 生成器")


def get_generator(name: str = "rule") -> SignalGenerator:
    """工厂方法：按名称获取信号生成器。"""
    generators = {"rule": RuleBasedGenerator, "ml": MLGenerator}
    cls = generators.get(name, RuleBasedGenerator)
    return cls()


def _col(df: pd.DataFrame, prefix: str) -> str | None:
    """按前缀模糊匹配列名。"""
    for c in df.columns:
        if c.startswith(prefix):
            return c
    return None
