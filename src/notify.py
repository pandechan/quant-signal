"""推送层：通过 Server酱 将信号推送到微信。"""

from __future__ import annotations

import os
import logging

import requests

from signals import Signal

logger = logging.getLogger(__name__)

SERVERCHAN_API = "https://sctapi.ftqq.com/{key}.send"

DIRECTION_EMOJI = {"BUY": "买入", "SELL": "卖出", "WATCH": "关注"}
DIRECTION_TAG = {"BUY": "买入", "SELL": "卖出", "WATCH": "关注"}


def send_signals(signals: list[Signal]) -> bool:
    """将信号列表推送至微信。

    Returns:
        True 表示推送成功（或无信号跳过），False 表示推送失败。
    """
    if not signals:
        logger.info("无信号需要推送")
        return True

    key = os.environ.get("SERVERCHAN_SENDKEY", "").strip()
    if not key:
        logger.warning("未配置 SERVERCHAN_SENDKEY，跳过推送（信号已生成）")
        for s in signals:
            logger.info("信号: %s %s %s 置信度=%d", s.symbol, s.direction, s.name, s.confidence)
        return True

    title, desp = _format_message(signals)

    try:
        resp = requests.post(
            SERVERCHAN_API.format(key=key),
            data={"title": title, "desp": desp},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("推送成功: %s", title)
        return True
    except Exception as e:
        logger.error("推送失败: %s", e)
        return False


def _format_message(signals: list[Signal]) -> tuple[str, str]:
    """构造推送标题与正文（Markdown）。"""
    # 标题：汇总方向
    parts = [f"{s.name}{DIRECTION_TAG.get(s.direction, s.direction)}" for s in signals]
    title = " | ".join(parts)

    lines = [
        "## 美股技术分析信号",
        "",
    ]
    for s in signals:
        lines.append(f"### {s.name}（{s.symbol}）— {DIRECTION_EMOJI.get(s.direction, s.direction)}")
        lines.append("")
        lines.append(f"- **周期**: {s.interval}")
        lines.append(f"- **现价**: ${s.price:.2f}")
        lines.append(f"- **置信度**: {'★' * s.confidence}{'☆' * (5 - s.confidence)}（{s.confidence}/5）")
        lines.append(f"- **触发条件**:")
        for r in s.reasons:
            lines.append(f"  - {r}")
        lines.append("")

    lines.append("---")
    lines.append("> ⚠️ 本信号基于技术指标自动生成，不构成投资建议。投资有风险，决策需谨慎。")

    return title, "\n".join(lines)
