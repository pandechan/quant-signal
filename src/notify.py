"""推送层：通过飞书自定义群机器人推送信号。

配置（环境变量）：
  FEISHU_WEBHOOK  飞书机器人 webhook 完整地址（必填）
  FEISHU_SECRET   签名校验密钥（可选，创建机器人时启用加签才填）
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import time

import requests

from signals import Signal

logger = logging.getLogger(__name__)

DIRECTION_LABEL = {"BUY": "买入", "SELL": "卖出", "WATCH": "关注"}


def send_signals(signals: list[Signal]) -> bool:
    """将信号列表推送至飞书群。

    Returns:
        True 表示推送成功（或无信号跳过），False 表示推送失败。
    """
    if not signals:
        logger.info("无信号需要推送")
        return True

    webhook = os.environ.get("FEISHU_WEBHOOK", "").strip()
    if not webhook:
        logger.warning("未配置 FEISHU_WEBHOOK，跳过推送（信号已生成）")
        for s in signals:
            logger.info("信号: %s %s %s 置信度=%d", s.symbol, s.direction, s.name, s.confidence)
        return True

    payload = _build_payload(signals)

    # 可选签名校验
    secret = os.environ.get("FEISHU_SECRET", "").strip()
    if secret:
        timestamp, sign = _sign(secret)
        payload["timestamp"] = timestamp
        payload["sign"] = sign

    try:
        resp = requests.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        code = data.get("code", data.get("StatusCode", -1))
        if code != 0:
            logger.error("飞书返回错误: %s", data)
            return False
        logger.info("飞书推送成功")
        return True
    except Exception as e:
        logger.error("飞书推送失败: %s", e)
        return False


def _sign(secret: str) -> tuple[str, str]:
    """计算飞书机器人签名。"""
    timestamp = str(int(time.time()))
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    sign = base64.b64encode(hmac_code).decode("utf-8")
    return timestamp, sign


def _card_color(signals: list[Signal]) -> str:
    """根据信号方向决定卡片颜色：买入绿、卖出红、混合橙、其他蓝。"""
    has_buy = any(s.direction == "BUY" for s in signals)
    has_sell = any(s.direction == "SELL" for s in signals)
    if has_buy and has_sell:
        return "orange"
    if has_buy:
        return "green"
    if has_sell:
        return "red"
    return "blue"


def _build_payload(signals: list[Signal]) -> dict:
    """构造飞书交互式卡片消息。"""
    color = _card_color(signals)
    title = " | ".join(
        f"{s.name}{DIRECTION_LABEL.get(s.direction, s.direction)}" for s in signals
    )

    elements: list[dict] = []
    for s in signals:
        conf = min(s.confidence, 5)
        stars = "*" * conf + "·" * (5 - conf)
        reason_lines = "\n".join(f"- {r}" for r in s.reasons)
        content = (
            f"**{s.name}（{s.symbol}）— {DIRECTION_LABEL.get(s.direction, s.direction)}**\n"
            f"周期：{s.interval}　|　现价：${s.price:.2f}\n"
            f"置信度：{stars}（{conf}/5）\n"
            f"触发条件：\n{reason_lines}"
        )
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": content}})
        elements.append({"tag": "hr"})

    # 去掉末尾多余分隔线
    if elements and elements[-1].get("tag") == "hr":
        elements.pop()

    elements.append({
        "tag": "note",
        "elements": [{
            "tag": "plain_text",
            "content": "本信号基于技术指标自动生成，不构成投资建议。投资有风险，决策需谨慎。",
        }],
    })

    return {
        "msg_type": "interactive",
        "card": {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"tag": "plain_text", "content": f"美股信号 | {title}"},
                "template": color,
            },
            "elements": elements,
        },
    }
