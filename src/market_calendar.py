"""交易日历层：判断当前是否美股交易时段。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytz
import pandas_market_calendars as mcal

NYSE = mcal.get_calendar("NYSE")
ET = pytz.timezone("US/Eastern")

# 美股常规交易时段（美东时间）
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MIN = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MIN = 0


def now_et() -> datetime:
    """当前美东时间。"""
    return datetime.now(ET)


def is_trading_day(dt: datetime | None = None) -> bool:
    """判断指定日期（美东）是否为 NYSE 交易日。"""
    dt = dt or now_et()
    dt = dt.astimezone(ET)
    # 取当天 12:00 ET 作为日程查询点
    check = ET.localize(datetime(dt.year, dt.month, dt.day, 12, 0)) \
        if dt.tzinfo is None else dt.replace(hour=12, minute=0, second=0)
    schedule = NYSE.schedule(start_date=check.date(), end_date=check.date())
    return not schedule.empty


def is_market_open(dt: datetime | None = None) -> bool:
    """判断当前是否在美股常规交易时段内。

    用于 GitHub Actions 高频 cron 触发时，过滤非交易时段。
    """
    dt = dt or now_et()
    dt = dt.astimezone(ET)

    if not is_trading_day(dt):
        return False

    minutes = dt.hour * 60 + dt.minute
    open_minutes = MARKET_OPEN_HOUR * 60 + MARKET_OPEN_MIN
    close_minutes = MARKET_CLOSE_HOUR * 60 + MARKET_CLOSE_MIN

    return open_minutes <= minutes <= close_minutes


def utc_now() -> datetime:
    """当前 UTC 时间。"""
    return datetime.now(timezone.utc)
