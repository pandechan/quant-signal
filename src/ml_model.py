"""ML 信号生成器：基于 lightgbm 的趋势预测模型。

特征：技术指标 + 衍生统计量（收益率/波动率/相对位置）
标签：未来 N 根 K 线收益率方向（涨=1）
模型：lightgbm 二分类，每只股票前 70% 训练、后 30% 测试（无前视偏差）
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import add_indicators
from signals import Signal

logger = logging.getLogger("ml")

MODEL_PATH = ROOT / "models" / "lgb_model.joblib"

HORIZON = 4
THRESHOLD = 0.001

FEATURE_COLS = [
    "ema_ratio", "macd_hist", "rsi", "stoch_k", "stoch_d",
    "bb_position", "adx", "atr_pct", "vol_ratio",
    "return_1", "return_3", "return_5", "return_std_5",
    "close_to_ema50",
]


def build_features(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """从已计算指标的 DataFrame 构建特征矩阵。"""
    ema_fast = f"EMA_{params.get('ema_fast', 12)}"
    ema_slow = f"EMA_{params.get('ema_slow', 26)}"
    trend_ema = f"EMA_{params.get('trend_ema', 50)}"
    bbands = params.get("bbands_period", 20)
    bbl = f"BBL_{bbands}_2.0"
    bbu = f"BBU_{bbands}_2.0"

    out = pd.DataFrame(index=df.index)
    out["ema_ratio"] = df[ema_fast] / df[ema_slow]
    out["macd_hist"] = df["MACDh_12_26_9"]
    out["rsi"] = df["RSI_14"]
    out["stoch_k"] = df["STOCHk_14_3_3"]
    out["stoch_d"] = df["STOCHd_14_3_3"]
    out["bb_position"] = (df["close"] - df[bbl]) / (df[bbu] - df[bbl])
    out["adx"] = df["ADX_14"]
    out["atr_pct"] = df["ATR_14"] / df["close"]
    out["vol_ratio"] = df["vol_ratio"]
    out["return_1"] = df["close"].pct_change(1)
    out["return_3"] = df["close"].pct_change(3)
    out["return_5"] = df["close"].pct_change(5)
    out["return_std_5"] = out["return_1"].rolling(5).std()
    out["close_to_ema50"] = df["close"] / df[trend_ema] - 1
    return out


def build_labels(df: pd.DataFrame, horizon: int = HORIZON, threshold: float = THRESHOLD) -> pd.Series:
    """未来 horizon 根 K 线收益率方向。"""
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    return (future_return > threshold).astype(int)


def train(watchlist: list[dict], params: dict, days: int | None = None) -> dict | None:
    """训练全局 lightgbm 模型。

    每只股票内部按 70/30 时间分割，前段训练后段测试，无前视偏差。
    返回 {"model": model, "test_splits": {symbol: start}, "data": {symbol: df}}。
    """
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, classification_report

    X_train_all, y_train_all = [], []
    X_test_all, y_test_all = [], []
    test_splits: dict[str, int] = {}
    data: dict[str, pd.DataFrame] = {}

    for item in watchlist:
        symbol = item["symbol"]
        interval = item.get("interval", "15m")
        d = days or (60 if interval == "15m" else 365)
        try:
            from data import fetch_klines
            df = fetch_klines(symbol, interval=interval, period=f"{d}d")
        except Exception as e:
            logger.warning("  %s 数据加载失败: %s", symbol, e)
            continue
        if df is None or df.empty or len(df) < 100:
            continue

        df = add_indicators(df, params).dropna()
        if len(df) < 50:
            continue
        data[symbol] = df

        features = build_features(df, params)
        labels = build_labels(df)
        split = int(len(df) * 0.7)
        test_splits[symbol] = split

        train_feat = features.iloc[:split]
        train_lab = labels.iloc[:split]
        train_valid = train_feat.notna().all(axis=1) & train_lab.notna()
        X_train_all.append(train_feat[train_valid][FEATURE_COLS])
        y_train_all.append(train_lab[train_valid])

        test_feat = features.iloc[split:]
        test_lab = labels.iloc[split:]
        test_valid = test_feat.notna().all(axis=1) & test_lab.notna()
        X_test_all.append(test_feat[test_valid][FEATURE_COLS])
        y_test_all.append(test_lab[test_valid])
        logger.info("  %s: 训练 %d / 测试 %d", symbol, split, len(df) - split)

    if not X_train_all:
        logger.error("无可用训练数据")
        return None

    X_train = pd.concat(X_train_all)
    y_train = pd.concat(y_train_all)
    X_test = pd.concat(X_test_all)
    y_test = pd.concat(y_test_all)

    logger.info("训练集 %d 样本 (正样本 %.1f%%)，测试集 %d 样本",
                len(X_train), y_train.mean() * 100, len(X_test))

    model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info("测试集准确率: %.3f", acc)
    print("\n=== 模型评估 ===")
    print(classification_report(y_test, y_pred, target_names=["跌/平", "涨"]))

    imp = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    print("\n=== 特征重要性 ===")
    for name, val in imp.items():
        print(f"  {name:<20} {val}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, MODEL_PATH)
    logger.info("模型已保存到 %s", MODEL_PATH)

    return {"model": model, "test_splits": test_splits, "data": data}


class MLGenerator:
    """ML 信号生成器，实现 SignalGenerator 接口。"""

    def __init__(self, model_data: dict | None = None):
        self.model = None
        self.feature_cols = FEATURE_COLS
        if model_data is not None:
            self.model = model_data["model"]
            self.feature_cols = model_data.get("feature_cols", FEATURE_COLS)
        elif MODEL_PATH.exists():
            data = joblib.load(MODEL_PATH)
            self.model = data["model"]
            self.feature_cols = data.get("feature_cols", FEATURE_COLS)

    def predict_series(self, df: pd.DataFrame, params: dict) -> pd.Series:
        """返回每行的上涨概率序列（回测用，高效）。"""
        if self.model is None:
            return pd.Series(0.5, index=df.index)
        features = build_features(df, params)
        feats = features[self.feature_cols].fillna(0)
        probs = self.model.predict_proba(feats)[:, 1]
        return pd.Series(probs, index=df.index)

    def generate(self, symbol: str, name: str, interval: str,
                 df: pd.DataFrame, params: dict) -> list[Signal]:
        """实时用：取最后一行预测生成信号。"""
        if self.model is None or len(df) < 30:
            return []

        probs = self.predict_series(df, params)
        prob = float(probs.iloc[-1])
        price = float(df.iloc[-1]["close"])

        use_trend_filter = params.get("use_trend_filter", True)
        if use_trend_filter:
            trend_col = f"EMA_{params.get('trend_ema', 50)}"
            if trend_col in df.columns and pd.notna(df.iloc[-1].get(trend_col)):
                if df.iloc[-1]["close"] <= df.iloc[-1][trend_col]:
                    return []

        buy_threshold = params.get("ml_buy_threshold", 0.6)
        if prob > buy_threshold:
            confidence = min(int(prob * 5) + 1, 5)
            return [Signal(
                symbol=symbol, name=name, direction="BUY", price=price,
                confidence=confidence,
                reasons=[f"ML预测上涨概率={prob:.1%}"],
                interval=interval,
            )]
        return []


def main() -> int:
    import yaml
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="训练 ML 模型")
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args()

    with open(ROOT / "config" / "watchlist.yaml", encoding="utf-8") as f:
        wl = yaml.safe_load(f)
    with open(ROOT / "config" / "strategy.yaml", encoding="utf-8") as f:
        st = yaml.safe_load(f)
    params = st.get("strategy", {})

    logger.info("开始训练 ML 模型...")
    result = train(wl.get("watchlist", []), params, args.days)
    if result:
        logger.info("训练完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
