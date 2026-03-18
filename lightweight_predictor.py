"""
轻量级预测模块 - 使用技术指标趋势分析（无机器学习，无数据泄露）
适合 Streamlit Cloud 等内存受限环境
"""
import json
import os
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    from ass1_core import load_bundle as _load_bundle, daily_returns as _daily_returns
    _ASS1_CORE_AVAILABLE = True
except ImportError:
    _ASS1_CORE_AVAILABLE = False
    _load_bundle = None
    _daily_returns = None


def calculate_trend_score(prices: pd.Series, lookback: int = 60) -> float:
    """
    基于技术指标计算趋势得分（-1到1之间）
    正值表示看涨，负值表示看跌，0表示震荡
    """
    if len(prices) < lookback:
        return 0.0

    recent = prices.tail(lookback)

    # 1. 短期 vs 长期均线（5日 vs 20日）
    ma_5 = recent.tail(5).mean()
    ma_20 = recent.mean()
    ma_score = (ma_5 / ma_20 - 1) * 10  # 放大到合理范围

    # 2. 价格位置（近期高低点之间）
    price_pos = (recent.iloc[-1] - recent.min()) / (recent.max() - recent.min() + 1e-10)
    # 0表示在低点，1表示在高点
    # 转换为得分：低点时看涨，高点时看跌
    position_score = (price_pos - 0.5) * -1  # 反转

    # 3. 近期动量（最近5天 vs 前5天）
    if len(recent) >= 10:
        recent_5d = recent.tail(5).mean()
        prev_5d = recent.tail(10).head(5).mean()
        momentum_score = (recent_5d / prev_5d - 1) * 10
    else:
        momentum_score = 0

    # 4. 波动率调整（波动大时降低信心）
    returns = recent.pct_change().dropna()
    volatility = returns.std() if len(returns) > 0 else 0
    vol_adjustment = 1 / (1 + volatility * 10)  # 波动大时降低得分

    # 综合得分（加权平均）
    raw_score = (ma_score * 0.3 + position_score * 0.4 + momentum_score * 0.3)

    # 限制在合理范围并应用波动率调整
    final_score = np.clip(raw_score, -0.3, 0.3) * vol_adjustment

    return float(final_score)


def lightweight_forecast(
    close_df: pd.DataFrame,
    symbols: List[str] = None,
    lookback: int = 60,
    pred_len: int = 7,
    model_type: str = None  # 保留参数但不用
) -> Dict[str, Any]:
    """
    使用技术指标趋势分析预测未来收益（无ML，无数据泄露）
    """
    if symbols is None:
        symbols = list(close_df.columns)

    reg_results = []
    cls_results = {}

    for symbol in symbols:
        if symbol not in close_df.columns:
            continue

        try:
            prices = close_df[symbol].dropna()

            if len(prices) < lookback + 10:
                continue

            # 计算趋势得分
            trend_score = calculate_trend_score(prices, lookback)

            # 根据趋势得分预测未来收益
            # 假设7天累计收益 = 趋势得分 * 调整系数
            # 限制在合理范围（-15%到+15%）
            pred_cum_return = np.clip(trend_score * 0.5, -0.15, 0.15)

            # 日均收益
            pred_daily_return = pred_cum_return / pred_len

            last_close = prices.iloc[-1]
            pred_future_close = last_close * (1 + pred_cum_return)

            reg_results.append({
                "symbol": symbol,
                "pred_daily_return": float(pred_daily_return),
                "pred_7d_cum_return": float(pred_cum_return),
                "last_close": float(last_close),
                "pred_future_close": float(pred_future_close)
            })

            # 分类预测
            if pred_cum_return > 0.02:
                pred_class = 2  # Up
                probs = [0.15, 0.25, 0.6]
            elif pred_cum_return < -0.02:
                pred_class = 0  # Down
                probs = [0.6, 0.25, 0.15]
            else:
                pred_class = 1  # Flat
                probs = [0.25, 0.5, 0.25]

            cls_results[symbol] = {
                "pred_class": pred_class,
                "pred_probs": probs,
                "description": f"Trend analysis (score={trend_score:.3f})",
                "pred_cum_return": float(pred_cum_return),
                "trend_score": float(trend_score)
            }

        except Exception as e:
            print(f"Prediction failed for {symbol}: {e}")
            continue

    reg_df = pd.DataFrame(reg_results)
    if not reg_df.empty:
        reg_df = reg_df.set_index("symbol").sort_values("pred_7d_cum_return", ascending=False)

    return {
        "regression": reg_df,
        "classification": cls_results,
        "model_info": f"Trend analysis (no ML, lookback={lookback})"
    }


def run_lightweight_optimization(
    data_json_path: str,
    dataset: str = "universe",
    lookback: int = 60,
    pred_len: int = 7,
    model_type: str = None,
    out_dir: str = None
) -> Dict[str, Any]:
    """运行轻量级预测"""
    if not _ASS1_CORE_AVAILABLE or _load_bundle is None or _daily_returns is None:
        return {
            "regression": {},
            "classification": {},
            "model_info": "ass1_core not available",
            "metrics": {}
        }

    bundle = _load_bundle(data_json_path)

    if dataset == "assets":
        close = bundle.close_assets
    elif dataset == "stocks":
        close = bundle.close_stocks
    else:
        close = bundle.close_universe

    forecast = lightweight_forecast(
        close,
        symbols=list(close.columns),
        lookback=lookback,
        pred_len=pred_len
    )

    rets = _daily_returns(close)

    result = {
        "regression": forecast["regression"].to_dict() if not forecast["regression"].empty else {},
        "classification": forecast["classification"],
        "model_info": forecast["model_info"],
        "metrics": {
            "mse": None,
            "mae": None,
            "accuracy": None
        },
        "note": "Trend-based prediction - no ML, no data leakage"
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"trend_{dataset}_forecast.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    return result


# 兼容接口
LIGHTWEIGHT_AVAILABLE = True

def kronos_forecast(*args, **kwargs):
    """兼容接口：实际使用趋势分析"""
    kwargs.pop('device', None)
    kwargs.pop('model_name', None)
    return lightweight_forecast(*args, **kwargs)


if __name__ == "__main__":
    test_data_path = "/Users/dyl/Downloads/AIE1902-Ass1-main/data.json"
    if os.path.exists(test_data_path):
        result = run_lightweight_optimization(test_data_path, dataset="stocks")
        print("Trend forecast completed!")
        print(f"Symbols: {list(result['classification'].keys())[:5]}...")
        print(f"Model: {result['model_info']}")
        if result['classification']:
            first = list(result['classification'].values())[0]
            print(f"Example trend score: {first.get('trend_score', 'N/A')}")
    else:
        print("Test data not found")
