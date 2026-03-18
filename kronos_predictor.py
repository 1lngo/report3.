"""
Kronos 预测模块 - 轻量级版本（云端部署）
使用技术指标趋势分析，无 PyTorch 依赖
"""
import json
import os
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Import ass1_core functions
try:
    from ass1_core import load_bundle as _load_bundle, daily_returns as _daily_returns
    _ASS1_CORE_AVAILABLE = True
except ImportError:
    _ASS1_CORE_AVAILABLE = False
    _load_bundle = None
    _daily_returns = None

# 导入轻量级预测模块
try:
    from lightweight_predictor import lightweight_forecast, run_lightweight_optimization
    KRONOS_AVAILABLE = True
except ImportError as e:
    KRONOS_AVAILABLE = False
    print(f"Lightweight predictor not available: {e}")
    lightweight_forecast = None
    run_lightweight_optimization = None

# 为了兼容，保留 KRONOS_MODE
KRONOS_MODE = "lightweight" if KRONOS_AVAILABLE else None


def prepare_ohlcv_from_close(
    close_df: pd.DataFrame,
    volume_df: pd.DataFrame = None,
    symbol: str = None
) -> pd.DataFrame:
    """
    从收盘价生成模拟的 OHLCV 数据
    """
    if symbol and symbol in close_df.columns:
        close = close_df[symbol].copy()
    else:
        close = close_df.iloc[:, 0].copy()

    ohlcv_data = []
    for i, (date, c) in enumerate(close.items()):
        if i > 0:
            o = close.iloc[i-1]
        else:
            o = c

        daily_range = abs(c - o) * 2 + c * 0.01
        h = max(o, c) + daily_range * 0.3
        l = min(o, c) - daily_range * 0.3
        h = max(h, o, c)
        l = min(l, o, c)

        if volume_df is not None and symbol in volume_df.columns:
            vol = volume_df.loc[date, symbol] if date in volume_df.index else np.random.randint(10000, 100000)
        else:
            vol = np.random.randint(10000, 100000)

        ohlcv_data.append({
            'timestamps': pd.to_datetime(date),
            'open': round(float(o), 2),
            'high': round(float(h), 2),
            'low': round(float(l), 2),
            'close': round(float(c), 2),
            'volume': int(vol)
        })

    return pd.DataFrame(ohlcv_data)


def kronos_forecast(
    close_df: pd.DataFrame,
    symbols: List[str] = None,
    lookback: int = 60,
    pred_len: int = 7,
    model_name: str = "lightweight",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    使用轻量级模型预测未来收益（无 PyTorch 依赖）
    """
    if not KRONOS_AVAILABLE or lightweight_forecast is None:
        return {
            "regression": pd.DataFrame(),
            "classification": {},
            "model_info": "Lightweight predictor not available"
        }

    # 调用轻量级预测
    return lightweight_forecast(
        close_df,
        symbols=symbols,
        lookback=lookback,
        pred_len=pred_len
    )


def run_kronos_optimization(
    data_json_path: str,
    dataset: str = "universe",
    lookback: int = 60,
    pred_len: int = 7,
    model_name: str = "lightweight",
    out_dir: str = None
) -> Dict[str, Any]:
    """
    运行轻量级预测优化
    """
    if not _ASS1_CORE_AVAILABLE or run_lightweight_optimization is None:
        return {
            "regression": {},
            "classification": {},
            "model_info": "ass1_core not available",
            "metrics": {}
        }

    return run_lightweight_optimization(
        data_json_path,
        dataset=dataset,
        lookback=lookback,
        pred_len=pred_len,
        out_dir=out_dir
    )


if __name__ == "__main__":
    test_data_path = "/Users/dyl/Downloads/AIE1902-Ass1-main/data.json"
    if os.path.exists(test_data_path):
        result = run_kronos_optimization(test_data_path, dataset="stocks")
        print("Lightweight forecast completed!")
        print(f"Symbols predicted: {list(result['classification'].keys())[:5]}...")
        print(f"Model: {result['model_info']}")
    else:
        print("Test data not found")
