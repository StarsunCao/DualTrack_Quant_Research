#!/usr/bin/env python
"""
SmartPromptAgent vs ML Tracks 对比回测。

对比 Agent 增强版 LLM 与三种 ML 模型在相同市场条件下的表现。
"""

import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def load_agent_cache(path: str, symbol: str = "CSI300") -> pd.DataFrame:
    """加载 Agent 缓存为信号 DataFrame。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                records.append({
                    "timestamp": data["timestamp"],
                    "symbol": data.get("symbol", symbol),
                    "signal": data["signal"],         # 注意：用 signal 而非 llm_signal
                    "confidence": data["confidence"],
                    "reasoning": data.get("reasoning", ""),
                    "latency_ms": data.get("latency_ms", 0),
                    "model": data.get("model", "smart-agent"),
                })
    return pd.DataFrame(records)


def run_ml_track(ml_model_name: str, ohlcv_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """运行 ML 模型回测信号生成（直接加载已训练模型，不重新训练）。"""
    from src.models.ml_track.features import FeatureEngineer
    from src.models.model_manager import get_model_manager, ModelType

    manager = get_model_manager()

    # 加载已训练模型
    if ml_model_name == "lr":
        model_type = ModelType.LOGISTIC_REGRESSION
    elif ml_model_name == "lstm":
        model_type = ModelType.LSTM
    elif ml_model_name == "lgb":
        model_type = ModelType.LIGHTGBM
    else:
        return pd.DataFrame()

    if not manager.model_exists(symbol, model_type):
        print(f"  ⚠️ {ml_model_name} 模型不存在: {symbol}")
        return pd.DataFrame()

    print(f"  加载已训练模型: {ml_model_name}...")
    model, metadata = manager.load_ml_model(symbol, model_type)

    # LSTM 模型需要移到 CPU 以避免 MPS 兼容性问题
    if ml_model_name == "lstm" and hasattr(model, 'model'):
        model.model = model.model.to('cpu')

    # 获取特征
    fe = FeatureEngineer()

    # 先用历史数据扩展避免窗口损失
    train_path = Path("data/raw/csi300_train_2015_2019.csv") if symbol == "CSI300" else Path("data/raw/qqq_train_2015_2017.csv")
    if train_path.exists():
        train_ohlcv = pd.read_csv(train_path, parse_dates=["date"])
        train_ohlcv.set_index("date", inplace=True)
        hist_start = "2019-01-01" if symbol == "CSI300" else "2017-10-01"
        historical = train_ohlcv[train_ohlcv.index >= hist_start]
        extended = pd.concat([historical, ohlcv_data])
    else:
        extended = ohlcv_data

    ext_features = fe.compute_all_features(extended, drop_na=True)

    test_start = "2020-01-01" if symbol == "CSI300" else "2018-01-01"
    test_features = ext_features[ext_features.index >= test_start]
    test_features_with_target = fe.create_target(test_features.copy(), forward_period=1)
    test_features_with_target = test_features_with_target.dropna()

    feature_cols = [c for c in test_features_with_target.columns if c not in ['target_label', 'target_return', 'symbol']]
    X_test = test_features_with_target[feature_cols].values
    proba = model.predict_proba(X_test)

    signals = pd.DataFrame({
        'timestamp': test_features_with_target.index,
        'symbol': symbol,
        'model_name': ml_model_name.upper(),
        'signal_strength_0_to_1': proba,
        'latency_ms': 2.0 if ml_model_name == "lr" else (15.0 if ml_model_name == "lstm" else 3.0)
    })

    return signals


def backtest_track(track_name: str, signals_df: pd.DataFrame, ohlcv_data: pd.DataFrame, symbol: str, initial_cash: float = 1000000, commission: float = 0.0002):
    """独立回测单个轨道。"""
    from src.orchestrator.signal_converter import SignalConverter
    from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

    # 转换信号到仓位
    if "signal_strength_0_to_1" in signals_df.columns:
        positions = SignalConverter.ml_signals_to_positions(signals_df, ohlcv_dates=ohlcv_data.index)
    elif "signal" in signals_df.columns:
        positions = SignalConverter.llm_signals_to_positions(signals_df, ohlcv_dates=ohlcv_data.index)
    else:
        print(f"  ⚠️ {track_name}: 无法识别信号列")
        return None

    if not positions:
        print(f"  ⚠️ {track_name}: 无仓位信号")
        return None

    engine = BacktestEngine(initial_cash=initial_cash, commission=commission)
    engine.add_data(ohlcv_data, name=symbol)
    engine.add_strategy(DualTrackStrategy, target_positions=positions, printlog=False, allow_short=False)
    result = engine.run()

    return result


def main():
    symbol = "CSI300"
    start = "2020-01-02"
    end = "2024-12-31"

    print("=" * 70)
    print("  SmartPromptAgent vs ML Tracks 对比回测")
    print("=" * 70)
    print(f"  标的: {symbol}")
    print(f"  日期: {start} ~ {end}")
    print("=" * 70)

    # 加载 OHLCV
    ohlcv_path = Path("data/raw/real_csi300_5y.csv")
    ohlcv_data = pd.read_csv(ohlcv_path, parse_dates=["date"])
    ohlcv_data.set_index("date", inplace=True)
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    ohlcv_data = ohlcv_data[(ohlcv_data.index >= start_dt) & (ohlcv_data.index <= end_dt)]
    print(f"\n  OHLCV: {len(ohlcv_data)} 条")

    results = {}

    # ========== 1. SmartPromptAgent ==========
    print(f"\n{'='*50}")
    print(f"  【轨道: SMART-AGENT】")
    print(f"{'='*50}")
    agent_cache = Path("data/llm_cache/llm_cache_CSI300_siliconflow_agent.jsonl")
    if agent_cache.exists():
        print(f"  加载 Agent 缓存: {agent_cache}")
        agent_signals = load_agent_cache(str(agent_cache), symbol)
        print(f"  信号数: {len(agent_signals)}")
        result = backtest_track("smart-agent", agent_signals, ohlcv_data, symbol)
        if result:
            results["SmartPromptAgent"] = result
            print(f"  ✅ 最终资产: {result.final_value:,.2f}")
            print(f"  ✅ 总收益率: {result.total_return:.2%}")
            print(f"  ✅ 夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  ✅ 最大回撤: {result.max_drawdown:.2%}")
    else:
        print(f"  ⚠️ Agent 缓存不存在")

    # ========== 2. Logistic Regression ==========
    print(f"\n{'='*50}")
    print(f"  【轨道: LR】")
    print(f"{'='*50}")
    lr_signals = run_ml_track("lr", ohlcv_data, symbol)
    if not lr_signals.empty:
        result = backtest_track("lr", lr_signals, ohlcv_data, symbol)
        if result:
            results["Logistic Regression"] = result
            print(f"  ✅ 最终资产: {result.final_value:,.2f}")
            print(f"  ✅ 总收益率: {result.total_return:.2%}")
            print(f"  ✅ 夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  ✅ 最大回撤: {result.max_drawdown:.2%}")

    # ========== 3. LSTM ==========
    print(f"\n{'='*50}")
    print(f"  【轨道: LSTM】")
    print(f"{'='*50}")
    lstm_signals = run_ml_track("lstm", ohlcv_data, symbol)
    if not lstm_signals.empty:
        result = backtest_track("lstm", lstm_signals, ohlcv_data, symbol)
        if result:
            results["LSTM"] = result
            print(f"  ✅ 最终资产: {result.final_value:,.2f}")
            print(f"  ✅ 总收益率: {result.total_return:.2%}")
            print(f"  ✅ 夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  ✅ 最大回撤: {result.max_drawdown:.2%}")

    # ========== 4. LightGBM ==========
    print(f"\n{'='*50}")
    print(f"  【轨道: LightGBM】")
    print(f"{'='*50}")
    lgb_signals = run_ml_track("lgb", ohlcv_data, symbol)
    if not lgb_signals.empty:
        result = backtest_track("lgb", lgb_signals, ohlcv_data, symbol)
        if result:
            results["LightGBM"] = result
            print(f"  ✅ 最终资产: {result.final_value:,.2f}")
            print(f"  ✅ 总收益率: {result.total_return:.2%}")
            print(f"  ✅ 夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  ✅ 最大回撤: {result.max_drawdown:.2%}")

    # ========== 对比表格 ==========
    print(f"\n{'='*70}")
    print("  【对比结果】")
    print(f"{'='*70}")
    print(f"\n{'轨道':<20} {'最终资产':>14} {'总收益率':>10} {'夏普比率':>10} {'最大回撤':>10}")
    print("-" * 70)

    for name, result in results.items():
        print(f"{name:<20} {result.final_value:>13,.2f} {result.total_return:>9.2%} {result.sharpe_ratio:>10.4f} {result.max_drawdown:>9.2%}")

    # 排序
    sorted_results = sorted(results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
    print(f"\n  📊 按夏普排序: {' > '.join(r[0] for r in sorted_results)}")

    # 保存权益曲线对比图
    if len(results) > 1:
        from src.evaluation.visualizer import plot_equity_curves
        output_path = Path("docs/figures")
        output_path.mkdir(parents=True, exist_ok=True)

        equity_curves = {}
        for name, result in results.items():
            equity_curves[name] = result.equity_curve

        plot_equity_curves(
            equity_curves,
            title=f"SmartPromptAgent vs ML Tracks: {symbol}",
            save_path=str(output_path / f"equity_curves_agent_vs_ml_{symbol}.png"),
        )
        print(f"\n  📊 对比图已保存: {output_path / f'equity_curves_agent_vs_ml_{symbol}.png'}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
