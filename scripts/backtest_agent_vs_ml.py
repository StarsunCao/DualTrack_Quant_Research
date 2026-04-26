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


def _tune_hyperparams(ml_model_name: str, X_tr: np.ndarray, y_tr: np.ndarray, feature_cols: list[str]) -> dict:
    """在第一个训练窗口上做超参数网格搜索，返回最优参数。"""
    from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
    from sklearn.linear_model import LogisticRegression
    import lightgbm as lgb

    cv = TimeSeriesSplit(n_splits=3)

    if ml_model_name == 'lr':
        param_dist = {
            'C': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear'],
            'max_iter': [1000],
        }
        base = LogisticRegression(random_state=42, n_jobs=-1)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=10, cv=cv, scoring='accuracy',
            random_state=42, n_jobs=-1,
        )
        # StandardScaler for LR
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_tr)
        search.fit(X_scaled, y_tr)
        best = search.best_params_
        print(f"  LR 超参搜索: C={best['C']}, penalty={best['penalty']} (CV acc={search.best_score_:.4f})")
        return best

    elif ml_model_name == 'lgb':
        param_dist = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [3, 4, 5, 6, 8, 10],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15, 0.2],
            'num_leaves': [15, 31, 63, 127],
            'min_child_samples': [5, 10, 20, 50],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0.0, 0.01, 0.1, 1.0],
            'reg_lambda': [0.0, 0.01, 0.1, 1.0],
        }
        base = lgb.LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
        search = RandomizedSearchCV(
            base, param_dist, n_iter=20, cv=cv, scoring='accuracy',
            random_state=42, n_jobs=-1,
        )
        search.fit(X_tr, y_tr)
        best = search.best_params_
        print(f"  LightGBM 超参搜索: n_estimators={best['n_estimators']}, max_depth={best['max_depth']}, "
              f"lr={best['learning_rate']}, num_leaves={best['num_leaves']}, "
              f"min_child={best['min_child_samples']}, subsample={best['subsample']:.1f} "
              f"(CV acc={search.best_score_:.4f})")
        return best

    else:
        return {}


def run_ml_track_walkforward(ml_model_name: str, full_ohlcv: pd.DataFrame, train_ohlcv: pd.DataFrame,
                             symbol: str, retrain_freq_days: int = 182, train_window_days: int = 912) -> pd.DataFrame:
    """Walk-Forward ML 信号生成：定期重训模型，生成全时间段信号。

    第一个训练窗口会做超参数搜索（LR/LightGBM），后续重训复用最优参数。

    Args:
        ml_model_name: 'lr', 'lstm', 或 'lgb'
        full_ohlcv: 完整 OHLCV 数据（训练 + 测试）
        train_ohlcv: 历史训练数据（用于初始化特征窗口）
        symbol: 资产代码
        retrain_freq_days: 重训频率（天），默认 182 天（约6个月）
        train_window_days: 训练窗口（天），默认 912 天（约2.5年）

    Returns:
        信号 DataFrame，与旧 run_ml_track 格式相同
    """
    from src.models.ml_track.features import FeatureEngineer
    from src.models.ml_track.baselines import LogisticRegressionModel, LightGBMModel, LSTMModel

    all_data = pd.concat([train_ohlcv, full_ohlcv])
    fe = FeatureEngineer()
    all_features = fe.compute_all_features(all_data, drop_na=False)

    test_dates = full_ohlcv.index
    feature_cols = [c for c in all_features.columns if c not in ['target_label', 'target_return', 'symbol']]

    ModelClass = {'lr': LogisticRegressionModel, 'lgb': LightGBMModel, 'lstm': LSTMModel}.get(ml_model_name)
    if ModelClass is None:
        return pd.DataFrame()

    all_probas = []
    retrain_freq = pd.Timedelta(days=retrain_freq_days)
    current_retrain_end = pd.Timestamp(test_dates[0])
    model = None
    retrain_count = 0
    best_params = None  # 超参搜索最优参数

    for dt in test_dates:
        # 定期重训
        if dt >= current_retrain_end:
            current_retrain_end = dt + retrain_freq
            train_start = dt - pd.Timedelta(days=train_window_days)
            train_end = dt
            train_feat = all_features[(all_features.index >= train_start) & (all_features.index <= train_end)].dropna()
            if len(train_feat) > 100:
                train_t = fe.create_target(train_feat.copy(), forward_period=1).dropna()
                if len(train_t) > 100 and len(set(train_t['target_label'])) > 1:
                    X_tr = train_t[feature_cols].values
                    y_tr = train_t['target_label'].values

                    # 第一个窗口做超参搜索
                    if best_params is None and ml_model_name in ('lr', 'lgb'):
                        print(f"  [{dt.date()}] 首个窗口，执行超参搜索...")
                        best_params = _tune_hyperparams(ml_model_name, X_tr, y_tr, feature_cols)
                        print(f"  使用搜索到的最优参数训练...")

                    if ml_model_name == 'lstm':
                        model = ModelClass(input_dim=X_tr.shape[1])
                    elif ml_model_name == 'lr':
                        if best_params:
                            model = LogisticRegressionModel(
                                C=best_params.get('C', 1.0),
                                max_iter=best_params.get('max_iter', 1000),
                            )
                        else:
                            model = ModelClass()
                    elif ml_model_name == 'lgb':
                        if best_params:
                            model = LightGBMModel(
                                n_estimators=best_params.get('n_estimators', 100),
                                max_depth=best_params.get('max_depth', 6),
                                learning_rate=best_params.get('learning_rate', 0.1),
                                num_leaves=best_params.get('num_leaves', 31),
                                min_child_samples=best_params.get('min_child_samples', 20),
                                subsample=best_params.get('subsample', 0.8),
                                colsample_bytree=best_params.get('colsample_bytree', 0.8),
                                reg_alpha=best_params.get('reg_alpha', 0.0),
                                reg_lambda=best_params.get('reg_lambda', 0.0),
                            )
                        else:
                            model = ModelClass()
                    model.fit(X_tr, y_tr)
                    retrain_count += 1

        # 生成信号
        if model is not None:
            # 对于 LSTM，需要传入截至当前的完整序列（至少 sequence_length 条）
            if ml_model_name == 'lstm':
                # 获取训练起始日到当前日期的全部特征
                hist_feat_start = dt - pd.Timedelta(days=train_window_days)
                hist = all_features[(all_features.index >= hist_feat_start) & (all_features.index <= dt)].dropna()
                if len(hist) >= 21:  # sequence_length + 1
                    X_hist = hist[feature_cols].values
                    proba = model.predict_proba(X_hist)
                    all_probas.append(proba[-1])  # 取最后一个（当前日期）
                else:
                    all_probas.append(0.5)
            else:
                row = all_features.loc[[dt]]
                row_clean = row.dropna(subset=feature_cols)
                if len(row_clean) == 1:
                    X = row_clean[feature_cols].values
                    proba = model.predict_proba(X)
                    all_probas.append(proba[0] if hasattr(proba, '__len__') else float(proba))
                else:
                    all_probas.append(0.5)
        else:
            all_probas.append(0.5)

    print(f"  Walk-Forward 信号: {len(all_probas)} 条, 重训 {retrain_count} 次")
    return pd.DataFrame({
        'timestamp': test_dates,
        'symbol': symbol,
        'model_name': ml_model_name.upper(),
        'signal_strength_0_to_1': all_probas,
        'latency_ms': 2.0 if ml_model_name == "lr" else (15.0 if ml_model_name == "lstm" else 3.0)
    })


def run_ml_track(ml_model_name: str, ohlcv_data: pd.DataFrame, symbol: str,
                 train_path: Path, hist_start: str, test_start: str) -> pd.DataFrame:
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
    if train_path.exists():
        train_ohlcv = pd.read_csv(train_path, parse_dates=["date"])
        train_ohlcv.set_index("date", inplace=True)
        historical = train_ohlcv[train_ohlcv.index >= hist_start]
        extended = pd.concat([historical, ohlcv_data])
    else:
        extended = ohlcv_data

    ext_features = fe.compute_all_features(extended, drop_na=True)

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


def backtest_track(track_name: str, signals_df: pd.DataFrame, ohlcv_data: pd.DataFrame, symbol: str, initial_cash: float = 1000000, commission: float = 0.0002, allow_short: bool = False, confidence_mode: str = "linear"):
    """独立回测单个轨道。"""
    from src.orchestrator.signal_converter import SignalConverter
    from src.execution.bt_engine import BacktestEngine, DualTrackStrategy
    from src.config.market_config import MarketConfig

    # 获取市场配置中的平滑参数
    mkt_config = MarketConfig.get_config_for_symbol(symbol)
    ema_alpha = mkt_config.ema_alpha
    decay_rate = mkt_config.decay_rate

    # 转换信号到仓位（带 EMA 平滑）
    if "signal_strength_0_to_1" in signals_df.columns:
        positions = SignalConverter.ml_signals_to_positions(
            signals_df, ohlcv_dates=ohlcv_data.index,
            ema_alpha=ema_alpha, decay_rate=decay_rate,
        )
    elif "signal" in signals_df.columns:
        positions = SignalConverter.llm_signals_to_positions(
            signals_df, ohlcv_dates=ohlcv_data.index, confidence_mode=confidence_mode,
            ema_alpha=ema_alpha, decay_rate=decay_rate,
        )
    else:
        print(f"  ⚠️ {track_name}: 无法识别信号列")
        return None

    if not positions:
        print(f"  ⚠️ {track_name}: 无仓位信号")
        return None

    # 计算换手率
    turnover = SignalConverter._compute_turnover(positions, ohlcv_data.index, symbol)

    engine = BacktestEngine(initial_cash=initial_cash, commission=commission)
    engine.add_data(ohlcv_data, name=symbol)
    engine.add_strategy(DualTrackStrategy, target_positions=positions, printlog=False, allow_short=allow_short)
    result = engine.run()

    # 附加换手率到结果
    result.turnover = turnover

    return result


def main(symbol="QQQ"):
    # 根据标的设置路径和参数
    if symbol == "QQQ":
        ohlcv_path = Path("data/raw/real_qqq_5y.csv")
        agent_cache_path = Path("data/llm_cache/llm_cache_QQQ_siliconflow_agent.jsonl")
        train_path_raw = Path("data/raw/qqq_train_2015_2017.csv")
        hist_start = "2017-10-01"
        test_start = "2018-01-01"
        start = "2018-01-02"
        end = "2024-12-31"
        commission = 0.0005  # 美股佣金
    else:
        ohlcv_path = Path("data/raw/real_csi300_5y.csv")
        agent_cache_path = Path("data/llm_cache/llm_cache_CSI300_siliconflow_agent.jsonl")
        train_path_raw = Path("data/raw/csi300_train_2015_2019.csv")
        hist_start = "2019-01-01"
        test_start = "2020-01-01"
        start = "2020-01-02"
        end = "2024-12-31"
        commission = 0.0012  # A股佣金（含印花税）

    print("=" * 70)
    print("  SmartPromptAgent vs ML Tracks 对比回测")
    print("=" * 70)
    print(f"  标的: {symbol}")
    print(f"  日期: {start} ~ {end}")
    print("=" * 70)

    # 根据市场配置决定允许做空
    from src.config.market_config import MarketConfig
    market_config = MarketConfig.get_config_for_symbol(symbol)
    allow_short = market_config.allow_short_selling
    print(f"  允许做空: {allow_short}")

    # 加载 OHLCV（完整数据，供 Walk-Forward 使用）
    full_ohlcv = pd.read_csv(ohlcv_path, parse_dates=["date"])
    full_ohlcv.set_index("date", inplace=True)

    # 训练数据（历史数据）
    if train_path_raw.exists():
        train_ohlcv = pd.read_csv(train_path_raw, parse_dates=["date"])
        train_ohlcv.set_index("date", inplace=True)
    else:
        train_ohlcv = pd.DataFrame()

    # 测试数据（过滤到回测区间）
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    ohlcv_data = full_ohlcv[(full_ohlcv.index >= start_dt) & (full_ohlcv.index <= end_dt)]
    print(f"\n  OHLCV: {len(ohlcv_data)} 条 (完整: {len(full_ohlcv)} 条)")

    results = {}

    # ========== 1. SmartPromptAgent ==========
    print(f"\n{'='*50}")
    print(f"  【轨道: SMART-AGENT】")
    print(f"{'='*50}")
    if agent_cache_path.exists():
        print(f"  加载 Agent 缓存: {agent_cache_path}")
        agent_signals = load_agent_cache(str(agent_cache_path), symbol)
        print(f"  信号数: {len(agent_signals)}")
        result = backtest_track("smart-agent", agent_signals, ohlcv_data, symbol, commission=commission, allow_short=allow_short)
        if result:
            results["SmartPromptAgent"] = result
            print(f"  ✅ 最终资产: {result.final_value:,.2f}")
            print(f"  ✅ 总收益率: {result.total_return:.2%}")
            print(f"  ✅ 夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  ✅ 最大回撤: {result.max_drawdown:.2%}")
            print(f"  ✅ 日换手率: {getattr(result, 'turnover', 0):.4f}")
    else:
        print(f"  ⚠️ Agent 缓存不存在")

    # ========== 2. Logistic Regression ==========
    print(f"\n{'='*50}")
    print(f"  【轨道: LR (Walk-Forward)】")
    print(f"{'='*50}")
    lr_signals = run_ml_track_walkforward("lr", full_ohlcv, train_ohlcv, symbol)
    if not lr_signals.empty:
        result = backtest_track("lr", lr_signals, ohlcv_data, symbol, commission=commission, allow_short=allow_short)
        if result:
            results["Logistic Regression"] = result
            print(f"  ✅ 最终资产: {result.final_value:,.2f}")
            print(f"  ✅ 总收益率: {result.total_return:.2%}")
            print(f"  ✅ 夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  ✅ 最大回撤: {result.max_drawdown:.2%}")
            print(f"  ✅ 日换手率: {getattr(result, 'turnover', 0):.4f}")

    # ========== 3. LSTM ==========
    print(f"\n{'='*50}")
    print(f"  【轨道: LSTM (Walk-Forward)】")
    print(f"{'='*50}")
    lstm_signals = run_ml_track_walkforward("lstm", full_ohlcv, train_ohlcv, symbol)
    if not lstm_signals.empty:
        result = backtest_track("lstm", lstm_signals, ohlcv_data, symbol, commission=commission, allow_short=allow_short)
        if result:
            results["LSTM"] = result
            print(f"  ✅ 最终资产: {result.final_value:,.2f}")
            print(f"  ✅ 总收益率: {result.total_return:.2%}")
            print(f"  ✅ 夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  ✅ 最大回撤: {result.max_drawdown:.2%}")
            print(f"  ✅ 日换手率: {getattr(result, 'turnover', 0):.4f}")

    # ========== 4. LightGBM ==========
    print(f"\n{'='*50}")
    print(f"  【轨道: LightGBM (Walk-Forward)】")
    print(f"{'='*50}")
    lgb_signals = run_ml_track_walkforward("lgb", full_ohlcv, train_ohlcv, symbol)
    if not lgb_signals.empty:
        result = backtest_track("lgb", lgb_signals, ohlcv_data, symbol, commission=commission, allow_short=allow_short)
        if result:
            results["LightGBM"] = result
            print(f"  ✅ 最终资产: {result.final_value:,.2f}")
            print(f"  ✅ 总收益率: {result.total_return:.2%}")
            print(f"  ✅ 夏普比率: {result.sharpe_ratio:.4f}")
            print(f"  ✅ 最大回撤: {result.max_drawdown:.2%}")
            print(f"  ✅ 日换手率: {getattr(result, 'turnover', 0):.4f}")

    # ========== 对比表格 ==========
    print(f"\n{'='*70}")
    print("  【对比结果】")
    print(f"{'='*70}")
    print(f"\n{'轨道':<20} {'最终资产':>14} {'总收益率':>10} {'夏普比率':>10} {'最大回撤':>10} {'日换手':>10}")
    print("-" * 80)

    for name, result in results.items():
        turnover = getattr(result, 'turnover', 0)
        print(f"{name:<20} {result.final_value:>13,.2f} {result.total_return:>9.2%} {result.sharpe_ratio:>10.4f} {result.max_drawdown:>9.2%} {turnover:>10.4f}")

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
    main("QQQ")
