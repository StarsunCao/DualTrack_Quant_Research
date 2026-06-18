#!/usr/bin/env python
"""
Chapter 3 实验结果分析：完整编排脚本

整合已有模块，生成论文第三章所需的所有表格、图表和分析报告。
不重新训练模型，不重新调用 LLM API。

使用方法:
    python scripts/chapter3_analysis.py --symbol CSI300
    python scripts/chapter3_analysis.py --symbol QQQ
    python scripts/chapter3_analysis.py --symbol ALL
"""

import sys
import json
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from dotenv import load_dotenv
load_dotenv()

# 输出目录
OUTPUT_DIR = Path("output/chapter3")
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
ANALYSIS_DIR = OUTPUT_DIR / "analysis"

for d in [TABLES_DIR, FIGURES_DIR, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# 市场配置映射
MARKET_CONFIG = {
    "CSI300": {
        "start": "2020-01-02",
        "end": "2024-12-31",
        "commission": 0.0012,
        "allow_short": False,
        "signal_threshold": 0.55,
        "ema_alpha": 0.30,
        "decay_rate": 0.70,
        "weak_buy_threshold": 0.60,
        "weak_short_threshold": 0.60,
    },
    "QQQ": {
        "start": "2018-01-02",
        "end": "2020-07-22",
        "commission": 0.0005,
        "allow_short": True,
        "signal_threshold": 0.55,
        "ema_alpha": 0.50,
        "decay_rate": 0.80,
        "weak_buy_threshold": 0.55,
        "weak_short_threshold": 0.60,
    },
}


# ============================================================================
# A. 数据准备层
# ============================================================================

def load_ml_signals(symbol: str, model_name: str) -> pd.DataFrame:
    """加载单个 ML 模型的信号 CSV。"""
    path = Path(f"data/signals/{symbol}_{model_name.upper()}_walkforward.csv")
    if not path.exists():
        print(f"  ⚠️ 信号文件不存在: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def load_all_ml_signals(symbol: str) -> dict:
    """加载 3 个 ML 模型 + ensemble 的信号。"""
    signals = {}
    for m in ["LR", "LSTM", "LGB"]:
        df = load_ml_signals(symbol, m)
        if not df.empty:
            signals[m] = df
    # 计算 ensemble
    if len(signals) >= 2:
        signals["ENSEMBLE"] = compute_ensemble(signals, symbol)
    return signals


def compute_ensemble(ml_signals: dict, symbol: str,
                     weights: dict = None) -> pd.DataFrame:
    """ML 模型集成：加权投票融合信号。"""
    if weights is None:
        weights = {'LR': 0.4, 'LGB': 0.35, 'LSTM': 0.25}

    signal_series = {}
    for name, df in ml_signals.items():
        if df.empty or name in ('ENSEMBLE',):
            continue
        series = df.set_index('timestamp')['signal_strength_0_to_1']
        signal_series[name] = (series, weights.get(name, 0.1))

    if not signal_series:
        return pd.DataFrame()

    all_indices = [s.index for s, _ in signal_series.values()]
    common_dates = all_indices[0]
    for idx in all_indices[1:]:
        common_dates = common_dates.intersection(idx)

    if len(common_dates) == 0:
        return pd.DataFrame()

    ensemble_proba = np.zeros(len(common_dates))
    total_weight = 0
    for name, (series, w) in signal_series.items():
        ensemble_proba += series.loc[common_dates].values * w
        total_weight += w

    if total_weight > 0:
        ensemble_proba /= total_weight

    return pd.DataFrame({
        'timestamp': common_dates,
        'symbol': symbol,
        'model_name': 'ENSEMBLE',
        'signal_strength_0_to_1': ensemble_proba,
        'latency_ms': 5.0,
    })


def load_llm_signals_from_cache(cache_path: str, symbol: str) -> pd.DataFrame:
    """从单个 LLM JSONL 缓存提取信号。"""
    records = []
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                records.append({
                    "timestamp": pd.to_datetime(data["timestamp"]),
                    "symbol": data.get("symbol", symbol),
                    "model_name": data.get("model", "unknown"),
                    "signal": data.get("signal", "hold"),
                    "confidence": data.get("confidence", 0.5),
                    "reasoning": data.get("reasoning", ""),
                    "latency_ms": data.get("latency_ms", 0),
                })
    return pd.DataFrame(records)


def load_all_llm_signals(symbol: str) -> dict:
    """加载该标的下所有 LLM 模型的信号。"""
    cache_dir = Path("data/llm_cache")
    signals = {}
    pattern = f"llm_cache_{symbol}_*_agent.jsonl"
    for cache_path in sorted(cache_dir.glob(pattern)):
        # 跳过 agent_memory 文件
        if "agent_memory" in cache_path.name:
            continue
        model_name = cache_path.stem.replace(f"llm_cache_{symbol}_", "").replace("_agent", "")
        df = load_llm_signals_from_cache(str(cache_path), symbol)
        if not df.empty:
            signals[model_name] = df
    return signals


def load_ohlcv(symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
    """加载 OHLCV 数据并过滤日期范围。"""
    cfg = MARKET_CONFIG[symbol]
    if symbol == "QQQ":
        path = Path("data/raw/real_qqq_5y.csv")
    else:
        path = Path("data/raw/real_csi300_5y.csv")

    df = pd.read_csv(path, parse_dates=["date"])
    df.set_index("date", inplace=True)
    df.sort_index(inplace=True)

    s = start or cfg["start"]
    e = end or cfg["end"]
    df = df.loc[s:e].copy()
    return df


def generate_dataset_stats() -> pd.DataFrame:
    """生成数据集描述统计（表 3）。"""
    records = []
    for symbol in ["CSI300", "QQQ"]:
        cfg = MARKET_CONFIG[symbol]
        ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])
        returns = ohlcv["close"].pct_change().dropna()

        records.append({
            "market": symbol,
            "start_date": cfg["start"],
            "end_date": cfg["end"],
            "trading_days": len(ohlcv),
            "avg_daily_return": returns.mean(),
            "annualized_return": returns.mean() * 252,
            "annualized_volatility": returns.std() * np.sqrt(252),
            "max_single_day_gain": returns.max(),
            "max_single_day_loss": returns.min(),
            "start_price": ohlcv["close"].iloc[0],
            "end_price": ohlcv["close"].iloc[-1],
            "buyhold_return": ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[0] - 1,
        })

    stats = pd.DataFrame(records)
    stats.to_csv(TABLES_DIR / "table3_dataset_stats.csv", index=False)
    print("\n表 3: 数据集统计")
    print(stats.to_string(index=False))
    return stats


# ============================================================================
# B. 回测执行层
# ============================================================================

def run_buyhold_baseline(symbol: str, ohlcv: pd.DataFrame) -> dict:
    """计算买入持有基线指标。"""
    cfg = MARKET_CONFIG[symbol]
    start_price = ohlcv["close"].iloc[0]
    end_price = ohlcv["close"].iloc[-1]
    total_return = end_price / start_price - 1

    returns = ohlcv["close"].pct_change().dropna()
    n_days = len(returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # 最大回撤
    cummax = ohlcv["close"].cummax()
    drawdown = (ohlcv["close"] - cummax) / cummax
    max_dd = abs(drawdown.min())

    return {
        "strategy": "Buy & Hold",
        "symbol": symbol,
        "total_return": total_return,
        "annual_return": ann_return,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "turnover": 0.0,
        "win_rate": (returns > 0).mean(),
    }


def run_ml_backtest_for_symbol(symbol: str) -> dict:
    """使用已有信号 CSV 运行 ML 策略回测（不重新训练）。"""
    from src.orchestrator.signal_converter import SignalConverter
    from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

    cfg = MARKET_CONFIG[symbol]
    ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])
    signals = load_all_ml_signals(symbol)

    results = {}
    for name, sig_df in signals.items():
        if sig_df.empty:
            continue

        # 过滤日期范围
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
        sig_df = sig_df[sig_df["timestamp"] >= pd.Timestamp(cfg["start"])].copy()
        sig_df = sig_df[sig_df["timestamp"] <= pd.Timestamp(cfg["end"])].copy()

        positions = SignalConverter.ml_signals_to_positions(
            sig_df, ohlcv_dates=ohlcv.index,
            ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
            signal_threshold=cfg["signal_threshold"],
        )

        if not positions:
            print(f"  ⚠️ {name}: 无仓位信号")
            continue

        turnover = SignalConverter._compute_turnover(positions, ohlcv.index, symbol)

        engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
        engine.add_data(ohlcv, name=symbol)
        engine.add_strategy(DualTrackStrategy, target_positions=positions,
                           printlog=False, allow_short=cfg["allow_short"])
        result = engine.run()

        results[name] = {
            "strategy": f"ML-{name}",
            "symbol": symbol,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "turnover": turnover,
            "final_value": result.final_value,
        }

        print(f"  {name}: return={result.total_return:.2%}, sharpe={result.sharpe_ratio:.4f}, "
              f"maxdd={result.max_drawdown:.2%}, turnover={turnover:.4f}")

    # 买入持有基线
    results["BuyHold"] = run_buyhold_baseline(symbol, ohlcv)
    print(f"  BuyHold: return={results['BuyHold']['total_return']:.2%}, "
          f"sharpe={results['BuyHold']['sharpe_ratio']:.4f}")

    return results


def run_llm_backtest_for_symbol(symbol: str) -> dict:
    """使用已有 LLM 缓存运行回测。"""
    from src.orchestrator.signal_converter import SignalConverter
    from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

    cfg = MARKET_CONFIG[symbol]
    ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])
    signals = load_all_llm_signals(symbol)

    results = {}
    for name, sig_df in signals.items():
        if sig_df.empty:
            continue

        # 过滤日期范围
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
        sig_df = sig_df[sig_df["timestamp"] >= pd.Timestamp(cfg["start"])].copy()
        sig_df = sig_df[sig_df["timestamp"] <= pd.Timestamp(cfg["end"])].copy()

        positions = SignalConverter.llm_signals_to_positions(
            sig_df, ohlcv_dates=ohlcv.index,
            confidence_mode="linear",
            ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
        )

        if not positions:
            continue

        turnover = SignalConverter._compute_turnover(positions, ohlcv.index, symbol)

        engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
        engine.add_data(ohlcv, name=symbol)
        engine.add_strategy(DualTrackStrategy, target_positions=positions,
                           printlog=False, allow_short=cfg["allow_short"])
        result = engine.run()

        results[name] = {
            "strategy": f"LLM-{name}",
            "symbol": symbol,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "sharpe_ratio": result.sharpe_ratio,
            "max_drawdown": result.max_drawdown,
            "turnover": turnover,
            "final_value": result.final_value,
        }

        print(f"  {name}: return={result.total_return:.2%}, sharpe={result.sharpe_ratio:.4f}")

    results["BuyHold"] = run_buyhold_baseline(symbol, ohlcv)
    return results


# ============================================================================
# C. 图表生成层
# ============================================================================

def plot_equity_curves_with_llm(ml_results: dict, llm_results: dict, symbol: str):
    """净值曲线对比图（图 7/8）：包含 ML + LLM + BuyHold，优化日期标签。"""
    from src.orchestrator.signal_converter import SignalConverter
    from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

    cfg = MARKET_CONFIG[symbol]
    ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])
    ml_signals = load_all_ml_signals(symbol)
    llm_signals = load_all_llm_signals(symbol)

    # 收集所有策略的净值曲线
    equity_curves = {}

    def run_strategy_and_get_equity(name, sig_df, is_llm=False):
        if sig_df.empty:
            return None
        sig_df = sig_df[sig_df["timestamp"] >= pd.Timestamp(cfg["start"])].copy()
        sig_df = sig_df[sig_df["timestamp"] <= pd.Timestamp(cfg["end"])].copy()
        if is_llm:
            positions = SignalConverter.llm_signals_to_positions(
                sig_df.copy(), ohlcv_dates=ohlcv.index,
                confidence_mode="linear",
                ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
            )
        else:
            positions = SignalConverter.ml_signals_to_positions(
                sig_df, ohlcv_dates=ohlcv.index,
                ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
                signal_threshold=cfg["signal_threshold"],
            )
        if not positions:
            return None
        engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
        engine.add_data(ohlcv, name=symbol)
        engine.add_strategy(DualTrackStrategy, target_positions=positions,
                           printlog=False, allow_short=cfg["allow_short"])
        result = engine.run()
        if hasattr(result, 'equity_curve') and result.equity_curve is not None and not result.equity_curve.empty:
            eq = result.equity_curve.copy()
            # equity_curve 的 date 已经是 index
            if "value" in eq.columns:
                eq["nav"] = eq["value"] / eq["value"].iloc[0]
                return eq["nav"]
        return None

    # 买入持有
    bh_nav = (ohlcv["close"] / ohlcv["close"].iloc[0])
    equity_curves["Buy & Hold"] = bh_nav

    # ML 策略
    for name, sig_df in sorted(ml_signals.items()):
        nav = run_strategy_and_get_equity(f"ML-{name}", sig_df, is_llm=False)
        if nav is not None:
            equity_curves[f"ML-{name}"] = nav

    # LLM 策略 — 只选表现最好的 2 个避免图例过密
    llm_sorted = sorted(llm_results.items(), key=lambda x: x[1]["sharpe_ratio"], reverse=True)
    for llm_name, _ in llm_sorted[:2]:
        sig_df = llm_signals.get(llm_name)
        if sig_df is not None and not sig_df.empty:
            nav = run_strategy_and_get_equity(f"LLM-{llm_name}", sig_df, is_llm=True)
            if nav is not None:
                equity_curves[f"LLM-{llm_name}"] = nav

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # 颜色方案
    color_map = {}
    color_map["Buy & Hold"] = '#95a5a6'
    for i, name in enumerate(equity_curves):
        if name.startswith("ML-"):
            color_map[name] = ['#2ecc71', '#27ae60', '#1abc9c', '#16a085'][i % 4]
        elif name.startswith("LLM-"):
            color_map[name] = ['#3498db', '#2980b9'][list(equity_curves.keys()).index(name) % 2]

    for name, nav in equity_curves.items():
        lw = 2.5 if name == "Buy & Hold" else 2.0
        ls = '--' if name == "Buy & Hold" else '-'
        ax1.plot(nav.index, nav.values, label=name, linewidth=lw,
                 linestyle=ls, color=color_map.get(name, None), alpha=0.9)

    ax1.set_ylabel('Normalized NAV', fontsize=12)
    ax1.set_title(f'{symbol} — Equity Curves Comparison (ML + LLM + BuyHold)', fontsize=13)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # 回撤子图
    for name, nav in equity_curves.items():
        dd = (nav - nav.cummax()) / nav.cummax()
        ax2.fill_between(nav.index, dd.values, 0, alpha=0.3,
                         color=color_map.get(name, None))
    ax2.set_ylabel('Drawdown', fontsize=11)
    ax2.grid(True, alpha=0.3)

    # 优化日期标签 — 每季度一个刻度
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    path = FIGURES_DIR / f"fig07_equity_curves_{symbol}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存净值曲线: {path}")


def plot_signal_distribution(ml_signals: dict, llm_signals: dict, symbol: str):
    """信号分布图（图 9）：堆积柱状图展示 buy / bearish / hold 比例。"""
    fig, ax = plt.subplots(figsize=(14, 7))

    categories = []
    buy_data = []
    short_data = []
    hold_data = []

    for name, sig_df in ml_signals.items():
        if sig_df.empty:
            continue
        categories.append(f"ML-{name}")
        sig = sig_df["signal_strength_0_to_1"]
        # ML 信号映射
        buy_pct = (sig >= 0.55).mean()
        short_pct = (sig <= 0.45).mean()
        hold_pct = 1 - buy_pct - short_pct
        buy_data.append(buy_pct * 100)
        short_data.append(short_pct * 100)
        hold_data.append(hold_pct * 100)

    for name, sig_df in llm_signals.items():
        if sig_df.empty:
            continue
        categories.append(f"LLM-{name}")
        sig = sig_df["signal"].str.lower()
        buy_pct = (sig == "buy").mean()
        bearish_pct = sig.isin(["short", "sell"]).mean()
        hold_pct = 1 - buy_pct - bearish_pct
        buy_data.append(buy_pct * 100)
        short_data.append(bearish_pct * 100)
        hold_data.append(hold_pct * 100)

    x = np.arange(len(categories))
    width = 0.6

    ax.bar(x, buy_data, width, label='Buy', color='#2ecc71', edgecolor='white')
    ax.bar(x, hold_data, width, bottom=buy_data, label='Hold/Neutral', color='#95a5a6', edgecolor='white')
    ax.bar(x, short_data, width, bottom=[b+h for b,h in zip(buy_data, hold_data)],
           label='Bearish', color='#e74c3c', edgecolor='white')

    ax.set_ylabel('Signal Distribution (%)', fontsize=12)
    ax.set_title(f'{symbol} Signal Distribution by Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=8)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 105)
    plt.tight_layout()

    path = FIGURES_DIR / f"fig09_signal_distribution_{symbol}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {path}")

    # 保存 CSV
    df = pd.DataFrame({
        'model': categories, 'buy_pct': buy_data,
        'hold_pct': hold_data, 'bearish_pct': short_data
    })
    df.to_csv(TABLES_DIR / "table7_signal_distribution.csv", index=False)


def plot_confidence_histogram(llm_signals: dict, symbol: str):
    """置信度分布直方图（图 11）。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, (name, sig_df) in enumerate(sorted(llm_signals.items())):
        if sig_df.empty or i >= 6:
            continue
        ax = axes[i]
        conf = sig_df["confidence"].values
        ax.hist(conf, bins=20, alpha=0.7, edgecolor='white', color='#3498db')
        ax.set_title(f"{name}\nmean={conf.mean():.3f}, std={conf.std():.3f}", fontsize=10)
        ax.set_xlim(0.3, 1.0)
        ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='0.5')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    # 隐藏多余的子图
    for i in range(len(llm_signals), 6):
        axes[i].set_visible(False)

    plt.suptitle(f'{symbol} LLM Confidence Distribution', fontsize=14, y=1.02)
    plt.tight_layout()

    path = FIGURES_DIR / f"fig11_confidence_histogram_{symbol}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {path}")


def plot_model_version_timeline(symbol: str):
    """模型版本时间线图（图 13）：从 versions.jsonl 绘制 Walk-Forward 重训窗口。"""
    fig, ax = plt.subplots(figsize=(14, 6))

    model_dir = Path(f"models")
    colors = {'logistic_regression': '#2ecc71', 'lstm': '#3498db', 'lightgbm': '#e67e22'}
    symbol_prefix = "csi300" if symbol == "CSI300" else "qqq"

    y_pos = 0
    for model_type, color in colors.items():
        ver_path = model_dir / f"{symbol_prefix}_{model_type}" / "versions.jsonl"
        if not ver_path.exists():
            continue

        versions = []
        with open(ver_path, 'r') as f:
            for line in f:
                versions.append(json.loads(line))

        for v in versions:
            train_start = pd.Timestamp(v["train_start"])
            train_end = pd.Timestamp(v["train_end"])
            ax.barh(y_pos, (train_end - train_start).days,
                   left=matplotlib.dates.date2num(train_start),
                   height=0.6, color=color, alpha=0.7, edgecolor='white')

        y_pos += 1

    # 设置日期格式
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')

    ax.set_yticks(range(len(colors)))
    ax.set_yticklabels(['Logistic Regression', 'LSTM', 'LightGBM'])
    ax.set_title(f'{symbol} Walk-Forward Retraining Timeline', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Model')
    plt.tight_layout()

    path = FIGURES_DIR / f"fig13_model_version_timeline_{symbol}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {path}")

    # 保存示例版本索引
    if versions:
        sample = pd.DataFrame(versions[:5])
        sample.to_csv(TABLES_DIR / "table12_model_versions_sample.csv", index=False)


def plot_underwater_comparison(all_results: dict, symbol: str):
    """回撤对比图（图 12）：所有策略的回撤曲线。"""
    from src.orchestrator.signal_converter import SignalConverter
    from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

    cfg = MARKET_CONFIG[symbol]
    ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])

    fig, ax = plt.subplots(figsize=(14, 6))
    colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    color_idx = 0

    # ML 策略
    ml_signals = load_all_ml_signals(symbol)
    for name, sig_df in ml_signals.items():
        if sig_df.empty or name == "ENSEMBLE":
            continue
        sig_df = sig_df[sig_df["timestamp"] >= pd.Timestamp(cfg["start"])].copy()
        sig_df = sig_df[sig_df["timestamp"] <= pd.Timestamp(cfg["end"])].copy()
        positions = SignalConverter.ml_signals_to_positions(
            sig_df, ohlcv_dates=ohlcv.index,
            ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
            signal_threshold=cfg["signal_threshold"],
        )
        if not positions:
            continue
        engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
        engine.add_data(ohlcv, name=symbol)
        engine.add_strategy(DualTrackStrategy, target_positions=positions,
                           printlog=False, allow_short=cfg["allow_short"])
        result = engine.run()
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            eq = result.equity_curve
            if 'value' in eq.columns:
                cummax = eq["value"].cummax()
                drawdown = (eq["value"] - cummax) / cummax
                ax.plot(range(len(drawdown)), drawdown.values, color=colors_list[color_idx % len(colors_list)],
                       linewidth=1.2, label=f"ML-{name} ({result.max_drawdown:.1%})")
                color_idx += 1

    # LLM 策略（取最佳 3 个）
    llm_signals = load_all_llm_signals(symbol)
    for name, sig_df in sorted(llm_signals.items())[:3]:
        if sig_df.empty:
            continue
        sig_df = sig_df[sig_df["timestamp"] >= pd.Timestamp(cfg["start"])].copy()
        sig_df = sig_df[sig_df["timestamp"] <= pd.Timestamp(cfg["end"])].copy()
        positions = SignalConverter.llm_signals_to_positions(
            sig_df.copy(), ohlcv_dates=ohlcv.index,
            confidence_mode="linear", ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
        )
        if not positions:
            continue
        engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
        engine.add_data(ohlcv, name=symbol)
        engine.add_strategy(DualTrackStrategy, target_positions=positions,
                           printlog=False, allow_short=cfg["allow_short"])
        result = engine.run()
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            eq = result.equity_curve
            if 'value' in eq.columns:
                cummax = eq["value"].cummax()
                drawdown = (eq["value"] - cummax) / cummax
                ax.plot(range(len(drawdown)), drawdown.values, color='gray',
                       linewidth=0.8, alpha=0.6, label=f"LLM-{name} ({result.max_drawdown:.1%})")

    # 买入持有
    cummax = ohlcv["close"].cummax()
    drawdown = (ohlcv["close"] - cummax) / cummax
    ax.plot(range(len(drawdown)), drawdown.values, color='black', linewidth=2.0,
           linestyle='--', label=f"Buy&Hold ({drawdown.min():.1%})")

    ax.set_ylabel('Drawdown', fontsize=12)
    ax.set_title(f'{symbol} Strategy Drawdown Comparison', fontsize=14)
    ax.legend(fontsize=7, loc='lower left')
    ax.set_ylim(min(drawdown.min() - 0.05, -0.05), 0.02)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    plt.tight_layout()

    path = FIGURES_DIR / f"fig12_drawdown_comparison_{symbol}.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 保存: {path}")


def generate_feature_importance_plot(symbol: str):
    """LightGBM 特征重要性图（图 4）。"""
    import joblib
    import matplotlib.pyplot as plt
    import numpy as np

    model_dir = Path(f"models/{symbol.lower()}_lightgbm")
    if not model_dir.exists():
        print(f"  ⚠️ 模型目录不存在: {model_dir}")
        return

    # 加载最新模型
    latest_meta = model_dir / "metadata_latest.json"
    if not latest_meta.exists():
        print(f"  ⚠️ metadata_latest.json 不存在")
        return

    with open(latest_meta, 'r') as f:
        meta = json.load(f)

    # 找到最新的模型文件
    model_files = list(model_dir.glob("model_*.pkl"))
    if not model_files:
        print(f"  ⚠️ 无模型文件")
        return
    model_path = sorted(model_files)[-1]

    model = joblib.load(model_path)
    inner = model.model if hasattr(model, 'model') else model
    feature_names = meta.get('feature_names', [f'f{i}' for i in range(len(inner.feature_importances_))])

    if hasattr(inner, 'feature_importances_'):
        imp = inner.feature_importances_
        sorted_idx = np.argsort(imp)[::-1]
        top_n = min(18, len(imp))

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = range(top_n)
        ax.barh(y_pos, [imp[i] for i in sorted_idx[:top_n]], color='#3498db', edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in sorted_idx[:top_n]], fontsize=9)
        ax.set_xlabel('Feature Importance (Gain)')
        ax.set_title(f'LightGBM Feature Importance ({symbol})')
        ax.invert_yaxis()
        plt.tight_layout()

        save_path = FIGURES_DIR / f"fig04_lgb_feature_importance_{symbol}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 保存: {save_path}")

        # 保存特征重要性表
        feat_imp = pd.DataFrame({
            'feature': [feature_names[i] for i in sorted_idx],
            'importance': [imp[i] for i in sorted_idx],
            'rank': range(1, len(imp) + 1)
        })
        feat_imp.to_csv(TABLES_DIR / f"feature_importance_{symbol}.csv", index=False)


# ============================================================================
# D. 分析输出层
# ============================================================================

def generate_full_comparison_table(ml_results: dict, llm_results: dict, symbol: str):
    """ML vs LLM vs 买入持有总对比表。"""
    all_rows = []
    for k, v in ml_results.items():
        all_rows.append(v)
    for k, v in llm_results.items():
        if k == "BuyHold":
            continue  # 避免重复
        all_rows.append(v)

    if "BuyHold" in ml_results:
        all_rows.append(ml_results["BuyHold"])

    df = pd.DataFrame(all_rows)
    df = df[['strategy', 'symbol', 'total_return', 'annual_return',
             'sharpe_ratio', 'max_drawdown', 'turnover', 'final_value']]
    df = df.drop_duplicates(subset=['strategy', 'symbol'])
    df = df.sort_values(['strategy'], key=lambda x: x.str.lower())

    path = TABLES_DIR / f"table_comparison_{symbol}.csv"
    df.to_csv(path, index=False)
    print(f"\n总对比表 ({symbol}):")
    print(df.to_string(index=False))
    return df


def run_sensitivity_analysis(symbol: str):
    """参数敏感性分析：EMA alpha, decay_rate, confidence threshold。"""
    from src.orchestrator.signal_converter import SignalConverter
    from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

    cfg = MARKET_CONFIG[symbol]
    ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])

    # 使用代表性 LLM 信号做敏感性分析
    llm_signals = load_all_llm_signals(symbol)
    if not llm_signals:
        print("  ⚠️ 无 LLM 信号，跳过敏感性分析")
        return

    # 选择第一个模型作为代表
    rep_name = list(llm_signals.keys())[0]
    rep_signals = llm_signals[rep_name].copy()
    rep_signals = rep_signals[rep_signals["timestamp"] >= pd.Timestamp(cfg["start"])].copy()
    rep_signals = rep_signals[rep_signals["timestamp"] <= pd.Timestamp(cfg["end"])].copy()

    results = []

    # EMA alpha 敏感性
    for alpha in [0.20, 0.30, 0.40, 0.50, 0.60]:
        positions = SignalConverter.llm_signals_to_positions(
            rep_signals.copy(), ohlcv_dates=ohlcv.index,
            confidence_mode="linear", ema_alpha=alpha, decay_rate=cfg["decay_rate"],
        )
        if positions:
            turnover = SignalConverter._compute_turnover(positions, ohlcv.index, symbol)
            engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
            engine.add_data(ohlcv, name=symbol)
            engine.add_strategy(DualTrackStrategy, target_positions=positions,
                               printlog=False, allow_short=cfg["allow_short"])
            r = engine.run()
            results.append({
                "param": "ema_alpha", "value": alpha,
                "total_return": r.total_return, "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown, "turnover": turnover,
            })

    # decay_rate 敏感性
    for decay in [0.60, 0.70, 0.80, 0.90]:
        positions = SignalConverter.llm_signals_to_positions(
            rep_signals.copy(), ohlcv_dates=ohlcv.index,
            confidence_mode="linear", ema_alpha=cfg["ema_alpha"], decay_rate=decay,
        )
        if positions:
            turnover = SignalConverter._compute_turnover(positions, ohlcv.index, symbol)
            engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
            engine.add_data(ohlcv, name=symbol)
            engine.add_strategy(DualTrackStrategy, target_positions=positions,
                               printlog=False, allow_short=cfg["allow_short"])
            r = engine.run()
            results.append({
                "param": "decay_rate", "value": decay,
                "total_return": r.total_return, "sharpe_ratio": r.sharpe_ratio,
                "max_drawdown": r.max_drawdown, "turnover": turnover,
            })

    # signal_threshold 敏感性
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        # ML 信号阈值敏感性
        ml_signals = load_all_ml_signals(symbol)
        if ml_signals:
            ens = ml_signals.get("ENSEMBLE")
            if ens is not None and not ens.empty:
                ens = ens[ens["timestamp"] >= pd.Timestamp(cfg["start"])].copy()
                ens = ens[ens["timestamp"] <= pd.Timestamp(cfg["end"])].copy()
                positions = SignalConverter.ml_signals_to_positions(
                    ens, ohlcv_dates=ohlcv.index,
                    ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
                    signal_threshold=thresh,
                )
                if positions:
                    turnover = SignalConverter._compute_turnover(positions, ohlcv.index, symbol)
                    engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
                    engine.add_data(ohlcv, name=symbol)
                    engine.add_strategy(DualTrackStrategy, target_positions=positions,
                                       printlog=False, allow_short=cfg["allow_short"])
                    r = engine.run()
                    results.append({
                        "param": "signal_threshold", "value": thresh,
                        "total_return": r.total_return, "sharpe_ratio": r.sharpe_ratio,
                        "max_drawdown": r.max_drawdown, "turnover": turnover,
                    })

    df = pd.DataFrame(results)
    df.to_csv(ANALYSIS_DIR / "sensitivity_analysis.csv", index=False)
    print(f"\n敏感性分析结果:")
    print(df.to_string(index=False))

    # 绘制热力图
    for param in df["param"].unique():
        sub = df[df["param"] == param]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sub["value"].values, sub["sharpe_ratio"].values, 'o-', color='#2980b9')
        ax.set_xlabel(param)
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title(f'Sensitivity: {param} ({symbol})')
        ax.grid(True, alpha=0.3)
        path = FIGURES_DIR / f"sensitivity_{param}_{symbol}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 保存: {path}")


def analyze_ml_failure_modes(symbol: str):
    """ML 失败模式分析。"""
    from src.models.ml_track.features import FeatureEngineer

    ohlcv = load_ohlcv(symbol)
    fe = FeatureEngineer(feature_subset="core")
    features_df = fe.compute_all_features(ohlcv)

    if features_df.empty:
        return

    # 计算每日实际收益
    features_df["actual_return"] = features_df["close"].pct_change()

    # 分析 ML 信号错误
    ml_signals = load_all_ml_signals(symbol)
    if not ml_signals:
        return

    report_lines = []
    report_lines.append(f"# ML 失败模式分析 — {symbol}\n")

    for name, sig_df in ml_signals.items():
        if sig_df.empty or name == "ENSEMBLE":
            continue
        sig_df = sig_df[sig_df["timestamp"].isin(features_df.index)].copy()
        signals = sig_df.set_index("timestamp")["signal_strength_0_to_1"]
        actual = features_df["actual_return"]

        # 预测方向 vs 实际方向
        aligned = pd.DataFrame({"signal": signals, "actual": actual}).dropna()
        pred_up = (aligned["signal"] >= 0.55).astype(int)
        actual_up = (aligned["actual"] > 0).astype(int)
        accuracy = (pred_up == actual_up).mean()

        # 找到最大错误预测日
        aligned["error"] = aligned["signal"] - 0.5  # 信号偏离中性
        aligned["actual_mag"] = aligned["actual"].abs()
        worst_day = aligned.loc[aligned["error"].abs().idxmax()]

        report_lines.append(f"## {name}")
        report_lines.append(f"- 方向准确率: {accuracy:.2%}")
        report_lines.append(f"- 最大信号偏离日: {worst_day.name.date()}, "
                          f"signal={worst_day['signal']:.4f}, actual_return={worst_day['actual']:.2%}")
        report_lines.append("")

    with open(ANALYSIS_DIR / "ml_failure_modes.md", "w") as f:
        f.write("\n".join(report_lines))
    print(f"  ✅ 保存: {ANALYSIS_DIR / 'ml_failure_modes.md'}")


def analyze_llm_failure_modes(symbol: str):
    """LLM 失败模式分析：过度解读、保守主义偏差。"""
    llm_signals = load_all_llm_signals(symbol)
    if not llm_signals:
        return

    ohlcv = load_ohlcv(symbol)
    ohlcv["actual_return"] = ohlcv["close"].pct_change()

    report_lines = []
    report_lines.append(f"# LLM 失败模式分析 — {symbol}\n")

    for name, sig_df in sorted(llm_signals.items()):
        if sig_df.empty:
            continue

        # 对齐日期
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
        merged = sig_df.merge(ohlcv[["actual_return"]],
                              left_on="timestamp", right_index=True, how="inner")
        if merged.empty:
            continue

        # 保守主义偏差：统计 buy 信号中 confidence 的分布
        buy_signals = merged[merged["signal"].str.lower() == "buy"]
        if not buy_signals.empty:
            high_conf_buy = (buy_signals["confidence"] >= 0.80).sum()
            med_conf_buy = ((buy_signals["confidence"] >= 0.60) & (buy_signals["confidence"] < 0.80)).sum()
            low_conf_buy = (buy_signals["confidence"] < 0.60).sum()
            report_lines.append(f"## {name}")
            report_lines.append(f"- Buy 信号总数: {len(buy_signals)}")
            report_lines.append(f"  - 高置信度 (≥0.80): {high_conf_buy} ({high_conf_buy/len(buy_signals):.1%})")
            report_lines.append(f"  - 中置信度 (0.60-0.80): {med_conf_buy} ({med_conf_buy/len(buy_signals):.1%})")
            report_lines.append(f"  - 低置信度 (<0.60): {low_conf_buy} ({low_conf_buy/len(buy_signals):.1%})")

            # 置信度-实际收益相关性
            corr = merged["confidence"].corr(merged["actual_return"])
            report_lines.append(f"- 置信度与实际收益相关性: {corr:.4f}")

            # 找到过度解读案例：LLM 极度悲观但实际上涨
            bearish = merged[
                (merged["signal"].str.lower().isin(["short", "sell"])) &
                (merged["confidence"] >= 0.70) &
                (merged["actual_return"] > 0)
            ]
            if not bearish.empty:
                report_lines.append(f"- 过度解读案例 (看空但实际上涨): {len(bearish)} 次")
                top = bearish.nlargest(3, "actual_return")
                for _, row in top.iterrows():
                    report_lines.append(
                        f"  - {row['timestamp'].date()}: signal={row['signal']}, "
                        f"conf={row['confidence']:.2f}, actual={row['actual_return']:.2%}")
            report_lines.append("")

    with open(ANALYSIS_DIR / "llm_failure_modes.md", "w") as f:
        f.write("\n".join(report_lines))
    print(f"  ✅ 保存: {ANALYSIS_DIR / 'llm_failure_modes.md'}")


def analyze_llm_missed_opportunities(symbol: str):
    """LLM 踏空分析：统计 LLM 在上涨日中的 hold/观望比例。"""
    llm_signals = load_all_llm_signals(symbol)
    if not llm_signals:
        return

    ohlcv = load_ohlcv(symbol)
    ohlcv["actual_return"] = ohlcv["close"].pct_change()

    all_records = []
    for name, sig_df in sorted(llm_signals.items()):
        if sig_df.empty:
            continue
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
        merged = sig_df.merge(ohlcv[["actual_return"]],
                              left_on="timestamp", right_index=True, how="inner")
        if merged.empty:
            continue

        up_days = merged[merged["actual_return"] > 0]
        if len(up_days) == 0:
            continue

        missed = up_days[up_days["signal"].str.lower().isin(["hold", "neutral"])]
        all_records.append({
            "model": name,
            "total_up_days": len(up_days),
            "missed_days": len(missed),
            "miss_rate": len(missed) / len(up_days),
            "avg_return_on_missed": missed["actual_return"].mean() if len(missed) > 0 else 0,
            "avg_confidence_on_up": up_days["confidence"].mean(),
        })

    df = pd.DataFrame(all_records)
    df.to_csv(ANALYSIS_DIR / "llm_missed_opportunities.csv", index=False)
    print(f"\nLLM 踏空分析 ({symbol}):")
    print(df.to_string(index=False))


def analyze_beta_drift_effect(symbol: str):
    """Beta Drift 效应分析。"""
    ohlcv = load_ohlcv(symbol)
    total_return = ohlcv["close"].iloc[-1] / ohlcv["close"].iloc[0] - 1

    # 统计做多 vs 做空信号收益差异
    llm_signals = load_all_llm_signals(symbol)
    if not llm_signals:
        return

    ohlcv["actual_return"] = ohlcv["close"].pct_change()

    records = []
    for name, sig_df in sorted(llm_signals.items()):
        if sig_df.empty:
            continue
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"])
        merged = sig_df.merge(ohlcv[["actual_return"]],
                              left_on="timestamp", right_index=True, how="inner")
        if merged.empty:
            continue

        buy_ret = merged[merged["signal"].str.lower() == "buy"]["actual_return"].mean()
        short_ret = merged[merged["signal"].str.lower() == "short"]["actual_return"].mean()
        hold_ret = merged[merged["signal"].str.lower().isin(["hold", "neutral"])]["actual_return"].mean()

        records.append({
            "model": name,
            "market_beta": total_return,
            "avg_return_on_buy": buy_ret,
            "avg_return_on_short": short_ret,
            "avg_return_on_hold": hold_ret,
        })

    df = pd.DataFrame(records)
    df.to_csv(ANALYSIS_DIR / "beta_drift_analysis.csv", index=False)
    print(f"\nBeta Drift 分析 ({symbol}):")
    print(df.to_string(index=False))


def run_statistical_tests(symbol: str, ml_results: dict, llm_results: dict):
    """运行统计显著性检验：Jobson-Korkie、Bootstrap、卡方等。"""
    from scipy import stats as sp_stats
    from scripts.statistical_tests import (
        jobson_korkie_test, bootstrap_sharpe_difference,
        bootstrap_max_drawdown_difference,
    )
    from src.orchestrator.signal_converter import SignalConverter
    from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

    cfg = MARKET_CONFIG[symbol]
    ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])

    def get_strategy_returns(name, sig_df, is_llm=False):
        """从信号计算策略日收益率。"""
        if sig_df.empty:
            return None
        sig_df = sig_df[sig_df["timestamp"] >= pd.Timestamp(cfg["start"])].copy()
        sig_df = sig_df[sig_df["timestamp"] <= pd.Timestamp(cfg["end"])].copy()
        if is_llm:
            positions = SignalConverter.llm_signals_to_positions(
                sig_df.copy(), ohlcv_dates=ohlcv.index,
                confidence_mode="linear",
                ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
            )
        else:
            positions = SignalConverter.ml_signals_to_positions(
                sig_df, ohlcv_dates=ohlcv.index,
                ema_alpha=cfg["ema_alpha"], decay_rate=cfg["decay_rate"],
                signal_threshold=cfg["signal_threshold"],
            )
        if not positions:
            return None
        engine = BacktestEngine(initial_cash=1000000, commission=cfg["commission"])
        engine.add_data(ohlcv, name=symbol)
        engine.add_strategy(DualTrackStrategy, target_positions=positions,
                           printlog=False, allow_short=cfg["allow_short"])
        result = engine.run()
        if hasattr(result, 'equity_curve') and result.equity_curve is not None:
            eq = result.equity_curve
            if 'value' in eq.columns:
                return eq["value"].pct_change().dropna().values
        return None

    test_results = []
    ml_signals = load_all_ml_signals(symbol)
    llm_signals = load_all_llm_signals(symbol)

    # 找最佳 ML 和最佳 LLM
    best_ml_name = None
    best_ml_sharpe = -999
    for n, v in ml_results.items():
        if n != "BuyHold" and v["sharpe_ratio"] > best_ml_sharpe:
            best_ml_sharpe = v["sharpe_ratio"]
            best_ml_name = n

    best_llm_name = None
    best_llm_sharpe = -999
    for n, v in llm_results.items():
        if n != "BuyHold" and v["sharpe_ratio"] > best_llm_sharpe:
            best_llm_sharpe = v["sharpe_ratio"]
            best_llm_name = n

    if not best_ml_name or not best_llm_name:
        print("  ⚠️ 无法找到最佳策略，跳过统计检验")
        return

    # 获取日收益率
    ret_ml = get_strategy_returns(best_ml_name, ml_signals[best_ml_name], is_llm=False)
    ret_llm = get_strategy_returns(best_llm_name, llm_signals[best_llm_name], is_llm=True)
    if ret_ml is None or ret_llm is None:
        print("  ⚠️ 无法计算日收益率，跳过统计检验")
        return

    min_len = min(len(ret_ml), len(ret_llm))

    # 1. Jobson-Korkie 检验
    try:
        stat, p_value = jobson_korkie_test(ret_ml[:min_len], ret_llm[:min_len])
        test_results.append({
            "test": "Jobson-Korkie (Sharpe difference)",
            "strategy_a": f"ML-{best_ml_name}",
            "strategy_b": f"LLM-{best_llm_name}",
            "statistic": f"{stat:.4f}",
            "p_value": f"{p_value:.4f}",
            "significant": "Yes" if p_value < 0.05 else "No",
            "interpretation": "Sharpe significantly different" if p_value < 0.05 else "No significant Sharpe difference",
        })
    except Exception as e:
        test_results.append({"test": "Jobson-Korkie", "error": str(e)})

    # 2. Bootstrap Sharpe 差异
    try:
        mean_diff, std_diff, ci = bootstrap_sharpe_difference(ret_ml[:min_len], ret_llm[:min_len], n_bootstrap=5000)
        test_results.append({
            "test": "Bootstrap Sharpe CI (95%)",
            "strategy_a": f"ML-{best_ml_name}",
            "strategy_b": f"LLM-{best_llm_name}",
            "mean_diff": f"{mean_diff:.4f}",
            "ci_lower": f"{ci[0]:.4f}",
            "ci_upper": f"{ci[1]:.4f}",
            "significant": "Yes" if ci[0] > 0 or ci[1] < 0 else "No",
            "interpretation": f"CI for Sharpe diff: [{ci[0]:.4f}, {ci[1]:.4f}]",
        })
    except Exception as e:
        test_results.append({"test": "Bootstrap Sharpe", "error": str(e)})

    # 3. Bootstrap 最大回撤差异
    try:
        mean_diff, std_diff, ci = bootstrap_max_drawdown_difference(ret_ml[:min_len], ret_llm[:min_len], n_bootstrap=5000)
        test_results.append({
            "test": "Bootstrap MaxDrawdown CI (95%)",
            "strategy_a": f"ML-{best_ml_name}",
            "strategy_b": f"LLM-{best_llm_name}",
            "mean_diff": f"{mean_diff:.4f}",
            "ci_lower": f"{ci[0]:.4f}",
            "ci_upper": f"{ci[1]:.4f}",
            "interpretation": f"CI for MDD diff: [{ci[0]:.4f}, {ci[1]:.4f}]",
        })
    except Exception as e:
        test_results.append({"test": "Bootstrap MDD", "error": str(e)})

    # 4. KS 检验
    try:
        ks_stat, p_value = sp_stats.ks_2samp(ret_ml[:min_len], ret_llm[:min_len])
        test_results.append({
            "test": "Kolmogorov-Smirnov (return distribution)",
            "strategy_a": f"ML-{best_ml_name}",
            "strategy_b": f"LLM-{best_llm_name}",
            "statistic": f"{ks_stat:.4f}",
            "p_value": f"{p_value:.4f}",
            "significant": "Yes" if p_value < 0.05 else "No",
            "interpretation": "Return distributions differ" if p_value < 0.05 else "Return distributions similar",
        })
    except Exception as e:
        test_results.append({"test": "KS test", "error": str(e)})

    # 5. 卡方检验（胜率）
    try:
        wins_a = int((ret_ml[:min_len] > 0).sum())
        losses_a = int((ret_ml[:min_len] <= 0).sum())
        wins_b = int((ret_llm[:min_len] > 0).sum())
        losses_b = int((ret_llm[:min_len] <= 0).sum())
        contingency = np.array([[wins_a, losses_a], [wins_b, losses_b]])
        chi2, p_value, dof, expected = sp_stats.chi2_contingency(contingency)
        test_results.append({
            "test": "Chi-square (win rate)",
            "strategy_a": f"ML-{best_ml_name}",
            "strategy_b": f"LLM-{best_llm_name}",
            "statistic": f"{chi2:.4f}",
            "p_value": f"{p_value:.4f}",
            "significant": "Yes" if p_value < 0.05 else "No",
            "interpretation": "Win rates differ" if p_value < 0.05 else "Win rates similar",
        })
    except Exception as e:
        test_results.append({"test": "Chi-square", "error": str(e)})

    df = pd.DataFrame(test_results)
    df.to_csv(TABLES_DIR / "table11_statistical_tests.csv", index=False)
    print(f"\n统计显著性检验 ({symbol}):")
    print(df.to_string(index=False))



# ============================================================================
# E. 主流程
# ============================================================================

def run_symbol_analysis(symbol: str):
    """运行单个市场的完整分析。"""
    cfg = MARKET_CONFIG[symbol]
    print(f"\n{'='*70}")
    print(f"  开始分析: {symbol} ({cfg['start']} ~ {cfg['end']})")
    print(f"{'='*70}")

    # 1. 数据集统计
    print("\n[1/10] 数据集统计...")
    stats = generate_dataset_stats()

    # 2. ML 回测
    print("\n[2/10] ML 轨道回测...")
    ml_results = run_ml_backtest_for_symbol(symbol)

    # 3. LLM 回测
    print("\n[3/10] LLM 轨道回测...")
    llm_results = run_llm_backtest_for_symbol(symbol)

    # 3b. 净值曲线图（ML + LLM + BuyHold）
    print("\n[3b/12] 净值曲线图...")
    plot_equity_curves_with_llm(ml_results, llm_results, symbol)

    # 4. 信号分布图
    print("\n[4/10] 信号分布图...")
    ml_signals = load_all_ml_signals(symbol)
    llm_signals = load_all_llm_signals(symbol)
    plot_signal_distribution(ml_signals, llm_signals, symbol)

    # 5. 置信度直方图
    print("\n[5/10] 置信度直方图...")
    plot_confidence_histogram(llm_signals, symbol)

    # 6. 模型版本时间线
    print("\n[6/10] 模型版本时间线...")
    plot_model_version_timeline(symbol)

    # 7. 总对比表
    print("\n[7/12] 总对比表...")
    generate_full_comparison_table(ml_results, llm_results, symbol)

    # 8. 特征重要性
    print("\n[8/12] LightGBM 特征重要性...")
    generate_feature_importance_plot(symbol)

    # 9. 敏感性分析
    print("\n[9/12] 敏感性分析...")
    run_sensitivity_analysis(symbol)

    # 10. 回撤对比图
    print("\n[10/12] 回撤对比图...")
    plot_underwater_comparison({}, symbol)

    # 11. 统计显著性
    print("\n[11/12] 统计显著性检验...")
    run_statistical_tests(symbol, ml_results, llm_results)

    # 11. 失败模式分析
    print("\n[11/12] 失败模式分析...")
    analyze_ml_failure_modes(symbol)
    analyze_llm_failure_modes(symbol)

    # 12. 踏空与 Beta Drift 分析
    print("\n[12/12] 踏空与 Beta Drift 分析...")
    analyze_llm_missed_opportunities(symbol)
    analyze_beta_drift_effect(symbol)

    print(f"\n{'='*70}")
    print(f"  {symbol} 分析完成！输出目录: {OUTPUT_DIR}")
    print(f"{'='*70}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Chapter 3 实验结果分析")
    parser.add_argument("--symbol", choices=["CSI300", "QQQ", "ALL"], default="ALL",
                       help="分析的市场")
    args = parser.parse_args()

    if args.symbol == "ALL":
        run_symbol_analysis("CSI300")
        run_symbol_analysis("QQQ")
    else:
        run_symbol_analysis(args.symbol)


if __name__ == "__main__":
    main()
