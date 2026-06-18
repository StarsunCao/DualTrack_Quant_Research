#!/usr/bin/env python
"""Generate market-state figures for Chapter 3."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from scripts.chapter3_analysis import (
    MARKET_CONFIG,
    load_ohlcv,
    load_all_ml_signals,
    load_all_llm_signals,
)
from src.evaluation.market_state_analyzer import MarketStateAnalyzer
from src.execution.bt_engine import BacktestEngine, DualTrackStrategy
from src.orchestrator.signal_converter import SignalConverter


FIGURES_DIR = Path("output/chapter3/figures")
TABLES_DIR = Path("output/chapter3/tables")


def _equity_from_positions(symbol: str, positions: dict) -> pd.DataFrame:
    cfg = MARKET_CONFIG[symbol]
    ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])
    engine = BacktestEngine(initial_cash=1_000_000, commission=cfg["commission"])
    engine.add_data(ohlcv, name=symbol)
    engine.add_strategy(
        DualTrackStrategy,
        target_positions=positions,
        printlog=False,
        allow_short=cfg["allow_short"],
    )
    result = engine.run()
    eq = result.equity_curve.copy()
    if "datetime" in eq.columns:
        eq["datetime"] = pd.to_datetime(eq["datetime"])
        eq = eq.set_index("datetime")
    elif "date" in eq.columns:
        eq["date"] = pd.to_datetime(eq["date"])
        eq = eq.set_index("date")
    if "value" not in eq.columns and "portfolio_value" in eq.columns:
        eq = eq.rename(columns={"portfolio_value": "value"})
    return eq


def generate_qqq_market_state_figures() -> None:
    symbol = "QQQ"
    cfg = MARKET_CONFIG[symbol]
    ohlcv = load_ohlcv(symbol, cfg["start"], cfg["end"])

    ml_signals = load_all_ml_signals(symbol)
    llm_signals = load_all_llm_signals(symbol)

    selected = {}

    for name in ["LR", "LGB", "ENSEMBLE"]:
        sig = ml_signals[name].copy()
        sig = sig[(sig["timestamp"] >= pd.Timestamp(cfg["start"])) & (sig["timestamp"] <= pd.Timestamp(cfg["end"]))]
        selected[f"ML-{name}"] = SignalConverter.ml_signals_to_positions(
            sig,
            ohlcv_dates=ohlcv.index,
            ema_alpha=cfg["ema_alpha"],
            decay_rate=cfg["decay_rate"],
            signal_threshold=cfg["signal_threshold"],
        )

    for name in ["qwen_qwen3.5-397b-a17b", "deepseek-v4-flash"]:
        sig = llm_signals[name].copy()
        sig = sig[(sig["timestamp"] >= pd.Timestamp(cfg["start"])) & (sig["timestamp"] <= pd.Timestamp(cfg["end"]))]
        selected[f"LLM-{name}"] = SignalConverter.llm_signals_to_positions(
            sig,
            ohlcv_dates=ohlcv.index,
            confidence_mode="linear",
            ema_alpha=cfg["ema_alpha"],
            decay_rate=cfg["decay_rate"],
        )

    analyzer = MarketStateAnalyzer()
    analyzer.load_vix_data("data/raw/vix_2015_2024.csv")

    summaries = {}
    rows = []
    for strategy, positions in selected.items():
        equity = _equity_from_positions(symbol, positions)
        summary = analyzer.analyze_strategy(equity, strategy_name=strategy)
        summaries[strategy] = summary
        for state, metrics in summary.state_metrics.items():
            rows.append(
                {
                    "strategy": strategy,
                    "state": state.value,
                    "days": metrics.days,
                    "total_return": metrics.total_return,
                    "sharpe": metrics.sharpe,
                    "max_drawdown": metrics.max_drawdown,
                    "win_rate": metrics.win_rate,
                }
            )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    analyzer.plot_state_heatmap(
        summaries,
        metric="sharpe",
        save_path=str(FIGURES_DIR / "fig10_market_state_heatmap_QQQ.png"),
    )
    analyzer.plot_strategy_comparison(
        summaries,
        save_path=str(FIGURES_DIR / "fig10_market_state_comparison_QQQ.png"),
    )

    pd.DataFrame(rows).to_csv(TABLES_DIR / "table10_market_state_QQQ.csv", index=False)
    print("Saved market-state figures and table for QQQ.")


if __name__ == "__main__":
    generate_qqq_market_state_figures()
