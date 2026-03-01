"""
实验对比分析模块。

核心功能：对比 ML Track 和 LLM Track 的实验结果。

这是 DualTrack 框架的核心模块，用于回答研究问题：
- RQ1: ML 和 LLM 谁的收益更高？
- RQ2: 谁更稳健？
- RQ3: 成本效益比如何？

注意：此模块只进行对比分析，不涉及信号融合。
融合功能在 fusion_engine.py 中，仅作为可选探索。
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class ComparisonResult:
    """对比分析结果。"""

    # 核心问题回答
    sharpe_winner: str  # "ML" 或 "LLM"
    sharpe_diff: float
    drawdown_winner: str
    return_winner: str

    # 详细指标对比
    ml_metrics: Dict[str, Any]
    llm_metrics: Dict[str, Any]

    # 结论
    conclusion: str
    recommendation: str


def compare_experiments(
    ml_result: Dict[str, Any],
    llm_result: Dict[str, Any],
) -> ComparisonResult:
    """
    对比 ML Track 和 LLM Track 的实验结果。

    Args:
        ml_result: ML Track 实验结果
        llm_result: LLM Track 实验结果

    Returns:
        对比分析结果
    """
    # 提取金融指标
    ml_financial = ml_result.get("financial_metrics", {})
    llm_financial = llm_result.get("financial_metrics", {})

    # 回答核心问题
    ml_sharpe = ml_financial.get("sharpe_ratio", 0.0)
    llm_sharpe = llm_financial.get("sharpe_ratio", 0.0)
    sharpe_winner = "ML" if ml_sharpe > llm_sharpe else "LLM"
    sharpe_diff = abs(ml_sharpe - llm_sharpe)

    ml_dd = ml_financial.get("max_drawdown", 0.0)
    llm_dd = llm_financial.get("max_drawdown", 0.0)
    drawdown_winner = "ML" if ml_dd < llm_dd else "LLM"

    ml_return = ml_financial.get("total_return", 0.0)
    llm_return = llm_financial.get("total_return", 0.0)
    return_winner = "ML" if ml_return > llm_return else "LLM"

    # 生成结论
    conclusion = _generate_conclusion(
        sharpe_winner, drawdown_winner, return_winner,
        ml_financial, llm_financial
    )

    recommendation = _generate_recommendation(
        sharpe_winner, drawdown_winner, return_winner
    )

    return ComparisonResult(
        sharpe_winner=sharpe_winner,
        sharpe_diff=sharpe_diff,
        drawdown_winner=drawdown_winner,
        return_winner=return_winner,
        ml_metrics=ml_financial,
        llm_metrics=llm_financial,
        conclusion=conclusion,
        recommendation=recommendation,
    )


def _generate_conclusion(
    sharpe_winner: str,
    drawdown_winner: str,
    return_winner: str,
    ml_metrics: Dict[str, Any],
    llm_metrics: Dict[str, Any],
) -> str:
    """生成对比结论。"""
    lines = [
        "=" * 70,
        "  DualTrack 对比实验结论",
        "=" * 70,
        "",
        "【核心问题回答】",
        "",
        f"Q1: 谁的收益更高？",
        f"  ML Track Sharpe:  {ml_metrics.get('sharpe_ratio', 0):.4f}",
        f"  LLM Track Sharpe: {llm_metrics.get('sharpe_ratio', 0):.4f}",
        f"  ✅ Winner: {sharpe_winner}",
        "",
        f"Q2: 谁更稳健？",
        f"  ML Track MaxDD:  {ml_metrics.get('max_drawdown', 0):.2%}",
        f"  LLM Track MaxDD: {llm_metrics.get('max_drawdown', 0):.2%}",
        f"  ✅ Winner: {drawdown_winner}",
        "",
        f"Q3: 收益率对比？",
        f"  ML Track Return:  {ml_metrics.get('total_return', 0):.2%}",
        f"  LLM Track Return: {llm_metrics.get('total_return', 0):.2%}",
        f"  ✅ Winner: {return_winner}",
        "",
    ]

    return "\n".join(lines)


def _generate_recommendation(
    sharpe_winner: str,
    drawdown_winner: str,
    return_winner: str,
) -> str:
    """生成实践建议。"""
    if sharpe_winner == "ML" and drawdown_winner == "ML":
        return "→ 推荐使用 ML Track（收益高且稳健）"
    elif sharpe_winner == "LLM" and drawdown_winner == "LLM":
        return "→ 推荐使用 LLM Track（收益高且稳健）"
    elif sharpe_winner == "ML" and drawdown_winner == "LLM":
        return "→ 根据风险偏好选择：追求收益选 ML，控制风险选 LLM"
    elif sharpe_winner == "LLM" and drawdown_winner == "ML":
        return "→ 根据风险偏好选择：追求收益选 LLM，控制风险选 ML"
    else:
        return "→ 两个轨道各有优势，建议根据具体场景选择"


def print_comparison_table(comparison: ComparisonResult) -> None:
    """打印对比结果表格。"""
    print("\n" + "=" * 70)
    print("  DualTrack 对比实验结果")
    print("=" * 70)
    print()
    print(comparison.conclusion)
    print()
    print("【实践建议】")
    print(f"  {comparison.recommendation}")
    print("=" * 70)


def signals_to_positions(
    signals: pd.DataFrame,
    track_type: str = "ml",
    method: str = "average"
) -> Dict[datetime, Dict[str, float]]:
    """
    将信号转换为目标仓位（不融合，只转换格式）。

    Args:
        signals: 信号DataFrame（ML或LLM）
        track_type: 轨道类型 ("ml" 或 "llm")
        method: 转换方法
            - "average": 平均多个模型的信号（ML Track）
            - "confidence_weighted": 置信度加权（LLM Track）

    Returns:
        目标仓位字典 {datetime: {symbol: weight}}
    """
    positions = {}

    if track_type == "ml":
        # ML 信号处理
        if "signal_strength_0_to_1" in signals.columns:
            grouped = signals.groupby("timestamp")
            for timestamp, group in grouped:
                avg_signal = group["signal_strength_0_to_1"].mean()
                weight = (avg_signal - 0.5) * 2  # 0-1 → -1到1
                symbol = group["symbol"].iloc[0] if "symbol" in group.columns else "CSI300"
                positions[timestamp] = {symbol: weight}

    elif track_type == "llm":
        # LLM 信号处理
        signal_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}

        if "llm_signal" in signals.columns:
            for _, row in signals.iterrows():
                timestamp = row.get("timestamp", datetime.now())
                symbol = row.get("symbol", "CSI300")
                signal = signal_map.get(row["llm_signal"], 0.0)

                if method == "confidence_weighted":
                    confidence = row.get("confidence", 0.5)
                    weight = signal * confidence
                else:
                    weight = signal

                positions[timestamp] = {symbol: weight}

    return positions
