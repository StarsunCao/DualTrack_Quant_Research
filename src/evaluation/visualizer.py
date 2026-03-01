"""
多维度可视化模块。

实现论文所需的各类图表，支持金融和工程指标的可视化对比。

图表类型：
1. 资金曲线对比图：ML Track vs LLM Track 对比
2. 最大回撤热力图：各模型回撤分布
3. 延迟分布箱线图：推理延迟对比

注意：
- 本项目是对比框架，核心是比较 ML 和 LLM 两个独立轨道
- 融合曲线（如果有）仅作为可选探索，用虚线表示
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 默认颜色方案
COLORS = {
    "ml": "#2E86AB",      # 蓝色 - ML
    "llm": "#A23B72",     # 紫红 - LLM
    "fusion": "#F18F01",  # 橙色 - 融合
    "benchmark": "#C73E1D",  # 红色 - 基准
}


# ============================================================================
# 资金曲线对比图
# ============================================================================
def plot_equity_curves(
    equity_curves: Dict[str, pd.DataFrame],
    benchmark: Optional[pd.Series] = None,
    title: str = "Strategy Equity Curves Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show_drawdown: bool = True,
) -> Figure:
    """
    绘制多策略资金曲线对比图。

    Args:
        equity_curves: 策略净值曲线字典，格式为 {策略名: DataFrame}。
            DataFrame 需包含 'nav' 或 'value' 列。
        benchmark: 基准净值曲线。
        title: 图表标题。
        figsize: 图表尺寸。
        save_path: 保存路径。
        show_drawdown: 是否显示回撤阴影。

    Returns:
        matplotlib Figure 对象。
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=figsize,
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    # 颜色映射
    color_map = {
        "ML_Baseline": COLORS["ml"],
        "ML": COLORS["ml"],
        "ML_Track": COLORS["ml"],
        "LLM_Baseline": COLORS["llm"],
        "LLM": COLORS["llm"],
        "LLM_Track": COLORS["llm"],
        "DualTrack_Fusion": COLORS["fusion"],
        "Fusion": COLORS["fusion"],
        "Fused_Track": COLORS["fusion"],
    }

    # 绘制各策略净值曲线
    for name, df in equity_curves.items():
        # 获取净值
        if "nav" in df.columns:
            nav = df["nav"]
        elif "value" in df.columns:
            nav = df["value"] / df["value"].iloc[0]
        else:
            continue

        color = color_map.get(name, None)

        # 判断是否为融合轨道（可选探索）
        is_fusion = "Fusion" in name or "Fused" in name

        # 绘制净值曲线
        # ML/LLM Track 用实线，融合轨道用虚线（可选探索）
        ax1.plot(
            nav.index,
            nav.values,
            label=name + (" (Exploratory)" if is_fusion else ""),
            linewidth=2 if not is_fusion else 1.5,
            linestyle="-" if not is_fusion else "--",
            color=color,
            alpha=1.0 if not is_fusion else 0.7,
        )

        # 绘制回撤阴影
        if show_drawdown:
            cummax = nav.cummax()
            drawdown = (nav - cummax) / cummax
            ax2.fill_between(
                nav.index,
                drawdown.values,
                0,
                alpha=0.3,
                color=color,
            )
            ax2.plot(
                nav.index,
                drawdown.values,
                linewidth=0.5,
                color=color,
                alpha=0.7,
            )

    # 绘制基准
    if benchmark is not None:
        ax1.plot(
            benchmark.index,
            benchmark.values,
            label="Benchmark",
            linewidth=1.5,
            linestyle="--",
            color=COLORS["benchmark"],
            alpha=0.7,
        )

    # 设置标题和标签
    ax1.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax1.set_ylabel("Net Asset Value", fontsize=12)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # 回撤子图
    ax2.set_ylabel("Drawdown", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax2.grid(True, alpha=0.3)

    # 设置 x 轴日期格式
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()

    # 保存图表
    if save_path:
        _save_figure(fig, save_path)

    return fig


# ============================================================================
# 最大回撤热力图
# ============================================================================
def plot_drawdown_heatmap(
    drawdown_data: Union[pd.DataFrame, Dict[str, Dict[str, float]]],
    title: str = "Maximum Drawdown Heatmap",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    cmap: str = "RdYlGn_r",
    annot: bool = True,
    fmt: str = ".2%",
) -> Figure:
    """
    绘制最大回撤热力图。

    Args:
        drawdown_data: 回撤数据。
            可以是 DataFrame（行=策略，列=时期），
            或字典格式 {策略名: {时期: 回撤值}}。
        title: 图表标题。
        figsize: 图表尺寸。
        save_path: 保存路径。
        cmap: 颜色映射。
        annot: 是否显示数值标注。
        fmt: 数值格式。

    Returns:
        matplotlib Figure 对象。
    """
    # 转换为 DataFrame
    if isinstance(drawdown_data, dict):
        df = pd.DataFrame(drawdown_data).T
    else:
        df = drawdown_data

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制热力图
    sns.heatmap(
        df,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Drawdown"},
        ax=ax,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Period", fontsize=12)
    ax.set_ylabel("Strategy", fontsize=12)

    plt.tight_layout()

    # 保存图表
    if save_path:
        _save_figure(fig, save_path)

    return fig


# ============================================================================
# 延迟分布箱线图
# ============================================================================
def plot_latency_boxplot(
    latency_data: Union[pd.DataFrame, Dict[str, List[float]]],
    title: str = "Latency Distribution Comparison",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show_points: bool = True,
    ylabel: str = "Latency (ms)",
) -> Figure:
    """
    绘制延迟分布箱线图。

    Args:
        latency_data: 延迟数据。
            可以是 DataFrame（列=策略），
            或字典格式 {策略名: [延迟列表]}。
        title: 图表标题。
        figsize: 图表尺寸。
        save_path: 保存路径。
        show_points: 是否显示散点。
        ylabel: Y 轴标签。

    Returns:
        matplotlib Figure 对象。
    """
    # 转换为 DataFrame
    if isinstance(latency_data, dict):
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in latency_data.items()]))
    else:
        df = latency_data

    fig, ax = plt.subplots(figsize=figsize)

    # 颜色列表
    strategies = df.columns.tolist()
    colors = [COLORS.get(s, sns.color_palette()[i]) for i, s in enumerate(strategies)]

    # 绘制箱线图
    boxprops = dict(linewidth=1.5)
    medianprops = dict(linewidth=2, color="black")
    whiskerprops = dict(linewidth=1.5)
    capprops = dict(linewidth=1.5)

    bp = ax.boxplot(
        [df[col].dropna().values for col in strategies],
        labels=strategies,
        patch_artist=True,
        showfliers=not show_points,
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
    )

    # 设置箱体颜色
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # 添加散点（抖动）
    if show_points:
        for i, col in enumerate(strategies):
            y = df[col].dropna().values
            x = np.random.normal(i + 1, 0.04, size=len(y))
            ax.scatter(
                x, y,
                alpha=0.4,
                s=20,
                color=colors[i],
                edgecolor="white",
                linewidth=0.5,
            )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()

    # 保存图表
    if save_path:
        _save_figure(fig, save_path)

    return fig


# ============================================================================
# 综合指标对比图
# ============================================================================
def plot_metrics_comparison(
    metrics_data: pd.DataFrame,
    title: str = "Strategy Metrics Comparison",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    kind: str = "bar",
) -> Figure:
    """
    绘制指标对比图。

    Args:
        metrics_data: 指标数据 DataFrame，行为策略，列为指标。
        title: 图表标题。
        figsize: 图表尺寸。
        save_path: 保存路径。
        kind: 图表类型 ('bar', 'radar').

    Returns:
        matplotlib Figure 对象。
    """
    if kind == "bar":
        return _plot_bar_comparison(metrics_data, title, figsize, save_path)
    elif kind == "radar":
        return _plot_radar_comparison(metrics_data, title, figsize, save_path)
    else:
        raise ValueError(f"Unknown kind: {kind}")


def _plot_bar_comparison(
    df: pd.DataFrame,
    title: str,
    figsize: Tuple[int, int],
    save_path: Optional[str],
) -> Figure:
    """绘制柱状对比图。"""
    fig, ax = plt.subplots(figsize=figsize)

    # 获取颜色
    strategies = df.index.tolist()
    colors = [COLORS.get(s, sns.color_palette()[i]) for i, s in enumerate(strategies)]

    # 绘制柱状图
    df.plot(
        kind="bar",
        ax=ax,
        color=colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("Strategy", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def _plot_radar_comparison(
    df: pd.DataFrame,
    title: str,
    figsize: Tuple[int, int],
    save_path: Optional[str],
) -> Figure:
    """绘制雷达对比图。"""
    # 标准化数据到 0-1 范围
    df_norm = (df - df.min()) / (df.max() - df.min())

    # 准备雷达图参数
    categories = df.columns.tolist()
    N = len(categories)

    # 创建角度
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    # 颜色
    strategies = df_norm.index.tolist()
    colors = [COLORS.get(s, sns.color_palette()[i]) for i, s in enumerate(strategies)]

    # 绘制每个策略
    for i, (idx, row) in enumerate(df_norm.iterrows()):
        values = row.tolist()
        values += values[:1]  # 闭合

        ax.plot(angles, values, "o-", linewidth=2, label=idx, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])

    # 设置标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


# ============================================================================
# 回撤水下图
# ============================================================================
def plot_underwater(
    equity_curve: pd.DataFrame,
    title: str = "Underwater Plot",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """
    绘制回撤水下图。

    Args:
        equity_curve: 净值曲线 DataFrame，需包含 'nav' 或 'value' 列。
        title: 图表标题。
        figsize: 图表尺寸。
        save_path: 保存路径。

    Returns:
        matplotlib Figure 对象。
    """
    # 获取净值
    if "nav" in equity_curve.columns:
        nav = equity_curve["nav"]
    elif "value" in equity_curve.columns:
        nav = equity_curve["value"] / equity_curve["value"].iloc[0]
    else:
        raise ValueError("equity_curve 必须包含 'nav' 或 'value' 列")

    # 计算回撤
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax

    fig, ax = plt.subplots(figsize=figsize)

    # 填充水下区域
    ax.fill_between(
        drawdown.index,
        drawdown.values,
        0,
        color=COLORS["fusion"],
        alpha=0.7,
    )

    # 标记最大回撤点
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min()

    if isinstance(max_dd_idx, pd.Timestamp):
        ax.scatter(
            [max_dd_idx],
            [max_dd_val],
            color="red",
            s=100,
            zorder=5,
            label=f"Max DD: {max_dd_val:.2%}",
        )
        ax.annotate(
            f"{max_dd_val:.2%}",
            xy=(max_dd_idx, max_dd_val),
            xytext=(max_dd_idx, max_dd_val + 0.05),
            fontsize=10,
            ha="center",
        )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Drawdown", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


# ============================================================================
# 滚动夏普比率图
# ============================================================================
def plot_rolling_sharpe(
    equity_curve: pd.DataFrame,
    window: int = 20,
    title: str = "Rolling Sharpe Ratio",
    figsize: Tuple[int, int] = (12, 4),
    save_path: Optional[str] = None,
) -> Figure:
    """
    绘制滚动夏普比率图。

    Args:
        equity_curve: 净值曲线 DataFrame。
        window: 滚动窗口大小。
        title: 图表标题。
        figsize: 图表尺寸。
        save_path: 保存路径。

    Returns:
        matplotlib Figure 对象。
    """
    # 获取净值
    if "nav" in equity_curve.columns:
        nav = equity_curve["nav"]
    elif "value" in equity_curve.columns:
        nav = equity_curve["value"] / equity_curve["value"].iloc[0]
    else:
        raise ValueError("equity_curve 必须包含 'nav' 或 'value' 列")

    # 计算日收益率
    returns = nav.pct_change()

    # 计算滚动夏普比率
    rolling_sharpe = (
        returns.rolling(window=window).mean()
        / returns.rolling(window=window).std()
        * np.sqrt(252)
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        rolling_sharpe.index,
        rolling_sharpe.values,
        color=COLORS["ml"],
        linewidth=1.5,
    )

    # 添加零线
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    # 添加 1.0 参考线（一般可接受的夏普比率）
    ax.axhline(y=1.0, color="green", linestyle=":", linewidth=1, alpha=0.7, label="Sharpe = 1.0")

    ax.set_title(f"{title} ({window}-day)", fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel("Sharpe Ratio", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


# ============================================================================
# 辅助函数
# ============================================================================
def _save_figure(fig: Figure, path: str) -> None:
    """
    保存图表到文件。

    Args:
        fig: matplotlib Figure 对象。
        path: 保存路径。
    """
    # 确保目录存在
    save_dir = Path(path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存
    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    print(f"图表已保存至: {path}")


def generate_all_figures(
    results: Dict[str, Any],
    output_dir: str = "docs/figures",
    prefix: str = "",
) -> Dict[str, Figure]:
    """
    生成所有论文图表。

    Args:
        results: 评估结果字典，包含各策略的净值曲线、延迟数据等。
        output_dir: 输出目录。
        prefix: 文件名前缀。

    Returns:
        图表字典。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures = {}

    # 1. 资金曲线对比图
    if "equity_curves" in results:
        fig = plot_equity_curves(
            results["equity_curves"],
            benchmark=results.get("benchmark"),
            title="Strategy Equity Curves Comparison",
            save_path=output_dir / f"{prefix}equity_curves.png",
        )
        figures["equity_curves"] = fig

    # 2. 最大回撤热力图
    if "drawdown_data" in results:
        fig = plot_drawdown_heatmap(
            results["drawdown_data"],
            title="Maximum Drawdown by Strategy and Period",
            save_path=output_dir / f"{prefix}drawdown_heatmap.png",
        )
        figures["drawdown_heatmap"] = fig

    # 3. 延迟分布箱线图
    if "latency_data" in results:
        fig = plot_latency_boxplot(
            results["latency_data"],
            title="Inference Latency Distribution",
            save_path=output_dir / f"{prefix}latency_boxplot.png",
        )
        figures["latency_boxplot"] = fig

    # 4. 指标对比图
    if "metrics_data" in results:
        fig = plot_metrics_comparison(
            results["metrics_data"],
            title="Strategy Metrics Comparison",
            save_path=output_dir / f"{prefix}metrics_comparison.png",
        )
        figures["metrics_comparison"] = fig

    return figures


# ============================================================================
# 模块测试
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  可视化模块示例")
    print("=" * 60)

    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")

    # 三条净值曲线
    ml_returns = np.random.randn(100) * 0.015 + 0.0008
    llm_returns = np.random.randn(100) * 0.020 + 0.0005
    fusion_returns = 0.6 * ml_returns + 0.4 * llm_returns + 0.0002

    equity_curves = {
        "ML_Baseline": pd.DataFrame({
            "nav": 1.0 * (1 + ml_returns).cumprod()
        }, index=dates),
        "LLM_Baseline": pd.DataFrame({
            "nav": 1.0 * (1 + llm_returns).cumprod()
        }, index=dates),
        "DualTrack_Fusion": pd.DataFrame({
            "nav": 1.0 * (1 + fusion_returns).cumprod()
        }, index=dates),
    }

    # 创建输出目录
    output_dir = Path("docs/figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制资金曲线对比图
    print("\n绘制资金曲线对比图...")
    fig1 = plot_equity_curves(
        equity_curves,
        title="DualTrack Strategy Equity Curves Comparison",
        save_path=str(output_dir / "equity_curves.png"),
    )

    # 绘制回撤热力图
    print("\n绘制最大回撤热力图...")
    drawdown_data = {
        "ML_Baseline": {"2024-Q1": 0.05, "2024-Q2": 0.08, "2024-Q3": 0.06, "2024-Q4": 0.04},
        "LLM_Baseline": {"2024-Q1": 0.07, "2024-Q2": 0.12, "2024-Q3": 0.09, "2024-Q4": 0.06},
        "DualTrack_Fusion": {"2024-Q1": 0.03, "2024-Q2": 0.05, "2024-Q3": 0.04, "2024-Q4": 0.02},
    }
    fig2 = plot_drawdown_heatmap(
        drawdown_data,
        title="Maximum Drawdown by Strategy and Quarter",
        save_path=str(output_dir / "drawdown_heatmap.png"),
    )

    # 绘制延迟分布箱线图
    print("\n绘制延迟分布箱线图...")
    latency_data = {
        "ML_Baseline": np.random.uniform(5, 20, 100).tolist(),
        "LLM_Baseline": np.random.uniform(100, 500, 100).tolist(),
        "DualTrack_Fusion": np.random.uniform(50, 200, 100).tolist(),
    }
    fig3 = plot_latency_boxplot(
        latency_data,
        title="Inference Latency Distribution",
        save_path=str(output_dir / "latency_boxplot.png"),
    )

    print("\n" + "=" * 60)
    print("  所有图表已生成！")
    print("=" * 60)