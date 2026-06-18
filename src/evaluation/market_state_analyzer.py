"""
市场状态切割分析器模块。

基于 VIX 指数对市场状态进行分类，揭示策略在不同市场环境下的表现差异。

市场状态分类:
- Bull Quiet: VIX < 15, 牛市平静
- Bull Volatile: VIX 15-20, 牛市波动
- Neutral: VIX 15-25, 震荡市
- Bear Volatile: VIX 20-30, 熊市波动
- Crisis: VIX > 30, 危机模式

学术价值:
- 验证"ML 是 Beta 放大器，LLM 是尾部风险切断器"假设
- 揭示策略在不同市场环境下的适应性
- 支持论文的核心结论
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketState(Enum):
    """市场状态枚举。"""
    BULL_QUIET = "bull_quiet"           # 牛市平静: VIX < 15
    BULL_VOLATILE = "bull_volatile"     # 牛市波动: VIX 15-20
    NEUTRAL = "neutral"                  # 震荡市: VIX 15-25
    BEAR_VOLATILE = "bear_volatile"     # 熊市波动: VIX 20-30
    CRISIS = "crisis"                    # 危机模式: VIX > 30


# VIX 阈值定义
VIX_THRESHOLDS = {
    "bull_quiet": (0, 15),
    "bull_volatile": (15, 20),
    "neutral": (15, 25),
    "bear_volatile": (20, 30),
    "crisis": (30, float('inf')),
}


@dataclass
class MarketStateMetrics:
    """
    单一市场状态下的策略指标数据类。

    Attributes:
        state: 市场状态。
        state_name: 状态中文名称。
        days: 该状态的交易日数量。
        days_pct: 占总交易日的比例。
        total_return: 该状态下的总收益率。
        avg_return: 该状态下的平均日收益率。
        sharpe: 该状态下的夏普比率。
        max_drawdown: 该状态下的最大回撤。
        win_rate: 该状态下的胜率。
        volatility: 该状态下的年化波动率。
        trade_count: 该状态下的交易次数。
    """
    state: MarketState
    state_name: str
    days: int = 0
    days_pct: float = 0.0
    total_return: float = 0.0
    avg_return: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    volatility: float = 0.0
    trade_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "state": self.state.value,
            "state_name": self.state_name,
            "days": self.days,
            "days_pct": f"{self.days_pct:.2%}",
            "total_return": f"{self.total_return:.2%}",
            "avg_return": f"{self.avg_return:.4%}",
            "sharpe": f"{self.sharpe:.4f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "volatility": f"{self.volatility:.2%}",
            "trade_count": self.trade_count,
        }


@dataclass
class MarketStateSummary:
    """
    市场状态切割汇总数据类。

    Attributes:
        strategy_name: 策略名称。
        state_metrics: 各市场状态的指标字典。
        overall_return: 总体收益率。
        crisis_return: 危机期间收益率。
        crisis_outperformance: 危机期间相对基准的超额收益。
        state_transition_count: 状态转换次数。
        most_common_state: 最常见的市场状态。
    """
    strategy_name: str
    state_metrics: Dict[MarketState, MarketStateMetrics] = field(default_factory=dict)
    overall_return: float = 0.0
    crisis_return: float = 0.0
    crisis_outperformance: float = 0.0
    state_transition_count: int = 0
    most_common_state: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "strategy_name": self.strategy_name,
            "state_metrics": {
                state.value: metrics.to_dict()
                for state, metrics in self.state_metrics.items()
            },
            "overall_return": f"{self.overall_return:.2%}",
            "crisis_return": f"{self.crisis_return:.2%}",
            "crisis_outperformance": f"{self.crisis_outperformance:.2%}",
            "state_transition_count": self.state_transition_count,
            "most_common_state": self.most_common_state,
        }

    def summary(self) -> str:
        """生成汇总摘要字符串。"""
        lines = [
            f"\n{'='*70}",
            f"  市场状态切割分析: {self.strategy_name}",
            f"{'='*70}",
            f"  {'状态':<15} {'天数':>8} {'占比':>8} {'收益':>10} {'夏普':>8} {'回撤':>8}",
            f"{'-'*70}",
        ]

        state_order = [
            MarketState.BULL_QUIET,
            MarketState.BULL_VOLATILE,
            MarketState.NEUTRAL,
            MarketState.BEAR_VOLATILE,
            MarketState.CRISIS,
        ]

        for state in state_order:
            if state in self.state_metrics:
                m = self.state_metrics[state]
                lines.append(
                    f"  {m.state_name:<15} {m.days:>8} {m.days_pct:>7.1%} "
                    f"{m.total_return:>9.2%} {m.sharpe:>8.4f} {m.max_drawdown:>7.2%}"
                )

        lines.extend([
            f"{'-'*70}",
            f"  总收益率: {self.overall_return:.2%}",
            f"  危机期间收益: {self.crisis_return:.2%}",
            f"  状态转换次数: {self.state_transition_count}",
            f"  最常见状态: {self.most_common_state}",
            f"{'='*70}",
        ])

        return "\n".join(lines)


class MarketStateAnalyzer:
    """
    市场状态切割分析器。

    基于 VIX 指数对市场状态进行分类，分析策略在不同市场环境下的表现。

    使用方法:
        analyzer = MarketStateAnalyzer()

        # 加载 VIX 数据
        analyzer.load_vix_data("data/raw/vix_2015_2024.csv")

        # 分析策略
        summary = analyzer.analyze_strategy(
            equity_curve=equity_df,
            trade_log=trade_df,
            strategy_name="LightGBM",
        )

        # 生成可视化
        analyzer.plot_state_heatmap(summaries, save_path="docs/figures/market_state_heatmap.png")
    """

    STATE_NAMES = {
        MarketState.BULL_QUIET: "Bull Quiet",
        MarketState.BULL_VOLATILE: "Bull Volatile",
        MarketState.NEUTRAL: "Neutral",
        MarketState.BEAR_VOLATILE: "Bear Volatile",
        MarketState.CRISIS: "Crisis",
    }

    STATE_COLORS = {
        MarketState.BULL_QUIET: "#2ecc71",      # 绿色
        MarketState.BULL_VOLATILE: "#f1c40f",  # 黄色
        MarketState.NEUTRAL: "#95a5a6",         # 灰色
        MarketState.BEAR_VOLATILE: "#e67e22",  # 橙色
        MarketState.CRISIS: "#e74c3c",          # 红色
    }

    def __init__(
        self,
        vix_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        初始化分析器。

        Args:
            vix_thresholds: 自定义 VIX 阈值，如果为 None 则使用默认值。
        """
        self.vix_thresholds = vix_thresholds or VIX_THRESHOLDS
        self.vix_data: Optional[pd.DataFrame] = None
        self.state_series: Optional[pd.Series] = None

    def load_vix_data(
        self,
        vix_path: str,
        date_col: str = None,
        vix_col: str = "Close",
    ) -> None:
        """
        加载 VIX 数据。

        Args:
            vix_path: VIX 数据文件路径。
            date_col: 日期列名，如果为 None 则自动检测。
            vix_col: VIX 收盘价列名。
        """
        # 先读取第一行检测列名
        sample = pd.read_csv(vix_path, nrows=1)

        # 自动检测日期列
        if date_col is None:
            for col in ["date", "Date", "timestamp", "Timestamp"]:
                if col in sample.columns:
                    date_col = col
                    break

        if date_col is None:
            raise ValueError(f"无法检测日期列，可用列: {list(sample.columns)}")

        vix_df = pd.read_csv(vix_path, parse_dates=[date_col])
        vix_df.set_index(date_col, inplace=True)

        # 确保有 VIX 列
        if vix_col in vix_df.columns:
            self.vix_data = vix_df[[vix_col]].rename(columns={vix_col: "vix"})
        elif "vix" in vix_df.columns:
            self.vix_data = vix_df[["vix"]]
        elif "Close" in vix_df.columns:
            self.vix_data = vix_df[["Close"]].rename(columns={"Close": "vix"})
        else:
            raise ValueError(f"VIX 数据缺少 '{vix_col}' 或 'vix' 列，可用列: {list(vix_df.columns)}")

        # 分类市场状态
        self.state_series = self._classify_market_states(self.vix_data["vix"])

        logger.info(f"VIX 数据已加载: {len(self.vix_data)} 条记录")
        logger.info(f"日期范围: {self.vix_data.index.min()} ~ {self.vix_data.index.max()}")

    def _classify_market_states(self, vix_series: pd.Series) -> pd.Series:
        """
        根据 VIX 值分类市场状态。

        Args:
            vix_series: VIX 值序列。

        Returns:
            市场状态序列。
        """
        def classify_vix(vix: float) -> MarketState:
            if pd.isna(vix):
                return MarketState.NEUTRAL
            if vix < 15:
                return MarketState.BULL_QUIET
            elif vix < 20:
                return MarketState.BULL_VOLATILE
            elif vix < 25:
                return MarketState.NEUTRAL
            elif vix < 30:
                return MarketState.BEAR_VOLATILE
            else:
                return MarketState.CRISIS

        return vix_series.apply(classify_vix)

    def analyze_strategy(
        self,
        equity_curve: pd.DataFrame,
        trade_log: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
        value_col: str = "value",
    ) -> MarketStateSummary:
        """
        分析策略在不同市场状态下的表现。

        Args:
            equity_curve: 净值曲线 DataFrame，索引为日期，包含 'value' 或 'nav' 列。
            trade_log: 交易日志 DataFrame（可选）。
            strategy_name: 策略名称。
            value_col: 资产价值列名。

        Returns:
            MarketStateSummary 对象。
        """
        if self.vix_data is None or self.state_series is None:
            raise ValueError("请先调用 load_vix_data() 加载 VIX 数据")

        # 确保日期索引
        if not isinstance(equity_curve.index, pd.DatetimeIndex):
            equity_curve = equity_curve.copy()
            equity_curve.index = pd.to_datetime(equity_curve.index)

        # 计算日收益率
        if "nav" in equity_curve.columns:
            nav = equity_curve["nav"]
        elif value_col in equity_curve.columns:
            nav = equity_curve[value_col] / equity_curve[value_col].iloc[0]
        else:
            raise ValueError("equity_curve 必须包含 'nav' 或 'value' 列")

        returns = nav.pct_change().dropna()

        # 对齐日期
        aligned_states = self.state_series.reindex(returns.index, method="ffill")
        aligned_states = aligned_states.dropna()

        # Calculate metrics for each state
        state_metrics: Dict[MarketState, MarketStateMetrics] = {}

        for state in MarketState:
            state_mask = aligned_states == state
            state_returns = returns[state_mask]

            if len(state_returns) > 0:
                # Basic statistics
                days = len(state_returns)
                days_pct = days / len(returns)
                total_return = (1 + state_returns).prod() - 1
                avg_return = state_returns.mean()
                volatility = state_returns.std() * np.sqrt(252)

                # Sharpe ratio
                if state_returns.std() > 0:
                    sharpe = state_returns.mean() / state_returns.std() * np.sqrt(252)
                else:
                    sharpe = 0.0

                # Max drawdown
                nav_state = (1 + state_returns).cumprod()
                cummax = nav_state.cummax()
                drawdown = (nav_state - cummax) / cummax
                max_drawdown = abs(drawdown.min())

                # Win rate. Prefer realized trade PnL when a trade log is supplied;
                # otherwise use the share of positive daily strategy returns.
                win_rate = (state_returns > 0).mean()
                trade_count = 0
                if trade_log is not None and isinstance(trade_log, pd.DataFrame) and not trade_log.empty:
                    # 匹配该状态下的交易
                    if "date" in trade_log.columns:
                        trade_dates = pd.to_datetime(trade_log["date"])
                        trade_in_state = trade_log[
                            trade_dates.isin(state_returns.index)
                        ]
                        if "pnl" in trade_in_state.columns and len(trade_in_state) > 0:
                            win_rate = (trade_in_state["pnl"] > 0).mean()
                            trade_count = len(trade_in_state)

                state_metrics[state] = MarketStateMetrics(
                    state=state,
                    state_name=self.STATE_NAMES[state],
                    days=days,
                    days_pct=days_pct,
                    total_return=total_return,
                    avg_return=avg_return,
                    sharpe=sharpe,
                    max_drawdown=max_drawdown,
                    win_rate=win_rate,
                    volatility=volatility,
                    trade_count=trade_count,
                )

        # 计算总体指标
        overall_return = (1 + returns).prod() - 1

        # 计算危机期间收益
        crisis_mask = aligned_states == MarketState.CRISIS
        crisis_returns = returns[crisis_mask]
        crisis_return = (1 + crisis_returns).prod() - 1 if len(crisis_returns) > 0 else 0.0

        # 计算状态转换次数
        state_transitions = (aligned_states != aligned_states.shift(1)).sum() - 1

        # 找出最常见的状态
        state_counts = aligned_states.value_counts()
        most_common_state = self.STATE_NAMES[state_counts.index[0]] if len(state_counts) > 0 else ""

        return MarketStateSummary(
            strategy_name=strategy_name,
            state_metrics=state_metrics,
            overall_return=overall_return,
            crisis_return=crisis_return,
            crisis_outperformance=0.0,  # 需要基准才能计算
            state_transition_count=state_transitions,
            most_common_state=most_common_state,
        )

    def plot_state_timeline(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (14, 6),
    ) -> plt.Figure:
        """
        绘制市场状态时间线。

        Args:
            start_date: 开始日期。
            end_date: 结束日期。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        if self.vix_data is None or self.state_series is None:
            raise ValueError("请先调用 load_vix_data() 加载 VIX 数据")

        # 过滤日期范围
        vix = self.vix_data["vix"].copy()
        states = self.state_series.copy()

        if start_date:
            start_dt = pd.to_datetime(start_date)
            vix = vix[vix.index >= start_dt]
            states = states[states.index >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            vix = vix[vix.index <= end_dt]
            states = states[states.index <= end_dt]

        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1, 2])

        # 1. VIX 曲线
        ax1 = axes[0]
        ax1.plot(vix.index, vix.values, color="steelblue", linewidth=1)

        # 添加阈值线
        threshold_values = [15, 20, 25, 30]
        threshold_labels = ["牛市平静边界(15)", "牛市波动边界(20)", "震荡市边界(25)", "危机边界(30)"]
        for val, label in zip(threshold_values, threshold_labels):
            ax1.axhline(y=val, color="gray", linestyle="--", alpha=0.5)
            ax1.text(vix.index[-1], val, f" {val}", fontsize=8, va="center")

        ax1.set_ylabel("VIX")
        ax1.set_title("VIX 指数与市场状态边界", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # 2. 市场状态色带
        ax2 = axes[1]

        # 创建状态数值映射（便于绘图）
        state_values = pd.Series(index=states.index, dtype=float)
        for i, state in enumerate(MarketState):
            state_values[states == state] = i

        # 使用颜色填充
        for state in MarketState:
            mask = states == state
            if mask.any():
                # 找到连续的状态段
                state_segments = self._find_segments(mask)
                for start, end in state_segments:
                    ax2.axvspan(start, end, alpha=0.7, color=self.STATE_COLORS[state])

        # 添加图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=self.STATE_COLORS[state], label=self.STATE_NAMES[state])
            for state in MarketState if state in states.values
        ]
        ax2.legend(handles=legend_elements, loc="upper right", ncol=5)
        ax2.set_xlabel("日期")
        ax2.set_ylabel("市场状态")
        ax2.set_title("市场状态时间线", fontsize=12, fontweight="bold")
        ax2.set_yticks([])

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"市场状态时间线已保存: {save_path}")

        return fig

    def _find_segments(self, mask: pd.Series) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """找到连续的 True 段。"""
        segments = []
        start = None
        for i, (date, is_true) in enumerate(mask.items()):
            if is_true and start is None:
                start = date
            elif not is_true and start is not None:
                segments.append((start, mask.index[i - 1]))
                start = None
        if start is not None:
            segments.append((start, mask.index[-1]))
        return segments

    def plot_state_heatmap(
        self,
        summaries: Dict[str, MarketStateSummary],
        metric: str = "total_return",
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """
        绘制多策略市场状态表现热力图。

        Args:
            summaries: 策略名称到 MarketStateSummary 的映射。
            metric: 对比的指标 ('total_return', 'sharpe', 'max_drawdown', 'win_rate')。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        if not summaries:
            logger.warning("没有策略数据可供对比")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No strategy data", ha="center", va="center")
            return fig

        # 构建数据矩阵
        state_order = [
            MarketState.BULL_QUIET,
            MarketState.BULL_VOLATILE,
            MarketState.NEUTRAL,
            MarketState.BEAR_VOLATILE,
            MarketState.CRISIS,
        ]

        strategies = list(summaries.keys())
        states_names = [self.STATE_NAMES[s] for s in state_order]

        data = np.zeros((len(strategies), len(state_order)))

        for i, strategy in enumerate(strategies):
            summary = summaries[strategy]
            for j, state in enumerate(state_order):
                if state in summary.state_metrics:
                    m = summary.state_metrics[state]
                    if metric == "total_return":
                        data[i, j] = m.total_return
                    elif metric == "sharpe":
                        data[i, j] = m.sharpe
                    elif metric == "max_drawdown":
                        data[i, j] = -m.max_drawdown  # 负值便于显示
                    elif metric == "win_rate":
                        data[i, j] = m.win_rate

        # 绘制热力图
        fig, ax = plt.subplots(figsize=figsize)

        if metric == "total_return":
            fmt = ".1%"
            cmap = "RdYlGn"
            center = 0
            title = "Strategy Returns Across Market States"
        elif metric == "sharpe":
            fmt = ".2f"
            cmap = "RdYlGn"
            center = 0
            title = "Strategy Sharpe Ratios Across Market States"
        elif metric == "max_drawdown":
            fmt = ".1%"
            cmap = "RdYlGn_r"
            center = 0
            title = "Strategy Maximum Drawdowns Across Market States"
        else:
            fmt = ".1%"
            cmap = "RdYlGn"
            center = 0.5
            title = "Strategy Win Rates Across Market States"

        sns.heatmap(
            data,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            center=center,
            xticklabels=states_names,
            yticklabels=strategies,
            ax=ax,
            linewidths=0.5,
        )

        ax.set_xlabel("Market State")
        ax.set_ylabel("Strategy")
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"市场状态热力图已保存: {save_path}")

        return fig

    def plot_strategy_comparison(
        self,
        summaries: Dict[str, MarketStateSummary],
        save_path: Optional[str] = None,
        figsize: tuple = (14, 10),
    ) -> plt.Figure:
        """
        绘制多策略市场状态对比图。

        Args:
            summaries: 策略名称到 MarketStateSummary 的映射。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        if not summaries:
            logger.warning("没有策略数据可供对比")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "No strategy data", ha="center", va="center")
            return fig

        strategies = list(summaries.keys())
        state_order = [
            MarketState.BULL_QUIET,
            MarketState.BULL_VOLATILE,
            MarketState.NEUTRAL,
            MarketState.BEAR_VOLATILE,
            MarketState.CRISIS,
        ]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. 收益率对比
        ax1 = axes[0, 0]
        x = np.arange(len(state_order))
        width = 0.8 / len(strategies)

        for i, strategy in enumerate(strategies):
            summary = summaries[strategy]
            returns = [
                summary.state_metrics.get(s, MarketStateMetrics(s, "")).total_return
                for s in state_order
            ]
            ax1.bar(x + i * width, returns, width, label=strategy)

        ax1.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax1.set_xticklabels([self.STATE_NAMES[s] for s in state_order], rotation=45, ha="right")
        ax1.set_ylabel("Return")
        ax1.set_title("Return Comparison", fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # 2. 夏普比率对比
        ax2 = axes[0, 1]
        for i, strategy in enumerate(strategies):
            summary = summaries[strategy]
            sharpes = [
                summary.state_metrics.get(s, MarketStateMetrics(s, "")).sharpe
                for s in state_order
            ]
            ax2.bar(x + i * width, sharpes, width, label=strategy)

        ax2.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax2.set_xticklabels([self.STATE_NAMES[s] for s in state_order], rotation=45, ha="right")
        ax2.set_ylabel("Sharpe Ratio")
        ax2.set_title("Sharpe Ratio Comparison", fontweight="bold")
        ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        # 3. 最大回撤对比
        ax3 = axes[1, 0]
        for i, strategy in enumerate(strategies):
            summary = summaries[strategy]
            drawdowns = [
                summary.state_metrics.get(s, MarketStateMetrics(s, "")).max_drawdown
                for s in state_order
            ]
            ax3.bar(x + i * width, drawdowns, width, label=strategy)

        ax3.set_xticks(x + width * (len(strategies) - 1) / 2)
        ax3.set_xticklabels([self.STATE_NAMES[s] for s in state_order], rotation=45, ha="right")
        ax3.set_ylabel("Maximum Drawdown")
        ax3.set_title("Maximum Drawdown Comparison", fontweight="bold")

        # 4. 危机期间表现
        ax4 = axes[1, 1]
        crisis_returns = [summaries[s].crisis_return for s in strategies]
        overall_returns = [summaries[s].overall_return for s in strategies]

        x = np.arange(len(strategies))
        ax4.bar(x - 0.2, overall_returns, 0.4, label="Overall Return", color="steelblue")
        ax4.bar(x + 0.2, crisis_returns, 0.4, label="Crisis Return", color="indianred")

        ax4.set_xticks(x)
        ax4.set_xticklabels(strategies, rotation=45, ha="right")
        ax4.set_ylabel("Return")
        ax4.set_title("Overall Return vs Crisis Return", fontweight="bold")
        ax4.legend()
        ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

        plt.suptitle("Multi-Strategy Performance Across Market States", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"策略对比图已保存: {save_path}")

        return fig


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("  市场状态切割分析器示例")
    print("=" * 60)

    # 创建模拟 VIX 数据
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq="B")
    vix_values = 15 + np.cumsum(np.random.randn(500) * 0.5)
    vix_values = np.clip(vix_values, 10, 50)

    vix_data = pd.DataFrame({"vix": vix_values}, index=dates)

    # 创建模拟净值曲线
    nav1 = 1.0 * (1 + np.random.randn(500) * 0.01 + 0.0003).cumprod()
    nav2 = 1.0 * (1 + np.random.randn(500) * 0.015 + 0.0005).cumprod()

    equity1 = pd.DataFrame({"value": nav1 * 100000}, index=dates)
    equity2 = pd.DataFrame({"value": nav2 * 100000}, index=dates)

    # 分析
    analyzer = MarketStateAnalyzer()
    analyzer.vix_data = vix_data
    analyzer.state_series = analyzer._classify_market_states(vix_data["vix"])

    summary1 = analyzer.analyze_strategy(equity1, strategy_name="策略A")
    summary2 = analyzer.analyze_strategy(equity2, strategy_name="策略B")

    print(summary1.summary())
    print("\n" + "="*60 + "\n")
    print(summary2.summary())
