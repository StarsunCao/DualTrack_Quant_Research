"""
跨市场规则敏感度分析模块。

对比双轨制引擎在不同市场（A股/美股）的表现差异，
验证 LLM 的零样本泛化能力。

分析内容:
- T+0 vs T+1 信号衰减分析
- 零样本泛化能力评估
- 跨市场信号相关性

学术价值:
- 验证"零样本跨市场泛化证明系统解耦性"
- 揭示 LLM 仅切换 Prompt 即可跨市场的灵活性
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SignalDecayResult:
    """
    信号衰减分析结果数据类。

    Attributes:
        strategy: 策略名称。
        market: 市场 ('A_share' / 'US_market')。
        signal_date: 信号日期。
        signal_strength: 信号强度。
        return_t: T 日收益。
        return_t1: T+1 日收益。
        return_t3: T+3 日收益。
        return_t5: T+5 日收益。
        cumulative_decay: 累计衰减。
    """
    strategy: str
    market: str
    signal_date: pd.Timestamp
    signal_strength: float
    return_t: float = 0.0
    return_t1: float = 0.0
    return_t3: float = 0.0
    return_t5: float = 0.0
    cumulative_decay: float = 0.0


@dataclass
class CrossMarketSummary:
    """
    跨市场分析汇总数据类。

    Attributes:
        strategy: 策略名称。
        a_share_sharpe: A 股夏普比率。
        us_market_sharpe: 美股夏普比率。
        a_share_return: A 股总收益。
        us_market_return: 美股总收益。
        sharpe_gap: 夏普比率差距。
        return_gap: 收益率差距。
        zero_shot_score: 零样本泛化评分。
        correlation: 跨市场信号相关性。
        decay_a_share: A 股信号衰减。
        decay_us_market: 美股信号衰减。
    """
    strategy: str
    a_share_sharpe: float = 0.0
    us_market_sharpe: float = 0.0
    a_share_return: float = 0.0
    us_market_return: float = 0.0
    sharpe_gap: float = 0.0
    return_gap: float = 0.0
    zero_shot_score: float = 0.0
    correlation: float = 0.0
    decay_a_share: float = 0.0
    decay_us_market: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "strategy": self.strategy,
            "a_share_sharpe": f"{self.a_share_sharpe:.4f}",
            "us_market_sharpe": f"{self.us_market_sharpe:.4f}",
            "a_share_return": f"{self.a_share_return:.2%}",
            "us_market_return": f"{self.us_market_return:.2%}",
            "sharpe_gap": f"{self.sharpe_gap:.4f}",
            "zero_shot_score": f"{self.zero_shot_score:.2f}",
            "correlation": f"{self.correlation:.4f}",
        }

    def summary(self) -> str:
        """生成汇总摘要。"""
        lines = [
            f"\n{'='*60}",
            f"  跨市场分析: {self.strategy}",
            f"{'='*60}",
            f"  {'指标':<20} {'A股':>15} {'美股':>15}",
            f"{'-'*60}",
            f"  {'夏普比率':<20} {self.a_share_sharpe:>15.4f} {self.us_market_sharpe:>15.4f}",
            f"  {'总收益率':<20} {self.a_share_return:>14.2%} {self.us_market_return:>14.2%}",
            f"  {'信号衰减':<20} {self.decay_a_share:>14.2%} {self.decay_us_market:>14.2%}",
            f"{'-'*60}",
            f"  夏普差距: {self.sharpe_gap:.4f}",
            f"  零样本泛化评分: {self.zero_shot_score:.2f}",
            f"  跨市场相关性: {self.correlation:.4f}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class CrossMarketAnalyzer:
    """
    跨市场规则敏感度分析器。

    分析策略在不同市场的表现差异，验证零样本泛化能力。

    使用方法:
        analyzer = CrossMarketAnalyzer()

        # 分析信号衰减
        decay_results = analyzer.analyze_signal_decay(
            signals=signals_df,
            ohlcv=ohlcv_df,
            market="A_share",
        )

        # 跨市场对比
        summary = analyzer.compare_markets(
            a_share_results=a_share_summary,
            us_market_results=us_market_summary,
            strategy_name="LightGBM",
        )

        # 生成可视化
        analyzer.plot_cross_market_radar(
            summaries=summaries,
            save_path="docs/figures/cross_market_radar.png",
        )
    """

    # 交易规则差异
    MARKET_RULES = {
        "A_share": {
            "settlement": "T+1",  # T+1 结算
            "short_allowed": False,  # 不允许做空
            "trading_hours": "9:30-11:30, 13:00-15:00",
            "price_limit": 0.1,  # 涨跌停限制 10%
        },
        "US_market": {
            "settlement": "T+0",  # T+0 结算
            "short_allowed": True,  # 允许做空
            "trading_hours": "9:30-16:00",
            "price_limit": None,  # 无涨跌停限制
        },
    }

    def __init__(self) -> None:
        """初始化分析器。"""
        self.decay_results: List[SignalDecayResult] = []

    def analyze_signal_decay(
        self,
        signals: pd.DataFrame,
        ohlcv: pd.DataFrame,
        market: str = "A_share",
        strategy: str = "Strategy",
        signal_col: str = "signal_strength_0_to_1",
    ) -> List[SignalDecayResult]:
        """
        分析信号衰减。

        计算信号发出后 T, T+1, T+3, T+5 的累计收益，
        对比 T+0 (美股) 和 T+1 (A股) 规则下的信号有效性。

        Args:
            signals: 信号 DataFrame，需包含 'timestamp' 和信号列。
            ohlcv: OHLCV 数据 DataFrame。
            market: 市场 ('A_share' / 'US_market')。
            strategy: 策略名称。
            signal_col: 信号列名。

        Returns:
            SignalDecayResult 列表。
        """
        if signals.empty or ohlcv.empty:
            logger.warning("信号或 OHLCV 数据为空")
            return []

        # 确保日期索引
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv = ohlcv.copy()
            ohlcv.index = pd.to_datetime(ohlcv.index)

        # 准备信号数据
        signals = signals.copy()
        if "timestamp" in signals.columns:
            signals["date"] = pd.to_datetime(signals["timestamp"])
        else:
            signals["date"] = pd.to_datetime(signals.index)

        # 计算收益率
        ohlcv["return"] = ohlcv["close"].pct_change()
        ohlcv["return_t1"] = ohlcv["return"].shift(-1)  # T+1 收益
        ohlcv["return_t3"] = ohlcv["return"].rolling(3).sum().shift(-3)  # T+3 累计收益
        ohlcv["return_t5"] = ohlcv["return"].rolling(5).sum().shift(-5)  # T+5 累计收益

        results: List[SignalDecayResult] = []

        for _, signal_row in signals.iterrows():
            signal_date = signal_row["date"]
            signal_strength = signal_row.get(signal_col, 0.5)

            # 找到对应的 OHLCV 数据
            try:
                # 使用最近的有效日期
                if signal_date in ohlcv.index:
                    ohlcv_row = ohlcv.loc[signal_date]
                else:
                    # 找最近的交易日
                    valid_dates = ohlcv.index[ohlcv.index <= signal_date]
                    if len(valid_dates) == 0:
                        continue
                    signal_date = valid_dates[-1]
                    ohlcv_row = ohlcv.loc[signal_date]

                return_t = ohlcv_row.get("return", 0)
                return_t1 = ohlcv_row.get("return_t1", 0)
                return_t3 = ohlcv_row.get("return_t3", 0)
                return_t5 = ohlcv_row.get("return_t5", 0)

                # 处理 NaN
                return_t = 0 if pd.isna(return_t) else return_t
                return_t1 = 0 if pd.isna(return_t1) else return_t1
                return_t3 = 0 if pd.isna(return_t3) else return_t3
                return_t5 = 0 if pd.isna(return_t5) else return_t5

                # 根据信号方向调整收益
                if signal_strength > 0.5:  # 买入信号
                    direction = 1
                else:  # 卖出信号
                    direction = -1
                    return_t = -return_t
                    return_t1 = -return_t1
                    return_t3 = -return_t3
                    return_t5 = -return_t5

                # 计算衰减（相对于 T 日收益）
                if return_t != 0:
                    decay = (return_t5 - return_t) / abs(return_t)
                else:
                    decay = 0

                results.append(SignalDecayResult(
                    strategy=strategy,
                    market=market,
                    signal_date=signal_date,
                    signal_strength=signal_strength,
                    return_t=return_t,
                    return_t1=return_t1,
                    return_t3=return_t3,
                    return_t5=return_t5,
                    cumulative_decay=decay,
                ))

            except (KeyError, IndexError):
                continue

        self.decay_results = results
        logger.info(f"信号衰减分析完成: {len(results)} 条记录 ({market})")
        return results

    def compare_markets(
        self,
        a_share_sharpe: float,
        a_share_return: float,
        us_market_sharpe: float,
        us_market_return: float,
        a_share_decay: Optional[List[SignalDecayResult]] = None,
        us_market_decay: Optional[List[SignalDecayResult]] = None,
        a_share_signals: Optional[pd.DataFrame] = None,
        us_market_signals: Optional[pd.DataFrame] = None,
        strategy_name: str = "Strategy",
    ) -> CrossMarketSummary:
        """
        对比策略在不同市场的表现。

        Args:
            a_share_sharpe: A 股夏普比率。
            a_share_return: A 股总收益。
            us_market_sharpe: 美股夏普比率。
            us_market_return: 美股总收益。
            a_share_decay: A 股信号衰减结果。
            us_market_decay: 美股信号衰减结果。
            a_share_signals: A 股信号 DataFrame。
            us_market_signals: 美股信号 DataFrame。
            strategy_name: 策略名称。

        Returns:
            CrossMarketSummary 对象。
        """
        # 计算差距
        sharpe_gap = abs(a_share_sharpe - us_market_sharpe)
        return_gap = a_share_return - us_market_return

        # 计算零样本泛化评分
        # 如果策略在两个市场都有正收益且夏普比率相近，说明泛化能力强
        if a_share_sharpe > 0 and us_market_sharpe > 0:
            min_sharpe = min(a_share_sharpe, us_market_sharpe)
            max_sharpe = max(a_share_sharpe, us_market_sharpe)
            zero_shot_score = min_sharpe / max_sharpe if max_sharpe > 0 else 0
        else:
            zero_shot_score = 0

        # 计算跨市场相关性
        correlation = 0.0
        if a_share_signals is not None and us_market_signals is not None:
            correlation = self._compute_cross_correlation(
                a_share_signals, us_market_signals
            )

        # 计算平均衰减
        decay_a = 0.0
        decay_us = 0.0
        if a_share_decay:
            decay_a = np.mean([d.cumulative_decay for d in a_share_decay])
        if us_market_decay:
            decay_us = np.mean([d.cumulative_decay for d in us_market_decay])

        return CrossMarketSummary(
            strategy=strategy_name,
            a_share_sharpe=a_share_sharpe,
            us_market_sharpe=us_market_sharpe,
            a_share_return=a_share_return,
            us_market_return=us_market_return,
            sharpe_gap=sharpe_gap,
            return_gap=return_gap,
            zero_shot_score=zero_shot_score,
            correlation=correlation,
            decay_a_share=decay_a,
            decay_us_market=decay_us,
        )

    def _compute_cross_correlation(
        self,
        a_share_signals: pd.DataFrame,
        us_market_signals: pd.DataFrame,
        signal_col: str = "signal_strength_0_to_1",
    ) -> float:
        """计算跨市场信号相关性。"""
        try:
            # 对齐日期
            a_share_signals = a_share_signals.copy()
            us_market_signals = us_market_signals.copy()

            if "timestamp" in a_share_signals.columns:
                a_share_signals["date"] = pd.to_datetime(a_share_signals["timestamp"]).dt.date
            else:
                a_share_signals["date"] = pd.to_datetime(a_share_signals.index).date

            if "timestamp" in us_market_signals.columns:
                us_market_signals["date"] = pd.to_datetime(us_market_signals["timestamp"]).dt.date
            else:
                us_market_signals["date"] = pd.to_datetime(us_market_signals.index).date

            # 合并
            merged = pd.merge(
                a_share_signals[["date", signal_col]],
                us_market_signals[["date", signal_col]],
                on="date",
                suffixes=("_a", "_us"),
            )

            if len(merged) > 10:
                return merged[f"{signal_col}_a"].corr(merged[f"{signal_col}_us"])

        except Exception as e:
            logger.warning(f"跨市场相关性计算失败: {e}")

        return 0.0

    def plot_decay_curves(
        self,
        decay_results: Optional[List[SignalDecayResult]] = None,
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """
        绘制信号衰减曲线。

        Args:
            decay_results: SignalDecayResult 列表。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        decay_results = decay_results or self.decay_results

        if not decay_results:
            logger.warning("没有衰减数据可供绘图")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "没有衰减数据", ha="center", va="center")
            return fig

        # 转换为 DataFrame
        df = pd.DataFrame([
            {
                "market": r.market,
                "return_t": r.return_t * 100,
                "return_t1": r.return_t1 * 100,
                "return_t3": r.return_t3 * 100,
                "return_t5": r.return_t5 * 100,
            }
            for r in decay_results
        ])

        # 按市场分组计算平均收益
        avg_by_market = df.groupby("market")[["return_t", "return_t1", "return_t3", "return_t5"]].mean()

        # 绘图
        fig, ax = plt.subplots(figsize=figsize)

        x = [0, 1, 3, 5]
        markers = ["o", "s"]

        for i, (market, row) in enumerate(avg_by_market.iterrows()):
            values = row.values
            ax.plot(x, values, marker=markers[i % 2], linewidth=2, markersize=8,
                    label=market, color="steelblue" if "A" in market else "indianred")

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("交易日")
        ax.set_ylabel("累计收益 (%)")
        ax.set_title("信号衰减曲线对比 (T+0 vs T+1)", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(["T", "T+1", "T+3", "T+5"])

        # 添加注释
        ax.text(
            0.02, 0.98,
            "T+0 市场: 信号当日即可交易\nT+1 市场: 信号次日才能交易",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"信号衰减曲线已保存: {save_path}")

        return fig

    def plot_cross_market_radar(
        self,
        summaries: Dict[str, CrossMarketSummary],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """
        绘制跨市场零样本泛化评分雷达图。

        Args:
            summaries: 策略名称到 CrossMarketSummary 的映射。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        if not summaries:
            logger.warning("没有策略数据可供绘图")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "没有策略数据", ha="center", va="center")
            return fig

        # 定义雷达图维度
        categories = [
            "A股夏普",
            "美股夏普",
            "A股收益",
            "美股收益",
            "零样本评分",
            "相关性",
        ]

        # 准备数据
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]

        colors = plt.cm.Set2(np.linspace(0, 1, len(summaries)))

        for i, (strategy, summary) in enumerate(summaries.items()):
            # 归一化各指标
            values = [
                max(0, summary.a_share_sharpe) / 2,  # 假设最大夏普为2
                max(0, summary.us_market_sharpe) / 2,
                max(0, summary.a_share_return) / 0.5,  # 假设最大收益50%
                max(0, summary.us_market_return) / 0.5,
                summary.zero_shot_score,
                abs(summary.correlation),
            ]
            values = [min(1, max(0, v)) for v in values]
            values += values[:1]

            ax.plot(angles, values, "o-", linewidth=2, label=strategy, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))

        plt.title("跨市场零样本泛化评分", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"跨市场雷达图已保存: {save_path}")

        return fig

    def plot_correlation_heatmap(
        self,
        correlation_matrix: pd.DataFrame,
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """
        绘制跨市场信号相关性热力图。

        Args:
            correlation_matrix: 相关性矩阵 DataFrame。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            center=0,
            ax=ax,
            linewidths=0.5,
        )

        ax.set_title("跨市场信号相关性热力图", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"相关性热力图已保存: {save_path}")

        return fig


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("  跨市场分析器示例")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

    signals = pd.DataFrame({
        "timestamp": dates[::5],
        "signal_strength_0_to_1": np.random.uniform(0.3, 0.7, 20),
    })

    ohlcv = pd.DataFrame({
        "close": 100 + np.cumsum(np.random.randn(100) * 1),
        "high": 100 + np.cumsum(np.random.randn(100) * 1) + 2,
        "low": 100 + np.cumsum(np.random.randn(100) * 1) - 2,
        "open": 100 + np.cumsum(np.random.randn(100) * 1),
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)

    # 分析
    analyzer = CrossMarketAnalyzer()
    decay = analyzer.analyze_signal_decay(signals, ohlcv, market="A_share", strategy="Test")

    summary = analyzer.compare_markets(
        a_share_sharpe=1.2,
        a_share_return=0.25,
        us_market_sharpe=1.0,
        us_market_return=0.20,
        strategy_name="LightGBM",
    )

    print(summary.summary())