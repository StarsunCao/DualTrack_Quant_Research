"""
归因对比分析模块。

对比 ML 模型的统计归因（SHAP）与 LLM 的逻辑归因（reasoning），
解决量化"黑盒信任危机"。

学术价值:
- 展示统计归因 vs 逻辑归因的对齐程度
- 揭示两种方法的互补性
- 支持论文核心创新点
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.evaluation.ml_explainer import MLExplanationResult, MLExplainer
from src.evaluation.llm_explainer import LLMExplanationResult, LLMExplainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AttributionAlignment:
    """
    归因对齐结果数据类。

    Attributes:
        feature: 特征/主题名称。
        ml_contribution: ML 贡献度（SHAP 值）。
        llm_contribution: LLM 贡献度（主题权重）。
        alignment_score: 对齐分数 (-1 到 1)。
        alignment_type: 对齐类型 ('aligned' / 'conflicting' / 'neutral')。
    """
    feature: str
    ml_contribution: float
    llm_contribution: float
    alignment_score: float
    alignment_type: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "feature": self.feature,
            "ml_contribution": f"{self.ml_contribution:.4f}",
            "llm_contribution": f"{self.llm_contribution:.1%}",
            "alignment_score": f"{self.alignment_score:.2f}",
            "alignment_type": self.alignment_type,
        }


@dataclass
class ComparisonResult:
    """
    归因对比结果数据类。

    Attributes:
        ml_result: ML 解释结果。
        llm_result: LLM 解释结果。
        alignments: 归因对齐列表。
        overall_alignment: 总体对齐分数。
        decision_consistency: 决策一致性。
        complementary_score: 互补性分数。
        explanation: 综合解释文本。
    """
    ml_result: Optional[MLExplanationResult] = None
    llm_result: Optional[LLMExplanationResult] = None
    alignments: List[AttributionAlignment] = field(default_factory=list)
    overall_alignment: float = 0.0
    decision_consistency: bool = False
    complementary_score: float = 0.0
    explanation: str = ""

    def summary(self) -> str:
        """生成汇总摘要。"""
        lines = [
            f"\n{'='*70}",
            f"  ML vs LLM 归因对比",
            f"{'='*70}",
        ]

        if self.ml_result:
            lines.extend([
                f"  ML 预测: {self.ml_result.prediction_direction.upper()} (置信度 {self.ml_result.confidence:.1%})",
                f"  ML Top 特征: {', '.join(self.ml_result.top_features[:3])}",
            ])

        if self.llm_result:
            lines.extend([
                f"  LLM 信号: {self.llm_result.signal.upper()} (置信度 {self.llm_result.confidence:.1%})",
                f"  LLM 关键因素: {', '.join(self.llm_result.key_factors[:3])}",
            ])

        lines.extend([
            f"{'-'*70}",
            f"  决策一致性: {'一致' if self.decision_consistency else '不一致'}",
            f"  总体对齐分数: {self.overall_alignment:.2f}",
            f"  互补性分数: {self.complementary_score:.2f}",
            f"{'-'*70}",
            f"  归因对齐:",
        ])

        for align in self.alignments[:5]:
            lines.append(
                f"    {align.feature:<20} ML:{align.ml_contribution:>7.2%}  "
                f"LLM:{align.llm_contribution:>7.1%}  [{align.alignment_type}]"
            )

        lines.extend([
            f"{'-'*70}",
            f"  综合解释: {self.explanation}",
            f"{'='*70}",
        ])

        return "\n".join(lines)


class AttributionComparator:
    """
    归因对比分析器。

    对比 ML 模型的统计归因与 LLM 的逻辑归因，
    揭示两者的对齐程度和互补性。

    使用方法:
        comparator = AttributionComparator()

        # 对比单次决策
        result = comparator.compare(
            ml_result=ml_explanation,
            llm_result=llm_explanation,
        )

        # 生成可视化
        comparator.plot_alignment_scatter(
            results=results,
            save_path="docs/figures/attribution_alignment.png",
        )
    """

    # ML 特征到 LLM 主题的映射
    FEATURE_THEME_MAPPING = {
        # 技术指标 -> 技术分析
        "RSI": "technical_analysis",
        "MACD": "technical_analysis",
        "MACD_Signal": "technical_analysis",
        "MACD_Hist": "technical_analysis",
        "Bollinger_Upper": "technical_analysis",
        "Bollinger_Lower": "technical_analysis",
        "ATR": "technical_analysis",
        "ADX": "technical_analysis",
        "SMA_5": "technical_analysis",
        "SMA_10": "technical_analysis",
        "SMA_20": "technical_analysis",
        "EMA_12": "technical_analysis",
        "EMA_26": "technical_analysis",
        "Volume_Ratio": "technical_analysis",
        "OBV": "technical_analysis",
        "CCI": "technical_analysis",
        "Williams_R": "technical_analysis",

        # 宏观指标 -> 宏观经济
        "VIX": "macro_economic",
        "VIX_Change": "macro_economic",
        "Interest_Rate": "macro_economic",
        "CPI": "macro_economic",
        "PMI": "macro_economic",

        # 情绪指标 -> 市场情绪
        "Sentiment_Score": "market_sentiment",
        "News_Sentiment": "market_sentiment",
        "Fear_Greed_Index": "market_sentiment",

        # 资金指标 -> 资金流向
        "Northbound_Flow": "capital_flow",
        "Volume": "capital_flow",
        "Turnover_Rate": "capital_flow",

        # 风险指标 -> 风险管理
        "Max_Drawdown": "risk_management",
        "Volatility": "risk_management",
        "Sharpe_Rolling": "risk_management",
    }

    def __init__(
        self,
        feature_theme_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        初始化对比器。

        Args:
            feature_theme_mapping: 自定义特征到主题的映射。
        """
        self.feature_theme_mapping = feature_theme_mapping or self.FEATURE_THEME_MAPPING

    def compare(
        self,
        ml_result: Optional[MLExplanationResult] = None,
        llm_result: Optional[LLMExplanationResult] = None,
    ) -> ComparisonResult:
        """
        对比 ML 和 LLM 的归因。

        Args:
            ml_result: ML 解释结果。
            llm_result: LLM 解释结果。

        Returns:
            ComparisonResult 对象。
        """
        result = ComparisonResult(
            ml_result=ml_result,
            llm_result=llm_result,
        )

        if ml_result is None and llm_result is None:
            return result

        # 检查决策一致性
        if ml_result and llm_result:
            ml_direction = ml_result.prediction_direction
            llm_direction = llm_result.signal
            result.decision_consistency = self._check_consistency(ml_direction, llm_direction)

        # 计算归因对齐
        result.alignments = self._compute_alignments(ml_result, llm_result)

        # 计算总体对齐分数
        if result.alignments:
            result.overall_alignment = np.mean([a.alignment_score for a in result.alignments])

        # 计算互补性分数
        result.complementary_score = self._compute_complementary_score(ml_result, llm_result)

        # 生成综合解释
        result.explanation = self._generate_explanation(ml_result, llm_result, result)

        return result

    def _check_consistency(self, ml_direction: str, llm_direction: str) -> bool:
        """检查决策一致性。"""
        # 标准化方向
        ml_dir = ml_direction.lower()
        llm_dir = llm_direction.lower()

        # 定义等价方向
        equivalents = {
            "buy": ["buy", "long", "买入", "看多"],
            "sell": ["sell", "short", "卖出", "看空"],
            "hold": ["hold", "neutral", "持有", "观望"],
        }

        for direction, equivalents_list in equivalents.items():
            if ml_dir in equivalents_list and llm_dir in equivalents_list:
                return True

        return ml_dir == llm_dir

    def _compute_alignments(
        self,
        ml_result: Optional[MLExplanationResult],
        llm_result: Optional[LLMExplanationResult],
    ) -> List[AttributionAlignment]:
        """计算归因对齐。"""
        alignments: List[AttributionAlignment] = []

        if ml_result is None or llm_result is None:
            return alignments

        # 获取 ML 特征贡献
        ml_contributions = {attr.feature_name: attr.contribution_pct for attr in ml_result.feature_attributions}

        # 获取 LLM 主题贡献
        llm_contributions = {attr.theme: attr.weight for attr in llm_result.theme_attributions}

        # 建立 ML 特征到 LLM 主题的映射
        feature_to_theme_contrib: Dict[str, Tuple[float, float]] = {}

        for feature, ml_contrib in ml_contributions.items():
            # 查找映射的主题
            theme = self.feature_theme_mapping.get(feature)

            if theme and theme in llm_contributions:
                llm_contrib = llm_contributions[theme]
                feature_to_theme_contrib[feature] = (ml_contrib, llm_contrib)
            else:
                # 未映射的特征，LLM 贡献设为 0
                feature_to_theme_contrib[feature] = (ml_contrib, 0.0)

        # 计算对齐分数
        for feature, (ml_contrib, llm_contrib) in feature_to_theme_contrib.items():
            # 对齐分数：两者贡献的相似程度
            if ml_contrib > 0 and llm_contrib > 0:
                # 使用调和平均或相似度度量
                alignment = 2 * ml_contrib * llm_contrib / (ml_contrib + llm_contrib) if (ml_contrib + llm_contrib) > 0 else 0
                alignment_type = "aligned"
            elif ml_contrib > 0 and llm_contrib == 0:
                alignment = 0
                alignment_type = "ml_only"
            elif ml_contrib == 0 and llm_contrib > 0:
                alignment = 0
                alignment_type = "llm_only"
            else:
                alignment = 0
                alignment_type = "neutral"

            alignments.append(AttributionAlignment(
                feature=feature,
                ml_contribution=ml_contrib,
                llm_contribution=llm_contrib,
                alignment_score=alignment,
                alignment_type=alignment_type,
            ))

        # 按对齐分数排序
        alignments.sort(key=lambda x: abs(x.alignment_score), reverse=True)

        return alignments

    def _compute_complementary_score(
        self,
        ml_result: Optional[MLExplanationResult],
        llm_result: Optional[LLMExplanationResult],
    ) -> float:
        """计算互补性分数。"""
        if ml_result is None or llm_result is None:
            return 0.0

        # 互补性体现在：
        # 1. 决策方向一致但依据不同
        # 2. 置信度互补（一个高一个低）
        # 3. 覆盖的因素不同

        score = 0.0

        # 置信度互补
        ml_conf = ml_result.confidence
        llm_conf = llm_result.confidence
        conf_diff = abs(ml_conf - llm_conf)
        if 0.2 < conf_diff < 0.5:  # 适度的置信度差异
            score += 0.3

        # 因素覆盖互补
        ml_features = set(a.feature_name for a in ml_result.feature_attributions[:5])
        llm_themes = set(a.theme for a in llm_result.theme_attributions[:3])

        # 检查 ML 特征是否有对应的 LLM 主题
        mapped_themes = set()
        for feature in ml_features:
            theme = self.feature_theme_mapping.get(feature)
            if theme:
                mapped_themes.add(theme)

        # 未覆盖的主题数量
        uncovered = len(llm_themes - mapped_themes)
        score += min(uncovered * 0.2, 0.4)

        # 决策一致性加分
        if self._check_consistency(ml_result.prediction_direction, llm_result.signal):
            score += 0.3

        return min(score, 1.0)

    def _generate_explanation(
        self,
        ml_result: Optional[MLExplanationResult],
        llm_result: Optional[LLMExplanationResult],
        comparison: ComparisonResult,
    ) -> str:
        """生成综合解释。"""
        parts = []

        if ml_result and llm_result:
            if comparison.decision_consistency:
                parts.append(f"ML 和 LLM 决策一致，均建议{ml_result.prediction_direction}")
            else:
                parts.append(f"ML 建议{ml_result.prediction_direction}，LLM 建议{llm_result.signal}，存在分歧")

            if comparison.overall_alignment > 0.5:
                parts.append("归因高度对齐，两者关注相似因素")
            elif comparison.overall_alignment > 0.2:
                parts.append("归因部分对齐，各有侧重")
            else:
                parts.append("归因差异较大，体现不同分析视角")

            if comparison.complementary_score > 0.5:
                parts.append("两者具有较强互补性，可综合考虑")

        elif ml_result:
            parts.append(f"仅 ML 分析：建议{ml_result.prediction_direction}，置信度 {ml_result.confidence:.0%}")

        elif llm_result:
            parts.append(f"仅 LLM 分析：建议{llm_result.signal}，置信度 {llm_result.confidence:.0%}")

        return "。".join(parts)

    def plot_alignment_scatter(
        self,
        results: List[ComparisonResult],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 8),
    ) -> plt.Figure:
        """
        绘制归因对齐散点图。

        Args:
            results: ComparisonResult 列表。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        if not results:
            logger.warning("没有对比结果可供绘图")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "没有对比数据", ha="center", va="center")
            return fig

        # 收集所有对齐数据
        data = []
        for result in results:
            for align in result.alignments:
                data.append({
                    "feature": align.feature,
                    "ml_contrib": align.ml_contribution,
                    "llm_contrib": align.llm_contribution,
                    "alignment": align.alignment_score,
                    "type": align.alignment_type,
                })

        df = pd.DataFrame(data)

        # 聚合相同特征
        df_agg = df.groupby("feature").agg({
            "ml_contrib": "mean",
            "llm_contrib": "mean",
            "alignment": "mean",
        }).reset_index()

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制散点图
        scatter = ax.scatter(
            df_agg["ml_contrib"],
            df_agg["llm_contrib"],
            c=df_agg["alignment"],
            cmap="RdYlGn",
            s=100,
            alpha=0.7,
            edgecolors="black",
        )

        # 添加对角线
        max_val = max(df_agg["ml_contrib"].max(), df_agg["llm_contrib"].max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="完全对齐")

        ax.set_xlabel("ML 贡献度 (SHAP)")
        ax.set_ylabel("LLM 贡献度 (主题权重)")
        ax.set_title("ML vs LLM 归因对齐散点图", fontsize=14, fontweight="bold")
        ax.legend()

        # 添加颜色条
        cbar = plt.colorbar(scatter)
        cbar.set_label("对齐分数")

        # 添加注释
        ax.text(
            0.02, 0.98,
            "右上区域: 高对齐\n左下区域: 低贡献\n偏离对角线: 分歧",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"归因对齐散点图已保存: {save_path}")

        return fig

    def plot_decision_consistency(
        self,
        results: List[ComparisonResult],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        """
        绘制决策一致性统计图。

        Args:
            results: ComparisonResult 列表。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        if not results:
            logger.warning("没有对比结果可供绘图")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "没有对比数据", ha="center", va="center")
            return fig

        # 统计决策组合
        decision_counts = {
            "一致买入": 0,
            "一致卖出": 0,
            "一致持有": 0,
            "ML买入/LLM卖出": 0,
            "ML卖出/LLM买入": 0,
            "其他分歧": 0,
        }

        for result in results:
            if result.ml_result and result.llm_result:
                ml_dir = result.ml_result.prediction_direction.lower()
                llm_dir = result.llm_result.signal.lower()

                if ml_dir == "buy" and llm_dir == "buy":
                    decision_counts["一致买入"] += 1
                elif ml_dir == "sell" and llm_dir == "sell":
                    decision_counts["一致卖出"] += 1
                elif ml_dir == "hold" and llm_dir == "hold":
                    decision_counts["一致持有"] += 1
                elif ml_dir == "buy" and llm_dir == "sell":
                    decision_counts["ML买入/LLM卖出"] += 1
                elif ml_dir == "sell" and llm_dir == "buy":
                    decision_counts["ML卖出/LLM买入"] += 1
                else:
                    decision_counts["其他分歧"] += 1

        # 绘图
        fig, ax = plt.subplots(figsize=figsize)

        labels = list(decision_counts.keys())
        values = list(decision_counts.values())
        colors = ["green", "red", "gray", "orange", "purple", "brown"]

        bars = ax.bar(labels, values, color=colors, edgecolor="black")

        ax.set_ylabel("次数")
        ax.set_title("ML vs LLM 决策一致性统计", fontsize=14, fontweight="bold")
        ax.set_xticklabels(labels, rotation=45, ha="right")

        # 添加数值标签
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(value), ha="center", va="bottom")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"决策一致性统计图已保存: {save_path}")

        return fig

    def generate_comparison_report(
        self,
        results: List[ComparisonResult],
    ) -> pd.DataFrame:
        """
        生成对比报告表格。

        Args:
            results: ComparisonResult 列表。

        Returns:
            对比报告 DataFrame。
        """
        rows = []

        for i, result in enumerate(results):
            row = {
                "序号": i + 1,
                "ML 预测": result.ml_result.prediction_direction if result.ml_result else "N/A",
                "ML 置信度": f"{result.ml_result.confidence:.1%}" if result.ml_result else "N/A",
                "LLM 信号": result.llm_result.signal if result.llm_result else "N/A",
                "LLM 置信度": f"{result.llm_result.confidence:.1%}" if result.llm_result else "N/A",
                "决策一致": "是" if result.decision_consistency else "否",
                "对齐分数": f"{result.overall_alignment:.2f}",
                "互补分数": f"{result.complementary_score:.2f}",
            }
            rows.append(row)

        return pd.DataFrame(rows)


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("  归因对比分析器示例")
    print("=" * 60)

    from src.evaluation.ml_explainer import MLExplanationResult, FeatureAttribution
    from src.evaluation.llm_explainer import LLMExplanationResult, ThemeAttribution

    # 创建模拟 ML 结果
    ml_result = MLExplanationResult(
        model_name="LightGBM",
        prediction_direction="buy",
        confidence=0.75,
        feature_attributions=[
            FeatureAttribution("RSI", 0.15, 0.25, "positive", 1),
            FeatureAttribution("MACD", 0.12, 0.20, "positive", 2),
            FeatureAttribution("Volume", 0.08, 0.15, "positive", 3),
        ],
        top_features=["RSI", "MACD", "Volume"],
    )

    # 创建模拟 LLM 结果
    llm_result = LLMExplanationResult(
        model_name="DeepSeek-V3.2",
        signal="buy",
        confidence=0.80,
        reasoning="技术指标显示买入信号",
        theme_attributions=[
            ThemeAttribution("technical_analysis", "技术分析", 10, ["RSI", "MACD"], 0.5, 0.8),
            ThemeAttribution("macro_economic", "宏观经济", 5, ["美联储"], 0.25, 0.6),
        ],
        key_factors=["RSI 超卖", "MACD 金叉"],
    )

    # 对比
    comparator = AttributionComparator()
    result = comparator.compare(ml_result, llm_result)

    print(result.summary())