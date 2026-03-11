"""
LLM 可解释性分析模块。

分析 LLM 的 reasoning 文本，提取关键主题和决策逻辑，
支持与 ML 模型的统计归因进行对比。

学术价值:
- 揭示 LLM 的"逻辑归因" vs ML 的"统计归因"
- 支持论文核心创新点：统计归因 vs 逻辑归因的对齐展示
- 解决量化"黑盒信任危机"
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Set
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.logger import get_logger

logger = get_logger(__name__)

# WordCloud 是可选依赖
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    logger.warning("WordCloud 未安装，词云功能不可用。请运行: uv add wordcloud")


# 定义主题关键词映射
THEME_KEYWORDS = {
    "technical_analysis": [
        "RSI", "MACD", "均线", "趋势", "突破", "支撑", "阻力", "布林带",
        "KDJ", "量价", "技术指标", "形态", "背离", "金叉", "死叉",
        "上升趋势", "下降趋势", "横盘", "震荡", "放量", "缩量",
    ],
    "macro_economic": [
        "美联储", "利率", "通胀", "CPI", "GDP", "就业", "经济数据",
        "货币政策", "财政政策", "PMI", "经济衰退", "经济复苏",
        "央行", "加息", "降息", "量化宽松", "鹰派", "鸽派",
    ],
    "market_sentiment": [
        "情绪", "恐慌", "贪婪", "乐观", "悲观", "市场信心",
        "投资者情绪", "风险偏好", "避险", "VIX", "恐慌指数",
    ],
    "news_events": [
        "新闻", "公告", "财报", "业绩", "利好", "利空",
        "政策", "监管", "事件", "突发", "消息面",
    ],
    "risk_management": [
        "风险", "止损", "仓位", "回撤", "波动", "杠杆",
        "风控", "止损位", "止盈", "持仓", "清仓",
    ],
    "fundamental_analysis": [
        "估值", "市盈率", "ROE", "盈利", "营收", "增长率",
        "基本面", "财务", "现金流", "负债", "资产",
    ],
    "capital_flow": [
        "资金", "流入", "流出", "北向", "南向", "外资",
        "主力", "散户", "机构", "成交量", "换手率",
    ],
}

# 中文停用词
CHINESE_STOPWORDS = {
    "的", "是", "在", "了", "和", "与", "或", "等", "这", "那",
    "有", "为", "对", "中", "上", "下", "不", "我", "你", "他",
    "其", "以", "及", "但", "而", "也", "就", "都", "又", "很",
    "能", "可", "要", "会", "应", "该", "当", "如", "若", "则",
    "并", "或", "且", "非", "无", "没", "到", "从", "被", "把",
    "让", "给", "向", "于", "由", "因", "所", "之", "着", "过",
}


@dataclass
class ThemeAttribution:
    """
    主题归因数据类。

    Attributes:
        theme: 主题名称。
        theme_name_cn: 主题中文名称。
        keyword_count: 关键词出现次数。
        keyword_list: 匹配的关键词列表。
        weight: 主题权重（归一化）。
        confidence: 该主题的置信度。
    """
    theme: str
    theme_name_cn: str
    keyword_count: int
    keyword_list: List[str] = field(default_factory=list)
    weight: float = 0.0
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "theme": self.theme,
            "theme_name_cn": self.theme_name_cn,
            "keyword_count": self.keyword_count,
            "keywords": self.keyword_list[:5],  # 只显示前5个
            "weight": f"{self.weight:.1%}",
            "confidence": f"{self.confidence:.2f}",
        }


@dataclass
class LLMExplanationResult:
    """
    LLM 可解释性分析结果数据类。

    Attributes:
        model_name: 模型名称。
        signal: 信号方向 ('buy' / 'sell' / 'hold')。
        confidence: 置信度。
        reasoning: 原始 reasoning 文本。
        theme_attributions: 主题归因列表。
        key_factors: 关键因素列表。
        summary: 摘要文本。
    """
    model_name: str
    signal: str
    confidence: float
    reasoning: str
    theme_attributions: List[ThemeAttribution] = field(default_factory=list)
    key_factors: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "model_name": self.model_name,
            "signal": self.signal,
            "confidence": f"{self.confidence:.1%}",
            "key_factors": self.key_factors,
            "summary": self.summary,
            "theme_attributions": [a.to_dict() for a in self.theme_attributions],
        }

    def summary_text(self) -> str:
        """生成解释摘要。"""
        lines = [
            f"\n{'='*60}",
            f"  {self.model_name} 决策解释",
            f"{'='*60}",
            f"  信号方向: {self.signal.upper()}",
            f"  置信度: {self.confidence:.1%}",
            f"{'-'*60}",
            f"  主题归因:",
        ]

        for attr in self.theme_attributions[:5]:
            if attr.weight > 0:
                lines.append(
                    f"    - {attr.theme_name_cn}: {attr.weight:.1%} "
                    f"(关键词: {', '.join(attr.keyword_list[:3])})"
                )

        lines.extend([
            f"{'-'*60}",
            f"  关键因素: {', '.join(self.key_factors[:5])}",
            f"{'-'*60}",
            f"  摘要: {self.summary}",
            f"{'='*60}",
        ])

        return "\n".join(lines)


class LLMExplainer:
    """
    LLM 可解释性分析器。

    从 LLM 的 reasoning 文本中提取关键主题和决策逻辑。

    使用方法:
        explainer = LLMExplainer()

        # 分析单个 reasoning
        result = explainer.analyze_reasoning(
            reasoning_text=reasoning,
            model_name="DeepSeek-V3.2",
            signal="buy",
            confidence=0.85,
        )

        # 批量分析
        results = explainer.batch_analyze(llm_cache_dir="docs/cache/llm_responses/")

        # 生成词云
        explainer.plot_reasoning_wordcloud(
            reasoning_list=reasoning_list,
            save_path="docs/figures/reasoning_wordcloud.png",
        )
    """

    THEME_NAMES_CN = {
        "technical_analysis": "技术分析",
        "macro_economic": "宏观经济",
        "market_sentiment": "市场情绪",
        "news_events": "新闻事件",
        "risk_management": "风险管理",
        "fundamental_analysis": "基本面分析",
        "capital_flow": "资金流向",
    }

    def __init__(
        self,
        theme_keywords: Optional[Dict[str, List[str]]] = None,
        stopwords: Optional[Set[str]] = None,
    ) -> None:
        """
        初始化分析器。

        Args:
            theme_keywords: 自定义主题关键词映射。
            stopwords: 自定义停用词集合。
        """
        self.theme_keywords = theme_keywords or THEME_KEYWORDS
        self.stopwords = stopwords or CHINESE_STOPWORDS

    def analyze_reasoning(
        self,
        reasoning_text: str,
        model_name: str = "LLM",
        signal: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> LLMExplanationResult:
        """
        分析 reasoning 文本。

        Args:
            reasoning_text: reasoning 文本。
            model_name: 模型名称。
            signal: 信号方向（如果为 None 则尝试从文本提取）。
            confidence: 置信度（如果为 None 则尝试从文本提取）。

        Returns:
            LLMExplanationResult 对象。
        """
        if not reasoning_text:
            return LLMExplanationResult(
                model_name=model_name,
                signal="hold",
                confidence=0.5,
                reasoning="",
                summary="无 reasoning 内容",
            )

        # 提取信号和置信度
        if signal is None:
            signal = self._extract_signal(reasoning_text)

        if confidence is None:
            confidence = self._extract_confidence(reasoning_text)

        # 分析主题
        theme_attributions = self._analyze_themes(reasoning_text)

        # 提取关键因素
        key_factors = self._extract_key_factors(reasoning_text)

        # 生成摘要
        summary = self._generate_summary(reasoning_text, signal, confidence, theme_attributions)

        return LLMExplanationResult(
            model_name=model_name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning_text,
            theme_attributions=theme_attributions,
            key_factors=key_factors,
            summary=summary,
        )

    def _extract_signal(self, text: str) -> str:
        """从文本中提取信号方向。"""
        text_lower = text.lower()

        buy_keywords = ["买入", "buy", "看多", "做多", "增持", "加仓", "推荐买入"]
        sell_keywords = ["卖出", "sell", "看空", "做空", "减持", "减仓", "清仓", "推荐卖出"]
        hold_keywords = ["持有", "hold", "观望", "中性", "等待"]

        buy_count = sum(1 for k in buy_keywords if k in text_lower)
        sell_count = sum(1 for k in sell_keywords if k in text_lower)
        hold_count = sum(1 for k in hold_keywords if k in text_lower)

        if buy_count > sell_count and buy_count > hold_count:
            return "buy"
        elif sell_count > buy_count and sell_count > hold_count:
            return "sell"
        else:
            return "hold"

    def _extract_confidence(self, text: str) -> float:
        """从文本中提取置信度。"""
        # 尝试匹配置信度数字
        patterns = [
            r"置信度[：:]\s*(\d+(?:\.\d+)?)\s*%?",
            r"confidence[：:]\s*(\d+(?:\.\d+)?)\s*%?",
            r"(\d+(?:\.\d+)?)\s*%?\s*(?:的)?(?:置信度|把握)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                conf = float(match.group(1))
                # 如果是百分数形式
                if conf > 1:
                    conf = conf / 100
                return min(max(conf, 0), 1)

        # 默认置信度
        return 0.7

    def _analyze_themes(self, text: str) -> List[ThemeAttribution]:
        """分析主题归因。"""
        attributions: List[ThemeAttribution] = []
        total_keywords = 0

        for theme, keywords in self.theme_keywords.items():
            matched_keywords = []
            count = 0

            for keyword in keywords:
                if keyword.lower() in text.lower():
                    count += text.lower().count(keyword.lower())
                    matched_keywords.append(keyword)

            if count > 0:
                total_keywords += count
                attributions.append(ThemeAttribution(
                    theme=theme,
                    theme_name_cn=self.THEME_NAMES_CN.get(theme, theme),
                    keyword_count=count,
                    keyword_list=matched_keywords,
                    weight=0.0,  # 后续归一化
                    confidence=min(count / 10, 1.0),
                ))

        # 归一化权重
        if total_keywords > 0:
            for attr in attributions:
                attr.weight = attr.keyword_count / total_keywords

        # 按权重排序
        attributions.sort(key=lambda x: x.weight, reverse=True)

        return attributions

    def _extract_key_factors(self, text: str) -> List[str]:
        """提取关键因素。"""
        # 关键词模式
        factor_patterns = [
            r"主要[因素原因][：:]\s*([^\n]+)",
            r"关键[因素指标][：:]\s*([^\n]+)",
            r"核心[因素观点][：:]\s*([^\n]+)",
            r"基于[：:]\s*([^\n]+)",
        ]

        factors = []
        for pattern in factor_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 分割并清理
                items = re.split(r"[,，、；;]", match)
                for item in items:
                    item = item.strip()
                    if item and len(item) > 2:
                        factors.append(item)

        # 如果没有匹配到，从主题中提取
        if not factors:
            theme_attrs = self._analyze_themes(text)
            factors = [attr.theme_name_cn for attr in theme_attrs[:3]]

        return factors[:10]

    def _generate_summary(
        self,
        text: str,
        signal: str,
        confidence: float,
        theme_attributions: List[ThemeAttribution],
    ) -> str:
        """生成摘要。"""
        signal_cn = {"buy": "买入", "sell": "卖出", "hold": "持有"}.get(signal, "持有")

        if theme_attributions:
            top_theme = theme_attributions[0]
            summary = (
                f"LLM 建议{signal_cn}，置信度 {confidence:.0%}，"
                f"主要基于{top_theme.theme_name_cn}分析"
            )
            if len(theme_attributions) > 1:
                summary += f"，次要考虑{theme_attributions[1].theme_name_cn}"
        else:
            summary = f"LLM 建议{signal_cn}，置信度 {confidence:.0%}"

        return summary

    def batch_analyze(
        self,
        llm_cache_dir: str,
        model_filter: Optional[str] = None,
    ) -> List[LLMExplanationResult]:
        """
        批量分析 LLM 缓存文件。

        Args:
            llm_cache_dir: LLM 缓存目录。
            model_filter: 模型过滤器，如果指定则只分析该模型。

        Returns:
            LLMExplanationResult 列表。
        """
        cache_path = Path(llm_cache_dir)
        if not cache_path.exists():
            logger.warning(f"LLM 缓存目录不存在: {cache_path}")
            return []

        results: List[LLMExplanationResult] = []

        for cache_file in cache_path.glob("*.jsonl"):
            # 过滤模型
            if model_filter and model_filter not in cache_file.stem:
                continue

            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        reasoning = data.get("reasoning", "")
                        signal = data.get("signal", data.get("decision", ""))
                        confidence = data.get("confidence", 0.7)
                        model_name = data.get("model", cache_file.stem)

                        result = self.analyze_reasoning(
                            reasoning_text=reasoning,
                            model_name=model_name,
                            signal=signal if isinstance(signal, str) else None,
                            confidence=confidence if isinstance(confidence, (int, float)) else None,
                        )
                        results.append(result)
                    except json.JSONDecodeError:
                        continue

        logger.info(f"批量分析完成: {len(results)} 条记录")
        return results

    def plot_reasoning_wordcloud(
        self,
        reasoning_list: List[str],
        save_path: Optional[str] = None,
        max_words: int = 200,
        figsize: tuple = (12, 8),
        colormap: str = "viridis",
    ) -> Optional[plt.Figure]:
        """
        绘制 reasoning 词云。

        Args:
            reasoning_list: reasoning 文本列表。
            save_path: 保存路径。
            max_words: 最大词数。
            figsize: 图表尺寸。
            colormap: 颜色映射。

        Returns:
            matplotlib Figure 对象（如果成功）。
        """
        if not WORDCLOUD_AVAILABLE:
            logger.warning("WordCloud 未安装，无法生成词云")
            return None

        if not reasoning_list:
            logger.warning("没有 reasoning 数据可供生成词云")
            return None

        # 合并所有文本
        combined_text = " ".join(reasoning_list)

        # 创建词云
        try:
            wordcloud = WordCloud(
                font_path=None,  # 使用默认字体
                width=figsize[0] * 100,
                height=figsize[1] * 100,
                max_words=max_words,
                background_color="white",
                colormap=colormap,
                stopwords=self.stopwords,
                collocations=False,
            ).generate(combined_text)

            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("LLM Reasoning 关键词词云", fontsize=14, fontweight="bold")

            plt.tight_layout()

            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                logger.info(f"Reasoning 词云已保存: {save_path}")

            return fig

        except Exception as e:
            logger.error(f"词云生成失败: {e}")
            return None

    def plot_theme_distribution(
        self,
        results: List[LLMExplanationResult],
        save_path: Optional[str] = None,
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        """
        绘制主题分布条形图。

        Args:
            results: LLMExplanationResult 列表。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        # 统计主题权重
        theme_weights: Dict[str, float] = {}
        for result in results:
            for attr in result.theme_attributions:
                if attr.theme not in theme_weights:
                    theme_weights[attr.theme] = 0.0
                theme_weights[attr.theme] += attr.weight

        # 归一化
        total = sum(theme_weights.values())
        if total > 0:
            for theme in theme_weights:
                theme_weights[theme] /= total

        # 排序
        sorted_themes = sorted(theme_weights.items(), key=lambda x: x[1], reverse=True)

        # 绘图
        fig, ax = plt.subplots(figsize=figsize)

        themes = [self.THEME_NAMES_CN.get(t, t) for t, _ in sorted_themes]
        weights = [w for _, w in sorted_themes]

        colors = plt.cm.Set2(np.linspace(0, 1, len(themes)))
        bars = ax.barh(themes, weights, color=colors, edgecolor="black")

        ax.set_xlabel("权重")
        ax.set_title("LLM Reasoning 主题分布", fontsize=14, fontweight="bold")
        ax.invert_yaxis()

        # 添加数值标签
        for bar, weight in zip(bars, weights):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{weight:.1%}", va="center", fontsize=10)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"主题分布图已保存: {save_path}")

        return fig

    def compare_themes_by_signal(
        self,
        results: List[LLMExplanationResult],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 8),
    ) -> plt.Figure:
        """
        对比不同信号方向的主题分布。

        Args:
            results: LLMExplanationResult 列表。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        # 按信号分组
        signals = {"buy": [], "sell": [], "hold": []}
        for result in results:
            if result.signal in signals:
                signals[result.signal].append(result)

        # 计算各信号的主题权重
        signal_themes: Dict[str, Dict[str, float]] = {}
        for signal, signal_results in signals.items():
            theme_weights: Dict[str, float] = {}
            for result in signal_results:
                for attr in result.theme_attributions:
                    if attr.theme not in theme_weights:
                        theme_weights[attr.theme] = 0.0
                    theme_weights[attr.theme] += attr.weight

            total = sum(theme_weights.values())
            if total > 0:
                for theme in theme_weights:
                    theme_weights[theme] /= total

            signal_themes[signal] = theme_weights

        # 准备绘图数据
        all_themes = list(self.theme_keywords.keys())
        x = np.arange(len(all_themes))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)

        colors = {"buy": "green", "sell": "red", "hold": "gray"}

        for i, (signal, theme_weights) in enumerate(signal_themes.items()):
            weights = [theme_weights.get(t, 0) for t in all_themes]
            ax.bar(x + i * width, weights, width, label=signal.upper(), color=colors[signal], alpha=0.7)

        ax.set_xlabel("主题")
        ax.set_ylabel("权重")
        ax.set_title("不同信号方向的主题分布对比", fontsize=14, fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels([self.THEME_NAMES_CN.get(t, t) for t in all_themes], rotation=45, ha="right")
        ax.legend()

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"信号主题对比图已保存: {save_path}")

        return fig


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("  LLM 可解释性分析器示例")
    print("=" * 60)

    # 示例 reasoning
    sample_reasoning = """
    基于技术分析，当前RSI指标显示超卖状态，MACD出现金叉信号。
    同时，美联储近期鸽派言论降低了加息预期，宏观环境改善。
    建议：买入
    置信度：75%
    主要因素：技术指标改善、宏观政策利好
    """

    explainer = LLMExplainer()
    result = explainer.analyze_reasoning(
        reasoning_text=sample_reasoning,
        model_name="DeepSeek-V3.2",
    )

    print(result.summary_text())