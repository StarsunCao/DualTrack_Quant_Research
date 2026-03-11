"""
评估模块。

提供金融指标和工程指标的计算和可视化功能。

主要组件：
- MetricsCalculator: 多维度指标计算器
- MultiStrategyComparator: 多策略对比器
- Visualizer: 论文图表生成器

高级评估组件（Phase 2 新增）：
- TradeAnalyzer: 交易质量分析器（MAE/MFE）
- MarketStateAnalyzer: 市场状态切割分析器
- MLExplainer: ML 模型可解释性分析器（SHAP）
- LLMExplainer: LLM 可解释性分析器
- CrossMarketAnalyzer: 跨市场规则敏感度分析器
- AttributionComparator: 归因对比分析器
- AdvancedVisualizer: 高级可视化器

金融指标：
- Sharpe Ratio: 夏普比率
- Maximum Drawdown: 最大回撤
- Win Rate: 胜率
- Sortino Ratio: 索提诺比率
- Calmar Ratio: 卡玛比率

工程指标：
- Latency: 推理延迟
- Throughput: 吞吐量
- Cost-per-Alpha: Alpha 信号成本
- Token Efficiency: Token 效率

学术指标（新增）：
- MAE: 最大不利偏移
- MFE: 最大有利偏移
- Trade Efficiency: 交易效率
- Zero-shot Score: 零样本泛化评分
- Attribution Alignment: 归因对齐分数
"""

from .metrics_calculator import (
    FinancialMetrics,
    EngineeringMetrics,
    EvaluationResult,
    MetricsCalculator,
    MultiStrategyComparator,
    calculate_metrics_from_backtest,
)

from .visualizer import (
    plot_equity_curves,
    plot_drawdown_heatmap,
    plot_latency_boxplot,
    plot_metrics_comparison,
    plot_underwater,
    plot_rolling_sharpe,
    generate_all_figures,
)

# 高级评估模块
from .trade_analyzer import (
    TradeQualityMetrics,
    TradeQualitySummary,
    TradeAnalyzer,
)

from .market_state_analyzer import (
    MarketState,
    MarketStateMetrics,
    MarketStateSummary,
    MarketStateAnalyzer,
)

from .ml_explainer import (
    FeatureAttribution,
    MLExplanationResult,
    MLExplainer,
)

from .llm_explainer import (
    ThemeAttribution,
    LLMExplanationResult,
    LLMExplainer,
)

from .cross_market_analyzer import (
    SignalDecayResult,
    CrossMarketSummary,
    CrossMarketAnalyzer,
)

from .attribution_comparator import (
    AttributionAlignment,
    ComparisonResult,
    AttributionComparator,
)

from .advanced_visualizer import (
    VisualizationConfig,
    AdvancedVisualizer,
)

__all__ = [
    # 基础指标类
    "FinancialMetrics",
    "EngineeringMetrics",
    "EvaluationResult",
    # 基础计算器
    "MetricsCalculator",
    "MultiStrategyComparator",
    "calculate_metrics_from_backtest",
    # 基础可视化函数
    "plot_equity_curves",
    "plot_drawdown_heatmap",
    "plot_latency_boxplot",
    "plot_metrics_comparison",
    "plot_underwater",
    "plot_rolling_sharpe",
    "generate_all_figures",
    # 高级评估 - 交易质量
    "TradeQualityMetrics",
    "TradeQualitySummary",
    "TradeAnalyzer",
    # 高级评估 - 市场状态
    "MarketState",
    "MarketStateMetrics",
    "MarketStateSummary",
    "MarketStateAnalyzer",
    # 高级评估 - ML 可解释性
    "FeatureAttribution",
    "MLExplanationResult",
    "MLExplainer",
    # 高级评估 - LLM 可解释性
    "ThemeAttribution",
    "LLMExplanationResult",
    "LLMExplainer",
    # 高级评估 - 跨市场
    "SignalDecayResult",
    "CrossMarketSummary",
    "CrossMarketAnalyzer",
    # 高级评估 - 归因对比
    "AttributionAlignment",
    "ComparisonResult",
    "AttributionComparator",
    # 高级可视化
    "VisualizationConfig",
    "AdvancedVisualizer",
]