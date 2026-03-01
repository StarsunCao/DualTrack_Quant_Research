"""
评估模块。

提供金融指标和工程指标的计算和可视化功能。

主要组件：
- MetricsCalculator: 多维度指标计算器
- MultiStrategyComparator: 多策略对比器
- Visualizer: 论文图表生成器

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

__all__ = [
    # 指标类
    "FinancialMetrics",
    "EngineeringMetrics",
    "EvaluationResult",
    # 计算器
    "MetricsCalculator",
    "MultiStrategyComparator",
    "calculate_metrics_from_backtest",
    # 可视化函数
    "plot_equity_curves",
    "plot_drawdown_heatmap",
    "plot_latency_boxplot",
    "plot_metrics_comparison",
    "plot_underwater",
    "plot_rolling_sharpe",
    "generate_all_figures",
]