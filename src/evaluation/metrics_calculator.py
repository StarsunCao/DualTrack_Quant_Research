"""
多维度指标计算器模块。

计算金融指标和工程指标，支持论文所需的对比分析。

金融指标：
- Sharpe Ratio: 风险调整收益
- Maximum Drawdown: 最大回撤
- Win Rate: 胜率
- Calmar Ratio: 卡玛比率
- Sortino Ratio: 索提诺比率
- Alpha/Beta: 相对基准的超额收益

工程指标：
- Latency: 平均推理延迟
- Throughput: 吞吐量
- Cost-per-Alpha: 每个 Alpha 信号的 API 成本
- Token Efficiency: Token 使用效率
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from pathlib import Path
import json


# ============================================================================
# 数据类定义
# ============================================================================
@dataclass
class FinancialMetrics:
    """
    金融指标数据类。

    Attributes:
        total_return: 总收益率。
        annual_return: 年化收益率。
        sharpe_ratio: 夏普比率。
        sortino_ratio: 索提诺比率。
        calmar_ratio: 卡玛比率。
        max_drawdown: 最大回撤。
        max_drawdown_duration: 最大回撤持续期（天）。
        win_rate: 胜率。
        profit_factor: 盈亏比。
        avg_trade_return: 平均单笔交易收益。
        volatility: 年化波动率。
        alpha: 相对基准的 Alpha。
        beta: 相对基准的 Beta。
    """
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    volatility: float = 0.0
    alpha: Optional[float] = None
    beta: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "total_return": f"{self.total_return:.2%}",
            "annual_return": f"{self.annual_return:.2%}",
            "sharpe_ratio": f"{self.sharpe_ratio:.4f}",
            "sortino_ratio": f"{self.sortino_ratio:.4f}",
            "calmar_ratio": f"{self.calmar_ratio:.4f}",
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "max_drawdown_duration": f"{self.max_drawdown_duration} days",
            "win_rate": f"{self.win_rate:.2%}",
            "profit_factor": f"{self.profit_factor:.4f}",
            "avg_trade_return": f"{self.avg_trade_return:.4%}",
            "volatility": f"{self.volatility:.2%}",
            "alpha": f"{self.alpha:.4f}" if self.alpha is not None else None,
            "beta": f"{self.beta:.4f}" if self.beta is not None else None,
        }

    def summary(self) -> str:
        """生成指标摘要字符串。"""
        lines = [
            "金融指标摘要:",
            f"  总收益率:     {self.total_return:>10.2%}",
            f"  年化收益率:   {self.annual_return:>10.2%}",
            f"  夏普比率:     {self.sharpe_ratio:>10.4f}",
            f"  索提诺比率:   {self.sortino_ratio:>10.4f}",
            f"  卡玛比率:     {self.calmar_ratio:>10.4f}",
            f"  最大回撤:     {self.max_drawdown:>10.2%}",
            f"  胜率:         {self.win_rate:>10.2%}",
            f"  盈亏比:       {self.profit_factor:>10.4f}",
        ]
        return "\n".join(lines)


@dataclass
class EngineeringMetrics:
    """
    工程指标数据类。

    Attributes:
        avg_latency_ms: 平均推理延迟（毫秒）。
        p50_latency_ms: P50 延迟（毫秒）。
        p95_latency_ms: P95 延迟（毫秒）。
        p99_latency_ms: P99 延迟（毫秒）。
        throughput: 吞吐量（请求/秒）。
        total_tokens: 总 Token 消耗。
        prompt_tokens: Prompt Token 数量。
        completion_tokens: Completion Token 数量。
        total_cost_usd: 总 API 成本（美元）。
        cost_per_alpha: 每个 Alpha 信号成本。
        cost_per_trade: 每笔交易成本。
        token_efficiency: Token 效率（Alpha 信号 / Token）。
        cache_hit_rate: 缓存命中率。
    """
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput: float = 0.0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = 0.0
    cost_per_alpha: float = 0.0
    cost_per_trade: float = 0.0
    token_efficiency: float = 0.0
    cache_hit_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "avg_latency_ms": f"{self.avg_latency_ms:.2f}",
            "p50_latency_ms": f"{self.p50_latency_ms:.2f}",
            "p95_latency_ms": f"{self.p95_latency_ms:.2f}",
            "p99_latency_ms": f"{self.p99_latency_ms:.2f}",
            "throughput": f"{self.throughput:.2f} req/s",
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_cost_usd": f"${self.total_cost_usd:.4f}",
            "cost_per_alpha": f"${self.cost_per_alpha:.6f}",
            "cost_per_trade": f"${self.cost_per_trade:.4f}",
            "token_efficiency": f"{self.token_efficiency:.6f}",
            "cache_hit_rate": f"{self.cache_hit_rate:.2%}",
        }

    def summary(self) -> str:
        """生成指标摘要字符串。"""
        lines = [
            "工程指标摘要:",
            f"  平均延迟:     {self.avg_latency_ms:>10.2f} ms",
            f"  P95 延迟:     {self.p95_latency_ms:>10.2f} ms",
            f"  吞吐量:       {self.throughput:>10.2f} req/s",
            f"  总 Token:     {self.total_tokens:>10,}",
            f"  API 成本:     ${self.total_cost_usd:>9.4f}",
            f"  Cost/Alpha:   ${self.cost_per_alpha:>9.6f}",
        ]
        return "\n".join(lines)


@dataclass
class EvaluationResult:
    """
    评估结果数据类。

    Attributes:
        strategy_name: 策略名称。
        financial_metrics: 金融指标。
        engineering_metrics: 工程指标。
        equity_curve: 净值曲线。
        trade_log: 交易日志。
        latency_log: 延迟日志。
        metadata: 元数据。
    """
    strategy_name: str
    financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
    engineering_metrics: EngineeringMetrics = field(default_factory=EngineeringMetrics)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_log: pd.DataFrame = field(default_factory=pd.DataFrame)
    latency_log: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "strategy_name": self.strategy_name,
            "financial_metrics": self.financial_metrics.to_dict(),
            "engineering_metrics": self.engineering_metrics.to_dict(),
            "metadata": self.metadata,
        }


# ============================================================================
# 指标计算器类
# ============================================================================
class MetricsCalculator:
    """
    多维度指标计算器。

    计算金融指标和工程指标，支持论文对比分析。

    使用方法:
        calculator = MetricsCalculator(risk_free_rate=0.02)

        # 从净值曲线计算金融指标
        financial = calculator.calculate_financial_metrics(
            equity_curve=equity_df,
            trade_log=trade_df,
        )

        # 从延迟日志计算工程指标
        engineering = calculator.calculate_engineering_metrics(
            latency_log=latency_list,
            token_log=token_list,
            num_signals=100,
        )
    """

    # API 定价（美元/1K tokens）
    PRICING = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-3.5-turbo": {"prompt": 0.0005, "completion": 0.0015},
        "deepseek-chat": {"prompt": 0.0001, "completion": 0.0002},
        "deepseek-reasoner": {"prompt": 0.00055, "completion": 0.00219},
        "ollama": {"prompt": 0.0, "completion": 0.0},  # 本地免费
    }

    def __init__(
        self,
        risk_free_rate: float = 0.02,
        trading_days_per_year: int = 252,
        default_model: str = "deepseek-chat",
    ) -> None:
        """
        初始化指标计算器。

        Args:
            risk_free_rate: 无风险利率，默认 2%。
            trading_days_per_year: 每年交易日数，默认 252。
            default_model: 默认 LLM 模型，用于成本计算。
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        self.default_model = default_model

    # ========================================================================
    # 金融指标计算
    # ========================================================================
    def calculate_financial_metrics(
        self,
        equity_curve: pd.DataFrame,
        trade_log: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> FinancialMetrics:
        """
        计算金融指标。

        Args:
            equity_curve: 净值曲线 DataFrame，需包含 'value' 或 'nav' 列。
            trade_log: 交易日志 DataFrame，需包含 'pnl' 列。
            benchmark_returns: 基准收益率序列，用于计算 Alpha/Beta。

        Returns:
            FinancialMetrics 对象。
        """
        # 确保净值列存在
        if "nav" in equity_curve.columns:
            nav = equity_curve["nav"]
        elif "value" in equity_curve.columns:
            nav = equity_curve["value"] / equity_curve["value"].iloc[0]
        else:
            raise ValueError("equity_curve 必须包含 'nav' 或 'value' 列")

        # 计算日收益率
        returns = nav.pct_change().dropna()

        # 总收益率
        total_return = nav.iloc[-1] / nav.iloc[0] - 1

        # 年化收益率
        num_days = len(nav)
        years = max(num_days / self.trading_days_per_year, 1)
        annual_return = (1 + total_return) ** (1 / years) - 1

        # 年化波动率
        volatility = returns.std() * np.sqrt(self.trading_days_per_year)

        # 夏普比率
        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess_returns = returns - daily_rf
        sharpe_ratio = (
            excess_returns.mean() / excess_returns.std() * np.sqrt(self.trading_days_per_year)
            if excess_returns.std() > 0 else 0.0
        )

        # 索提诺比率（只考虑下行风险）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0.001
        sortino_ratio = (
            (returns.mean() - daily_rf) / downside_std * np.sqrt(self.trading_days_per_year)
            if downside_std > 0 else 0.0
        )

        # 最大回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        # 最大回撤持续期
        max_dd_idx = drawdown.idxmin()
        if isinstance(max_dd_idx, pd.Timestamp):
            # 找到回撤开始的点
            peak_idx = nav.loc[:max_dd_idx].idxmax()
            # 找到恢复的点
            recovery_mask = nav.loc[max_dd_idx:] >= nav.loc[peak_idx]
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                max_dd_duration = len(nav.loc[peak_idx:recovery_idx])
            else:
                max_dd_duration = len(nav.loc[peak_idx:])
        else:
            max_dd_duration = 0

        # 卡玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0.0

        # 交易统计
        win_rate = 0.0
        profit_factor = 0.0
        avg_trade_return = 0.0

        if trade_log is not None and len(trade_log) > 0:
            # 确保有 PnL 列
            pnl_col = None
            for col in ["pnl", "pnl_comm", "profit", "return"]:
                if col in trade_log.columns:
                    pnl_col = col
                    break

            if pnl_col:
                trade_pnl = trade_log[pnl_col]

                # 胜率
                winning_trades = (trade_pnl > 0).sum()
                total_trades = len(trade_pnl)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

                # 盈亏比
                total_profit = trade_pnl[trade_pnl > 0].sum()
                total_loss = abs(trade_pnl[trade_pnl < 0].sum())
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0

                # 平均单笔收益
                avg_trade_return = trade_pnl.mean()

        # Alpha/Beta（如果有基准）
        alpha = None
        beta = None
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            # 计算 Beta
            covariance = returns.cov(benchmark_returns)
            benchmark_variance = benchmark_returns.var()
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0.0

            # 计算 Alpha（年化）
            alpha = annual_return - self.risk_free_rate - beta * (
                benchmark_returns.mean() * self.trading_days_per_year - self.risk_free_rate
            )

        return FinancialMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            alpha=alpha,
            beta=beta,
        )

    # ========================================================================
    # 工程指标计算
    # ========================================================================
    def calculate_engineering_metrics(
        self,
        latency_log: Optional[List[float]] = None,
        token_log: Optional[List[Dict[str, int]]] = None,
        num_signals: int = 1,
        num_trades: int = 1,
        cache_hits: int = 0,
        cache_misses: int = 0,
        model: Optional[str] = None,
    ) -> EngineeringMetrics:
        """
        计算工程指标。

        Args:
            latency_log: 延迟日志列表（毫秒）。
            token_log: Token 日志列表，每个元素包含 'prompt_tokens' 和 'completion_tokens'。
            num_signals: 生成的信号总数。
            num_trades: 执行的交易总数。
            cache_hits: 缓存命中次数。
            cache_misses: 缓存未命中次数。
            model: 使用的模型名称。

        Returns:
            EngineeringMetrics 对象。
        """
        latency_log = latency_log or []
        token_log = token_log or []
        model = model or self.default_model

        # 延迟统计
        if latency_log:
            latencies = np.array(latency_log)
            avg_latency = np.mean(latencies)
            p50_latency = np.percentile(latencies, 50)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            # 吞吐量（假设总时间 = 延迟总和）
            total_time_s = np.sum(latencies) / 1000
            throughput = len(latencies) / total_time_s if total_time_s > 0 else 0.0
        else:
            avg_latency = p50_latency = p95_latency = p99_latency = 0.0
            throughput = 0.0

        # Token 统计
        total_prompt = sum(t.get("prompt_tokens", 0) for t in token_log)
        total_completion = sum(t.get("completion_tokens", 0) for t in token_log)
        total_tokens = total_prompt + total_completion

        # 成本计算
        pricing = self.PRICING.get(model, self.PRICING["deepseek-chat"])
        prompt_cost = total_prompt / 1000 * pricing["prompt"]
        completion_cost = total_completion / 1000 * pricing["completion"]
        total_cost = prompt_cost + completion_cost

        # Cost-per-Alpha
        cost_per_alpha = total_cost / num_signals if num_signals > 0 else 0.0

        # Cost-per-Trade
        cost_per_trade = total_cost / num_trades if num_trades > 0 else 0.0

        # Token 效率
        token_efficiency = num_signals / total_tokens if total_tokens > 0 else 0.0

        # 缓存命中率
        total_cache_requests = cache_hits + cache_misses
        cache_hit_rate = cache_hits / total_cache_requests if total_cache_requests > 0 else 0.0

        return EngineeringMetrics(
            avg_latency_ms=avg_latency,
            p50_latency_ms=p50_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            throughput=throughput,
            total_tokens=total_tokens,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_cost_usd=total_cost,
            cost_per_alpha=cost_per_alpha,
            cost_per_trade=cost_per_trade,
            token_efficiency=token_efficiency,
            cache_hit_rate=cache_hit_rate,
        )

    # ========================================================================
    # 综合评估
    # ========================================================================
    def evaluate(
        self,
        strategy_name: str,
        equity_curve: pd.DataFrame,
        trade_log: Optional[pd.DataFrame] = None,
        latency_log: Optional[List[float]] = None,
        token_log: Optional[List[Dict[str, int]]] = None,
        benchmark_returns: Optional[pd.Series] = None,
        num_signals: int = 1,
        cache_hits: int = 0,
        cache_misses: int = 0,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EvaluationResult:
        """
        执行综合评估。

        Args:
            strategy_name: 策略名称。
            equity_curve: 净值曲线。
            trade_log: 交易日志。
            latency_log: 延迟日志。
            token_log: Token 日志。
            benchmark_returns: 基准收益率。
            num_signals: 信号总数。
            cache_hits: 缓存命中次数。
            cache_misses: 缓存未命中次数。
            model: 模型名称。
            metadata: 额外元数据。

        Returns:
            EvaluationResult 对象。
        """
        # 计算金融指标
        financial = self.calculate_financial_metrics(
            equity_curve=equity_curve,
            trade_log=trade_log,
            benchmark_returns=benchmark_returns,
        )

        # 计算工程指标
        num_trades = len(trade_log) if trade_log is not None else 1
        engineering = self.calculate_engineering_metrics(
            latency_log=latency_log,
            token_log=token_log,
            num_signals=num_signals,
            num_trades=num_trades,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            model=model,
        )

        return EvaluationResult(
            strategy_name=strategy_name,
            financial_metrics=financial,
            engineering_metrics=engineering,
            equity_curve=equity_curve,
            trade_log=trade_log if trade_log is not None else pd.DataFrame(),
            latency_log=latency_log or [],
            metadata=metadata or {},
        )


# ============================================================================
# 多策略对比器
# ============================================================================
class MultiStrategyComparator:
    """
    多策略对比器。

    用于生成论文所需的对比分析表格。

    使用方法:
        comparator = MultiStrategyComparator()

        # 添加各策略评估结果
        comparator.add_result("ML_Baseline", ml_result)
        comparator.add_result("LLM_Baseline", llm_result)
        comparator.add_result("DualTrack_Fusion", fusion_result)

        # 生成对比表格
        financial_df = comparator.compare_financial_metrics()
        engineering_df = comparator.compare_engineering_metrics()
    """

    def __init__(self) -> None:
        """初始化对比器。"""
        self.results: Dict[str, EvaluationResult] = {}

    def add_result(self, strategy_name: str, result: EvaluationResult) -> None:
        """添加策略评估结果。"""
        self.results[strategy_name] = result

    def compare_financial_metrics(self) -> pd.DataFrame:
        """
        对比金融指标。

        Returns:
            金融指标对比 DataFrame。
        """
        rows = []
        for name, result in self.results.items():
            fm = result.financial_metrics
            rows.append({
                "Strategy": name,
                "Total Return": fm.total_return,
                "Annual Return": fm.annual_return,
                "Sharpe Ratio": fm.sharpe_ratio,
                "Sortino Ratio": fm.sortino_ratio,
                "Calmar Ratio": fm.calmar_ratio,
                "Max Drawdown": fm.max_drawdown,
                "Win Rate": fm.win_rate,
                "Profit Factor": fm.profit_factor,
                "Volatility": fm.volatility,
            })
        return pd.DataFrame(rows).set_index("Strategy")

    def compare_engineering_metrics(self) -> pd.DataFrame:
        """
        对比工程指标。

        Returns:
            工程指标对比 DataFrame。
        """
        rows = []
        for name, result in self.results.items():
            em = result.engineering_metrics
            rows.append({
                "Strategy": name,
                "Avg Latency (ms)": em.avg_latency_ms,
                "P95 Latency (ms)": em.p95_latency_ms,
                "Throughput (req/s)": em.throughput,
                "Total Tokens": em.total_tokens,
                "API Cost ($)": em.total_cost_usd,
                "Cost/Alpha ($)": em.cost_per_alpha,
                "Cache Hit Rate": em.cache_hit_rate,
            })
        return pd.DataFrame(rows).set_index("Strategy")

    def generate_latex_table(
        self,
        metric_type: str = "financial",
        caption: str = "Strategy Comparison",
        label: str = "tab:comparison",
    ) -> str:
        """
        生成 LaTeX 表格。

        Args:
            metric_type: 指标类型 ('financial' 或 'engineering')。
            caption: 表格标题。
            label: 表格标签。

        Returns:
            LaTeX 表格字符串。
        """
        df = (
            self.compare_financial_metrics()
            if metric_type == "financial"
            else self.compare_engineering_metrics()
        )

        # 格式化数值
        formatters = {
            "Total Return": "{:.2%}".format,
            "Annual Return": "{:.2%}".format,
            "Sharpe Ratio": "{:.4f}".format,
            "Sortino Ratio": "{:.4f}".format,
            "Calmar Ratio": "{:.4f}".format,
            "Max Drawdown": "{:.2%}".format,
            "Win Rate": "{:.2%}".format,
            "Profit Factor": "{:.4f}".format,
            "Volatility": "{:.2%}".format,
            "Avg Latency (ms)": "{:.2f}".format,
            "P95 Latency (ms)": "{:.2f}".format,
            "Throughput (req/s)": "{:.2f}".format,
            "API Cost ($)": "{:.4f}".format,
            "Cost/Alpha ($)": "{:.6f}".format,
            "Cache Hit Rate": "{:.2%}".format,
        }

        for col in df.columns:
            if col in formatters:
                df[col] = df[col].apply(formatters[col])

        # 生成 LaTeX
        latex = df.to_latex(
            caption=caption,
            label=label,
            escape=False,
            column_format="l" + "r" * len(df.columns),
        )
        return latex


# ============================================================================
# 便捷函数
# ============================================================================
def calculate_metrics_from_backtest(
    equity_curve: pd.DataFrame,
    trade_log: Optional[pd.DataFrame] = None,
    strategy_name: str = "Strategy",
    risk_free_rate: float = 0.02,
) -> EvaluationResult:
    """
    从 Backtrader 回测结果计算指标。

    Args:
        equity_curve: 净值曲线 DataFrame。
        trade_log: 交易日志 DataFrame。
        strategy_name: 策略名称。
        risk_free_rate: 无风险利率。

    Returns:
        EvaluationResult 对象。
    """
    calculator = MetricsCalculator(risk_free_rate=risk_free_rate)
    return calculator.evaluate(
        strategy_name=strategy_name,
        equity_curve=equity_curve,
        trade_log=trade_log,
    )


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("  指标计算器示例")
    print("=" * 60)

    # 创建示例净值曲线
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=100, freq="B")
    returns = np.random.randn(100) * 0.02 + 0.0005  # 带正偏的日收益
    nav = 1.0 * (1 + returns).cumprod()
    equity_curve = pd.DataFrame({"nav": nav}, index=dates)

    # 创建示例交易日志
    trade_log = pd.DataFrame({
        "pnl": np.random.choice([100, -50, 200, -80, 150], size=20),
    })

    # 创建示例延迟日志
    latency_log = np.random.uniform(50, 200, size=100).tolist()

    # 计算指标
    calculator = MetricsCalculator()
    result = calculator.evaluate(
        strategy_name="DualTrack_Fusion",
        equity_curve=equity_curve,
        trade_log=trade_log,
        latency_log=latency_log,
        num_signals=100,
    )

    print("\n" + result.financial_metrics.summary())
    print("\n" + result.engineering_metrics.summary())