#!/usr/bin/env python3
"""
统计显著性检验脚本。

为论文假设验证提供统计学支撑，包括：
- 夏普比率差异检验 (Jobson-Korkie / Bootstrap)
- 最大回撤差异检验 (Bootstrap)
- 胜率差异检验 (Chi-square)
- 收益率分布检验 (Kolmogorov-Smirnov)

用法:
    uv run python scripts/statistical_tests.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, ttest_ind, mannwhitneyu

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StatisticalTestResult:
    """统计检验结果数据类。"""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    significant_at_05: bool = field(init=False)
    significant_at_01: bool = field(init=False)

    def __post_init__(self):
        self.significant_at_05 = self.p_value < 0.05
        self.significant_at_01 = self.p_value < 0.01

    def to_markdown(self) -> str:
        """转换为 Markdown 格式。"""
        sig_05 = "✓" if self.significant_at_05 else "✗"
        sig_01 = "✓" if self.significant_at_01 else "✗"

        return f"""
### {self.test_name}

| 指标 | 值 |
|------|-----|
| 检验统计量 | {self.statistic:.4f} |
| p-value | {self.p_value:.4f} |
| Effect Size (Cohen's d) | {self.effect_size:.4f} |
| 95% 置信区间 | [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}] |
| 显著性 (α=0.05) | {sig_05} |
| 显著性 (α=0.01) | {sig_01} |

**解释**: {self.interpretation}
"""


def load_backtest_results(track_name: str, symbol: str = "CSI300") -> Dict[str, Any]:
    """
    加载回测结果。

    Args:
        track_name: 轨道名称 (如 'lr', 'lstm', 'lgb', 'llm-cloud')。
        symbol: 市场代码。

    Returns:
        包含 trades, positions, metrics 的字典。
    """
    results = {}

    # 尝试多个可能的位置
    possible_paths = [
        Path(f"docs/output/experiments/track_results/{track_name}"),
        Path(f"docs/output/track_results/{track_name}"),
    ]

    for base_path in possible_paths:
        if base_path.exists():
            # 加载 trades
            trades_path = base_path / "trades.csv"
            if trades_path.exists():
                results['trades'] = pd.read_csv(trades_path)

            # 加载 positions
            positions_path = base_path / "positions.csv"
            if positions_path.exists():
                results['positions'] = pd.read_csv(positions_path)

            # 加载 metadata
            metadata_path = base_path / "result" / "metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        content = f.read().strip()
                        if content:
                            results['metadata'] = json.loads(content)
                except json.JSONDecodeError:
                    pass

            break

    return results


def calculate_daily_returns(positions_df: pd.DataFrame) -> pd.Series:
    """
    从持仓数据计算日收益率。

    Args:
        positions_df: 持仓 DataFrame。

    Returns:
        日收益率 Series。
    """
    # 尝试多个可能的列名
    value_col = None
    for col in ['portfolio_value', 'total_value', 'value']:
        if col in positions_df.columns:
            value_col = col
            break

    if value_col is None:
        logger.warning("No portfolio value column found")
        return pd.Series(dtype=float)

    portfolio_value = positions_df[value_col].values
    daily_returns = np.diff(portfolio_value) / portfolio_value[:-1]

    return pd.Series(daily_returns)


def sharpe_ratio(returns: np.ndarray, rf: float = 0.0, periods: int = 252) -> float:
    """
    计算年化夏普比率。

    Args:
        returns: 日收益率序列。
        rf: 无风险利率（日）。
        periods: 年化周期数。

    Returns:
        年化夏普比率。
    """
    excess_returns = returns - rf
    if np.std(excess_returns) == 0:
        return 0.0

    return np.sqrt(periods) * np.mean(excess_returns) / np.std(excess_returns)


def jobson_korkie_test(returns_a: np.ndarray, returns_b: np.ndarray, rf: float = 0.0) -> Tuple[float, float]:
    """
    Jobson-Korkie 检验：检验两个夏普比率的差异是否显著。

    参考: Jobson & Korkie (1981), "Performance Hypothesis Testing with the Sharpe Measure"

    Args:
        returns_a: 策略 A 的日收益率。
        returns_b: 策略 B 的日收益率。
        rf: 无风险利率（日）。

    Returns:
        (z_statistic, p_value) 元组。
    """
    n = len(returns_a)

    # 计算夏普比率
    sr_a = sharpe_ratio(returns_a, rf)
    sr_b = sharpe_ratio(returns_b, rf)

    # 计算所需统计量
    mean_a = np.mean(returns_a - rf)
    mean_b = np.mean(returns_b - rf)
    std_a = np.std(returns_a, ddof=1)
    std_b = np.std(returns_b, ddof=1)

    # 协方差
    cov_ab = np.cov(returns_a, returns_b)[0, 1]

    # 相关系数
    corr = cov_ab / (std_a * std_b)

    # Jobson-Korkie 统计量
    # Var(SR_a - SR_b) ≈ (1/n) * [2 - 2*corr + 0.5*(SR_a^2 + SR_b^2 - 2*corr*SR_a*SR_b)]
    var_diff = (1/n) * (2 - 2*corr + 0.5*(sr_a**2 + sr_b**2 - 2*corr*sr_a*sr_b))

    if var_diff <= 0:
        var_diff = 1e-10

    z_stat = (sr_a - sr_b) / np.sqrt(var_diff)

    # 双尾检验 p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value


def bootstrap_sharpe_difference(returns_a: np.ndarray, returns_b: np.ndarray,
                                 n_bootstrap: int = 10000, rf: float = 0.0) -> Tuple[float, float, Tuple[float, float]]:
    """
    Bootstrap 方法估计夏普比率差异的置信区间。

    Args:
        returns_a: 策略 A 的日收益率。
        returns_b: 策略 B 的日收益率。
        n_bootstrap: Bootstrap 次数。
        rf: 无风险利率。

    Returns:
        (mean_diff, p_value, confidence_interval) 元组。
    """
    n = len(returns_a)
    sr_diffs = []

    for _ in range(n_bootstrap):
        # 有放回采样
        indices = np.random.choice(n, size=n, replace=True)
        sample_a = returns_a[indices]
        sample_b = returns_b[indices]

        sr_a = sharpe_ratio(sample_a, rf)
        sr_b = sharpe_ratio(sample_b, rf)
        sr_diffs.append(sr_a - sr_b)

    sr_diffs = np.array(sr_diffs)
    mean_diff = np.mean(sr_diffs)

    # 计算 p-value (双边检验)
    p_value = 2 * min(np.mean(sr_diffs > 0), np.mean(sr_diffs < 0))

    # 95% 置信区间
    ci_lower = np.percentile(sr_diffs, 2.5)
    ci_upper = np.percentile(sr_diffs, 97.5)

    return mean_diff, p_value, (ci_lower, ci_upper)


def bootstrap_max_drawdown_difference(returns_a: np.ndarray, returns_b: np.ndarray,
                                       n_bootstrap: int = 10000) -> Tuple[float, float, Tuple[float, float]]:
    """
    Bootstrap 方法估计最大回撤差异的置信区间。

    Args:
        returns_a: 策略 A 的日收益率。
        returns_b: 策略 B 的日收益率。
        n_bootstrap: Bootstrap 次数。

    Returns:
        (mean_diff, p_value, confidence_interval) 元组。
    """

    def calculate_max_drawdown(returns: np.ndarray) -> float:
        """计算最大回撤。"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown)

    n = len(returns_a)
    dd_diffs = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        sample_a = returns_a[indices]
        sample_b = returns_b[indices]

        dd_a = calculate_max_drawdown(sample_a)
        dd_b = calculate_max_drawdown(sample_b)
        dd_diffs.append(dd_b - dd_a)  # 正值表示 B 回撤更大

    dd_diffs = np.array(dd_diffs)
    mean_diff = np.mean(dd_diffs)

    p_value = 2 * min(np.mean(dd_diffs > 0), np.mean(dd_diffs < 0))

    ci_lower = np.percentile(dd_diffs, 2.5)
    ci_upper = np.percentile(dd_diffs, 97.5)

    return mean_diff, p_value, (ci_lower, ci_upper)


def chi_square_win_rate_test(trades_a: pd.DataFrame, trades_b: pd.DataFrame) -> Tuple[float, float]:
    """
    卡方检验：检验两个策略胜率差异是否显著。

    Args:
        trades_a: 策略 A 的交易记录。
        trades_b: 策略 B 的交易记录。

    Returns:
        (chi2_stat, p_value) 元组。
    """
    # 统计盈亏交易
    def count_wins_losses(trades_df: pd.DataFrame) -> Tuple[int, int]:
        if 'pnl' not in trades_df.columns:
            return 0, 0
        wins = (trades_df['pnl'] > 0).sum()
        losses = (trades_df['pnl'] <= 0).sum()
        return wins, losses

    wins_a, losses_a = count_wins_losses(trades_a)
    wins_b, losses_b = count_wins_losses(trades_b)

    if wins_a + losses_a == 0 or wins_b + losses_b == 0:
        logger.warning("Insufficient trade data for chi-square test")
        return 0.0, 1.0

    # 构建 2x2 列联表
    contingency_table = np.array([
        [wins_a, losses_a],
        [wins_b, losses_b]
    ])

    chi2, p_value, dof, expected = chi2_contingency(contingency_table)

    return chi2, p_value


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    计算 Cohen's d 效应量。

    Args:
        group_a: 组 A 数据。
        group_b: 组 B 数据。

    Returns:
        Cohen's d 值。
    """
    n_a, n_b = len(group_a), len(group_b)
    var_a = np.var(group_a, ddof=1)
    var_b = np.var(group_b, ddof=1)

    # 池化标准差
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group_a) - np.mean(group_b)) / pooled_std


def run_all_tests() -> Dict[str, StatisticalTestResult]:
    """
    运行所有统计检验。

    Returns:
        {test_name: StatisticalTestResult} 字典。
    """
    results = {}

    # 定义要比较的轨道
    ml_tracks = ['lr', 'lstm', 'lgb']
    llm_tracks = ['llm-cloud', 'llm-local', 'llm-siliconflow']

    # 加载数据
    logger.info("Loading backtest results...")

    all_returns = {}
    all_trades = {}

    for track in ml_tracks + llm_tracks:
        data = load_backtest_results(track)
        if 'positions' in data and len(data['positions']) > 0:
            returns = calculate_daily_returns(data['positions'])
            if len(returns) > 0:
                all_returns[track] = returns.values
                logger.info(f"  {track}: {len(returns)} daily returns")
        if 'trades' in data and len(data['trades']) > 0:
            all_trades[track] = data['trades']

    if not all_returns:
        logger.error("No return data loaded!")
        return results

    # 找到最佳的 ML 和 LLM 轨道
    best_ml = None
    best_ml_sr = -np.inf
    best_llm = None
    best_llm_sr = -np.inf

    for track, returns in all_returns.items():
        sr = sharpe_ratio(returns)
        if track in ml_tracks and sr > best_ml_sr:
            best_ml = track
            best_ml_sr = sr
        elif track in llm_tracks and sr > best_llm_sr:
            best_llm = track
            best_llm_sr = sr

    logger.info(f"\nBest ML track: {best_ml} (SR={best_ml_sr:.2f})")
    logger.info(f"Best LLM track: {best_llm} (SR={best_llm_sr:.2f})")

    if best_ml and best_llm and best_ml in all_returns and best_llm in all_returns:
        returns_ml = all_returns[best_ml]
        returns_llm = all_returns[best_llm]

        # 1. 夏普比率差异检验 (Jobson-Korkie)
        logger.info("\n1. Testing Sharpe Ratio difference (Jobson-Korkie)...")
        z_stat, p_value = jobson_korkie_test(returns_llm, returns_ml)
        effect_size = cohens_d(returns_llm, returns_ml)

        results['sharpe_ratio_jk'] = StatisticalTestResult(
            test_name=f"Sharpe Ratio Difference ({best_llm} vs {best_ml}) - Jobson-Korkie Test",
            statistic=z_stat,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=(-np.inf, np.inf),  # JK test 不提供 CI
            interpretation=f"LLM track ({best_llm}) 与 ML track ({best_ml}) 的夏普比率差异"
                          f"{'显著' if p_value < 0.05 else '不显著'} (p={p_value:.4f})"
        )

        # 2. 夏普比率差异检验 (Bootstrap)
        logger.info("2. Testing Sharpe Ratio difference (Bootstrap)...")
        mean_diff, p_value, ci = bootstrap_sharpe_difference(returns_llm, returns_ml)

        results['sharpe_ratio_bootstrap'] = StatisticalTestResult(
            test_name=f"Sharpe Ratio Difference ({best_llm} vs {best_ml}) - Bootstrap",
            statistic=mean_diff,
            p_value=p_value,
            effect_size=effect_size,
            confidence_interval=ci,
            interpretation=f"夏普比率差异的 Bootstrap 估计: {mean_diff:.4f}, "
                          f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]"
        )

        # 3. 最大回撤差异检验 (Bootstrap)
        logger.info("3. Testing Max Drawdown difference...")
        mean_diff, p_value, ci = bootstrap_max_drawdown_difference(returns_llm, returns_ml)

        results['max_drawdown'] = StatisticalTestResult(
            test_name=f"Max Drawdown Difference ({best_llm} vs {best_ml})",
            statistic=mean_diff,
            p_value=p_value,
            effect_size=cohens_d(returns_llm, returns_ml),
            confidence_interval=ci,
            interpretation=f"最大回撤差异: LLM 比 ML {'大' if mean_diff > 0 else '小'} "
                          f"{abs(mean_diff):.2%}, p={p_value:.4f}"
        )

        # 4. 收益率分布差异检验 (K-S Test)
        logger.info("4. Testing Return distribution difference (K-S Test)...")
        ks_stat, p_value = ks_2samp(returns_llm, returns_ml)

        results['ks_test'] = StatisticalTestResult(
            test_name=f"Return Distribution Difference (K-S Test)",
            statistic=ks_stat,
            p_value=p_value,
            effect_size=cohens_d(returns_llm, returns_ml),
            confidence_interval=(0, 1),
            interpretation=f"K-S 检验表明收益率分布差异"
                          f"{'显著' if p_value < 0.05 else '不显著'}"
        )

        # 5. 收益率均值差异检验 (t-test)
        logger.info("5. Testing Mean return difference (t-test)...")
        t_stat, p_value = ttest_ind(returns_llm, returns_ml)

        results['t_test'] = StatisticalTestResult(
            test_name=f"Mean Return Difference (t-test)",
            statistic=t_stat,
            p_value=p_value,
            effect_size=cohens_d(returns_llm, returns_ml),
            confidence_interval=(-np.inf, np.inf),
            interpretation=f"t 检验表明均值差异"
                          f"{'显著' if p_value < 0.05 else '不显著'}"
        )

    # 6. 胜率差异检验 (Chi-square)
    if best_ml and best_llm and best_ml in all_trades and best_llm in all_trades:
        logger.info("6. Testing Win rate difference (Chi-square)...")
        chi2, p_value = chi_square_win_rate_test(all_trades[best_ml], all_trades[best_llm])

        results['win_rate_chi2'] = StatisticalTestResult(
            test_name=f"Win Rate Difference ({best_llm} vs {best_ml}) - Chi-square Test",
            statistic=chi2,
            p_value=p_value,
            effect_size=0.0,  # Chi-square 不计算 Cohen's d
            confidence_interval=(0, 1),
            interpretation=f"卡方检验表明胜率差异"
                          f"{'显著' if p_value < 0.05 else '不显著'} (χ²={chi2:.4f})"
        )

    # 7. 所有 ML vs 所有 LLM 整体对比
    ml_returns_all = []
    llm_returns_all = []

    for track in ml_tracks:
        if track in all_returns:
            ml_returns_all.extend(all_returns[track])

    for track in llm_tracks:
        if track in all_returns:
            llm_returns_all.extend(all_returns[track])

    if ml_returns_all and llm_returns_all:
        ml_returns_all = np.array(ml_returns_all)
        llm_returns_all = np.array(llm_returns_all)

        logger.info("7. Testing ML vs LLM overall difference...")
        mw_stat, p_value = mannwhitneyu(llm_returns_all, ml_returns_all, alternative='two-sided')

        results['mann_whitney'] = StatisticalTestResult(
            test_name="ML vs LLM Overall (Mann-Whitney U Test)",
            statistic=mw_stat,
            p_value=p_value,
            effect_size=cohens_d(llm_returns_all, ml_returns_all),
            confidence_interval=(0, 1),
            interpretation=f"Mann-Whitney U 检验表明 ML 和 LLM 整体收益分布差异"
                          f"{'显著' if p_value < 0.05 else '不显著'}"
        )

    return results


def generate_report(results: Dict[str, StatisticalTestResult], output_path: str) -> None:
    """
    生成 Markdown 格式的统计检验报告。

    Args:
        results: 统计检验结果字典。
        output_path: 输出文件路径。
    """
    report = """# Statistical Significance Tests Report

**Generated**: 2026-03-18

This report contains statistical tests for hypothesis validation in the thesis.

---

## Overview

These tests validate the following research hypotheses:

- **H1**: LLM Tracks outperform ML Tracks in risk-adjusted returns (Sharpe Ratio)
- **H2**: LLM Tracks provide better risk control during high volatility
- **H3**: Trade-off between speed and intelligence

---

## Test Results

"""

    for test_name, result in results.items():
        report += result.to_markdown()
        report += "\n---\n"

    # 添加汇总表格
    report += """
## Summary Table

| Test | Statistic | p-value | Significant (α=0.05) | Effect Size |
|------|-----------|---------|---------------------|-------------|
"""

    for test_name, result in results.items():
        sig = "Yes" if result.significant_at_05 else "No"
        report += f"| {result.test_name[:40]}... | {result.statistic:.4f} | {result.p_value:.4f} | {sig} | {result.effect_size:.4f} |\n"

    report += """
---

## Interpretation Guidelines

### Effect Size (Cohen's d)

| Range | Interpretation |
|-------|---------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Small |
| 0.5 - 0.8 | Medium |
| > 0.8 | Large |

### p-value Thresholds

- **p < 0.01**: Strong evidence against null hypothesis
- **p < 0.05**: Moderate evidence against null hypothesis
- **p < 0.10**: Weak evidence against null hypothesis
- **p >= 0.10**: Insufficient evidence to reject null hypothesis

---

## Methodology

1. **Jobson-Korkie Test**: Tests difference between two Sharpe ratios
2. **Bootstrap**: Resampling method for confidence intervals
3. **Chi-square Test**: Tests difference in win rates
4. **Kolmogorov-Smirnov Test**: Tests difference in return distributions
5. **Mann-Whitney U Test**: Non-parametric test for distribution differences

---

*Report generated by statistical_tests.py*
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Statistical tests report saved to: {output_path}")


def main():
    """主函数。"""
    logger.info("="*60)
    logger.info("  Statistical Significance Tests")
    logger.info("="*60)

    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 运行所有检验
    results = run_all_tests()

    if results:
        # 生成报告
        output_path = "docs/reference/statistical_tests.md"
        generate_report(results, output_path)

        # 打印摘要
        logger.info("\n" + "="*60)
        logger.info("  Summary")
        logger.info("="*60)

        for test_name, result in results.items():
            sig = "SIGNIFICANT" if result.significant_at_05 else "not significant"
            logger.info(f"\n{result.test_name}:")
            logger.info(f"  p-value: {result.p_value:.4f} ({sig})")
            logger.info(f"  Effect size: {result.effect_size:.4f}")
    else:
        logger.error("No test results generated!")


if __name__ == "__main__":
    main()