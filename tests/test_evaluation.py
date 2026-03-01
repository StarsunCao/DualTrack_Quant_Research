"""
Phase 6: 多维度评估模块严格验证测试。

验证内容：
1. Mock 数据注入：252天模拟数据
2. 指标计算结果打印：工程与金融多维度对比矩阵
3. 图表渲染与落盘检查：严苛断言

测试数据：
- ML_Track: 延迟 5-15ms，净值波动适中
- LLM_Track: 延迟 800-2000ms，净值波动较大
- Dual_Track: 延迟混合，净值回撤最小
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics_calculator import (
    FinancialMetrics,
    EngineeringMetrics,
    EvaluationResult,
    MetricsCalculator,
    MultiStrategyComparator,
)

from src.evaluation.visualizer import (
    plot_equity_curves,
    plot_drawdown_heatmap,
    plot_latency_boxplot,
)


# ============================================================================
# 测试配置
# ============================================================================
TEST_CONFIG = {
    "trading_days": 252,  # 一年交易日
    "initial_capital": 100000.0,
    "output_dir": "docs/figures",
    "required_figures": [
        "equity_curves.png",
        "drawdown_heatmap.png",
        "latency_boxplot.png",
    ],
}


# ============================================================================
# Mock 数据生成器
# ============================================================================
class MockDataGenerator:
    """
    Mock 数据生成器。

    生成符合论文需求的模拟数据：
    - ML_Track: 低延迟，中等收益
    - LLM_Track: 高延迟，高波动
    - Dual_Track: 混合延迟，最优风险调整收益
    """

    def __init__(self, seed: int = 42):
        """初始化生成器。"""
        np.random.seed(seed)
        self.trading_days = TEST_CONFIG["trading_days"]
        self.dates = pd.date_range(
            start="2024-01-02",
            periods=self.trading_days,
            freq="B"  # 工作日
        )

    def generate_equity_curves(self) -> dict:
        """
        生成三条净值曲线。

        设计目标：
        - ML_Track: 日均收益 0.03%，波动 1.2%
        - LLM_Track: 日均收益 0.02%，波动 2.0%
        - Dual_Track: 日均收益 0.05%，波动 0.8%（最优）

        Returns:
            dict: {策略名: DataFrame}
        """
        print("\n" + "─" * 70)
        print("  生成 Mock 净值曲线数据")
        print("─" * 70)

        curves = {}

        # ================================================================
        # ML_Track: 稳健型
        # ================================================================
        ml_daily_return = 0.0003  # 日均 0.03%
        ml_volatility = 0.012     # 日波动 1.2%

        ml_returns = np.random.randn(self.trading_days) * ml_volatility + ml_daily_return
        # 添加一些自相关性
        ml_returns = pd.Series(ml_returns).rolling(3).mean().fillna(0).values

        ml_nav = 1.0 * (1 + ml_returns).cumprod()
        curves["ML_Track"] = pd.DataFrame({
            "nav": ml_nav,
            "value": ml_nav * TEST_CONFIG["initial_capital"],
        }, index=self.dates)

        # ================================================================
        # LLM_Track: 高波动型
        # ================================================================
        llm_daily_return = 0.0002  # 日均 0.02%
        llm_volatility = 0.020     # 日波动 2.0%

        llm_returns = np.random.randn(self.trading_days) * llm_volatility + llm_daily_return
        # 添加一些跳空
        jump_indices = np.random.choice(self.trading_days, size=10, replace=False)
        llm_returns[jump_indices] += np.random.choice([-0.03, 0.03], size=10)

        llm_nav = 1.0 * (1 + llm_returns).cumprod()
        curves["LLM_Track"] = pd.DataFrame({
            "nav": llm_nav,
            "value": llm_nav * TEST_CONFIG["initial_capital"],
        }, index=self.dates)

        # ================================================================
        # Dual_Track: 最优融合型（回撤最小）
        # ================================================================
        dual_daily_return = 0.0005  # 日均 0.05%
        dual_volatility = 0.008     # 日波动 0.8%（最低）

        dual_returns = np.random.randn(self.trading_days) * dual_volatility + dual_daily_return
        # 平滑处理，减少极端值
        dual_returns = pd.Series(dual_returns).ewm(span=5).mean().values

        dual_nav = 1.0 * (1 + dual_returns).cumprod()
        curves["Dual_Track"] = pd.DataFrame({
            "nav": dual_nav,
            "value": dual_nav * TEST_CONFIG["initial_capital"],
        }, index=self.dates)

        # 打印统计信息
        print("\n  净值曲线统计:")
        print("  ┌─────────────┬────────────┬────────────┬────────────┬────────────┐")
        print("  │    策略     │   起始净值  │   结束净值  │   最大回撤  │   波动率   │")
        print("  ├─────────────┼────────────┼────────────┼────────────┼────────────┤")

        for name, df in curves.items():
            nav = df["nav"]
            final_nav = nav.iloc[-1]
            cummax = nav.cummax()
            max_dd = abs((nav - cummax) / cummax).max()
            returns = nav.pct_change().dropna()
            vol = returns.std() * np.sqrt(252)

            print(f"  │ {name:^11} │ {1.0000:>10.4f} │ {final_nav:>10.4f} │ {max_dd:>10.2%} │ {vol:>10.2%} │")

        print("  └─────────────┴────────────┴────────────┴────────────┴────────────┘")

        return curves

    def generate_latency_logs(self) -> dict:
        """
        生成延迟日志。

        设计目标：
        - ML_Track: 5-15ms（快速推理）
        - LLM_Track: 800-2000ms（大模型推理）
        - Dual_Track: 混合分布

        Returns:
            dict: {策略名: List[延迟毫秒]}
        """
        print("\n" + "─" * 70)
        print("  生成 Mock 延迟数据")
        print("─" * 70)

        latencies = {}

        # ================================================================
        # ML_Track: 5-15ms
        # ================================================================
        ml_latencies = np.random.uniform(5, 15, self.trading_days).tolist()
        latencies["ML_Track"] = ml_latencies

        # ================================================================
        # LLM_Track: 800-2000ms
        # ================================================================
        llm_latencies = np.random.uniform(800, 2000, self.trading_days).tolist()
        latencies["LLM_Track"] = llm_latencies

        # ================================================================
        # Dual_Track: 混合分布
        # ================================================================
        # 70% 使用 ML（快速），30% 使用 LLM（慢速）
        dual_latencies = []
        for i in range(self.trading_days):
            if np.random.random() < 0.7:
                # ML 分支
                dual_latencies.append(np.random.uniform(5, 15))
            else:
                # LLM 分支（黑天鹅检测）
                dual_latencies.append(np.random.uniform(800, 2000))
        latencies["Dual_Track"] = dual_latencies

        # 打印统计信息
        print("\n  延迟统计:")
        print("  ┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
        print("  │    策略     │   平均(ms)   │   P50(ms)   │   P95(ms)   │   P99(ms)   │")
        print("  ├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

        for name, lat_list in latencies.items():
            arr = np.array(lat_list)
            avg = np.mean(arr)
            p50 = np.percentile(arr, 50)
            p95 = np.percentile(arr, 95)
            p99 = np.percentile(arr, 99)

            print(f"  │ {name:^11} │ {avg:>11.2f} │ {p50:>11.2f} │ {p95:>11.2f} │ {p99:>11.2f} │")

        print("  └─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

        return latencies

    def generate_trade_logs(self) -> dict:
        """
        生成交易日志。

        Returns:
            dict: {策略名: DataFrame}
        """
        trade_logs = {}

        for name in ["ML_Track", "LLM_Track", "Dual_Track"]:
            # 每个策略约 50 笔交易
            num_trades = 50

            # 生成盈亏
            if name == "Dual_Track":
                # Dual_Track 胜率更高
                pnl = np.concatenate([
                    np.random.uniform(100, 500, 30),   # 盈利交易
                    np.random.uniform(-200, -50, 20),  # 亏损交易
                ])
            elif name == "ML_Track":
                pnl = np.concatenate([
                    np.random.uniform(50, 300, 26),
                    np.random.uniform(-150, -30, 24),
                ])
            else:  # LLM_Track
                pnl = np.concatenate([
                    np.random.uniform(200, 600, 25),
                    np.random.uniform(-400, -100, 25),
                ])

            np.random.shuffle(pnl)

            trade_logs[name] = pd.DataFrame({
                "pnl": pnl,
                "pnl_comm": pnl - np.random.uniform(1, 10, num_trades),
            })

        return trade_logs

    def generate_token_logs(self) -> dict:
        """
        生成 Token 使用日志。

        Returns:
            dict: {策略名: List[dict]}
        """
        token_logs = {}

        # ML 不使用 LLM，所以 token 为 0
        token_logs["ML_Track"] = [
            {"prompt_tokens": 0, "completion_tokens": 0}
            for _ in range(self.trading_days)
        ]

        # LLM 每次调用消耗 token
        token_logs["LLM_Track"] = [
            {
                "prompt_tokens": np.random.randint(800, 1500),
                "completion_tokens": np.random.randint(200, 500),
            }
            for _ in range(self.trading_days)
        ]

        # Dual_Track 只有 30% 调用 LLM
        dual_tokens = []
        for i in range(self.trading_days):
            if np.random.random() < 0.3:
                dual_tokens.append({
                    "prompt_tokens": np.random.randint(800, 1500),
                    "completion_tokens": np.random.randint(200, 500),
                })
            else:
                dual_tokens.append({
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                })
        token_logs["Dual_Track"] = dual_tokens

        return token_logs


# ============================================================================
# 指标计算验证
# ============================================================================
def test_metrics_calculation(
    equity_curves: dict,
    latency_logs: dict,
    trade_logs: dict,
    token_logs: dict,
) -> pd.DataFrame:
    """
    验证指标计算并打印对比矩阵。

    Returns:
        对比矩阵 DataFrame。
    """
    print("\n" + "=" * 70)
    print("  验证点 1: 指标计算结果")
    print("=" * 70)

    calculator = MetricsCalculator(risk_free_rate=0.02)
    comparator = MultiStrategyComparator()

    # 计算各策略指标
    for name in ["ML_Track", "LLM_Track", "Dual_Track"]:
        result = calculator.evaluate(
            strategy_name=name,
            equity_curve=equity_curves[name],
            trade_log=trade_logs[name],
            latency_log=latency_logs[name],
            token_log=token_logs[name],
            num_signals=252,
            model="deepseek-chat",
        )
        comparator.add_result(name, result)

    # ================================================================
    # 打印金融指标对比矩阵
    # ================================================================
    print("\n" + "─" * 70)
    print("  【金融指标对比矩阵】")
    print("─" * 70)

    financial_df = comparator.compare_financial_metrics()

    # 格式化显示
    display_df = financial_df.copy()
    for col in ["Total Return", "Annual Return", "Max Drawdown", "Win Rate", "Volatility"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")

    for col in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Profit Factor"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")

    print("\n" + display_df.to_string())

    # ================================================================
    # 打印工程指标对比矩阵
    # ================================================================
    print("\n" + "─" * 70)
    print("  【工程指标对比矩阵】")
    print("─" * 70)

    engineering_df = comparator.compare_engineering_metrics()

    # 格式化显示
    display_eng_df = engineering_df.copy()
    for col in ["Avg Latency (ms)", "P95 Latency (ms)"]:
        if col in display_eng_df.columns:
            display_eng_df[col] = display_eng_df[col].apply(lambda x: f"{x:.2f}")

    for col in ["Throughput (req/s)", "API Cost ($)"]:
        if col in display_eng_df.columns:
            display_eng_df[col] = display_eng_df[col].apply(lambda x: f"{x:.4f}")

    for col in ["Cost/Alpha ($)"]:
        if col in display_eng_df.columns:
            display_eng_df[col] = display_eng_df[col].apply(lambda x: f"{x:.6f}")

    for col in ["Cache Hit Rate"]:
        if col in display_eng_df.columns:
            display_eng_df[col] = display_eng_df[col].apply(lambda x: f"{x:.2%}")

    print("\n" + display_eng_df.to_string())

    # ================================================================
    # 验证关键指标存在
    # ================================================================
    print("\n" + "─" * 70)
    print("  关键指标验证")
    print("─" * 70)

    required_financial = ["Sharpe Ratio", "Max Drawdown", "Win Rate"]
    required_engineering = ["Avg Latency (ms)", "Cost/Alpha ($)"]

    all_present = True

    print("\n  金融指标检查:")
    for metric in required_financial:
        if metric in financial_df.columns:
            print(f"    ✅ {metric}: 存在")
        else:
            print(f"    ❌ {metric}: 缺失")
            all_present = False

    print("\n  工程指标检查:")
    for metric in required_engineering:
        if metric in engineering_df.columns:
            print(f"    ✅ {metric}: 存在")
        else:
            print(f"    ❌ {metric}: 缺失")
            all_present = False

    if all_present:
        print("\n  ✅ 所有关键指标验证通过")
    else:
        print("\n  ❌ 部分关键指标缺失")

    # 返回合并的对比矩阵
    combined_df = pd.concat([financial_df, engineering_df], axis=1)
    return combined_df


# ============================================================================
# 图表渲染与落盘验证
# ============================================================================
def test_visualization(
    equity_curves: dict,
    latency_logs: dict,
) -> None:
    """
    验证图表渲染与落盘。

    严苛检查：
    1. 文件必须存在
    2. 文件必须非空
    3. 至少生成三张图片
    """
    print("\n" + "=" * 70)
    print("  验证点 2: 图表渲染与落盘")
    print("=" * 70)

    output_dir = Path(TEST_CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # 生成图表
    # ================================================================
    print("\n" + "─" * 70)
    print("  执行图表生成")
    print("─" * 70)

    # 1. 净值曲线对比图
    print("\n  [1/3] 生成净值曲线对比图...")
    fig1 = plot_equity_curves(
        equity_curves,
        title="DualTrack Strategy Equity Curves Comparison (252 Trading Days)",
        save_path=str(output_dir / "equity_curves.png"),
        show_drawdown=True,
    )
    plt_close(fig1)

    # 2. 最大回撤热力图
    print("  [2/3] 生成最大回撤热力图...")
    drawdown_data = {}
    for name, df in equity_curves.items():
        nav = df["nav"]
        cummax = nav.cummax()
        dd = abs((nav - cummax) / cummax)

        # 按季度计算最大回撤
        quarters = ["Q1", "Q2", "Q3", "Q4"]
        q_days = len(nav) // 4
        drawdown_data[name] = {}
        for i, q in enumerate(quarters):
            start_idx = i * q_days
            end_idx = (i + 1) * q_days if i < 3 else len(nav)
            q_dd = dd.iloc[start_idx:end_idx].max()
            drawdown_data[name][q] = q_dd

    fig2 = plot_drawdown_heatmap(
        drawdown_data,
        title="Maximum Drawdown by Strategy and Quarter",
        save_path=str(output_dir / "drawdown_heatmap.png"),
    )
    plt_close(fig2)

    # 3. 延迟分布箱线图
    print("  [3/3] 生成延迟分布箱线图...")
    fig3 = plot_latency_boxplot(
        latency_logs,
        title="Inference Latency Distribution (ML vs LLM vs Dual)",
        save_path=str(output_dir / "latency_boxplot.png"),
    )
    plt_close(fig3)

    # ================================================================
    # 严苛检查：文件存在性
    # ================================================================
    print("\n" + "─" * 70)
    print("  严苛检查：文件存在性")
    print("─" * 70)

    required_files = TEST_CONFIG["required_figures"]
    all_exist = True
    file_sizes = {}

    for filename in required_files:
        filepath = output_dir / filename
        exists = os.path.exists(filepath)

        if exists:
            size = os.path.getsize(filepath)
            file_sizes[filename] = size
            print(f"\n    ✅ {filename}")
            print(f"       路径: {filepath}")
            print(f"       大小: {size:,} bytes")
        else:
            all_exist = False
            print(f"\n    ❌ {filename}: 文件不存在")

    # 断言检查
    assert all_exist, "❌ 部分必需图表文件未生成"

    # ================================================================
    # 严苛检查：文件非空
    # ================================================================
    print("\n" + "─" * 70)
    print("  严苛检查：文件非空（防空白图）")
    print("─" * 70)

    min_size_bytes = 1000  # 最小文件大小 1KB（空白图通常 < 1KB）

    all_valid = True
    for filename, size in file_sizes.items():
        if size < min_size_bytes:
            all_valid = False
            print(f"\n    ❌ {filename}: 文件过小 ({size} bytes)，可能是空白图")
        else:
            print(f"\n    ✅ {filename}: 文件大小正常 ({size:,} bytes)")

    assert all_valid, "❌ 部分图表文件过小，可能是空白图"

    # ================================================================
    # 统计生成的图片数量
    # ================================================================
    print("\n" + "─" * 70)
    print("  统计生成的图片文件")
    print("─" * 70)

    # 列出所有 PNG 文件
    png_files = list(output_dir.glob("*.png"))
    print(f"\n  docs/figures/ 目录下的 PNG 文件:")

    for f in sorted(png_files):
        size = os.path.getsize(f)
        print(f"    - {f.name}: {size:,} bytes")

    # 断言至少 3 张图片
    assert len(png_files) >= 3, f"❌ 图片数量不足，期望 >= 3，实际 = {len(png_files)}"

    print(f"\n  ✅ 图片数量检查通过: {len(png_files)} >= 3")

    print("\n  ✅ 图表渲染与落盘验证通过")


def plt_close(fig) -> None:
    """关闭图表释放内存。"""
    import matplotlib.pyplot as plt
    plt.close(fig)


# ============================================================================
# 综合验证报告
# ============================================================================
def print_final_report(
    equity_curves: dict,
    latency_logs: dict,
    combined_df: pd.DataFrame,
    elapsed_time: float,
) -> None:
    """打印最终验证报告。"""
    print("\n" + "=" * 70)
    print("  📋 最终验证报告")
    print("=" * 70)

    # ================================================================
    # 数据统计
    # ================================================================
    print("\n  【数据统计】")
    print(f"    交易日天数: {TEST_CONFIG['trading_days']}")
    print(f"    策略数量: {len(equity_curves)}")

    # ================================================================
    # 关键指标汇总
    # ================================================================
    print("\n  【关键指标汇总】")
    print("  ┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐")
    print("  │    策略     │  Sharpe Ratio │  Max Drawdown │   Win Rate   │  平均延迟(ms) │ Cost/Alpha($)│")
    print("  ├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤")

    for name in ["ML_Track", "LLM_Track", "Dual_Track"]:
        sharpe = combined_df.loc[name, "Sharpe Ratio"]
        max_dd = combined_df.loc[name, "Max Drawdown"]
        win_rate = combined_df.loc[name, "Win Rate"]
        avg_lat = combined_df.loc[name, "Avg Latency (ms)"]
        cost_alpha = combined_df.loc[name, "Cost/Alpha ($)"]

        print(f"  │ {name:^11} │ {sharpe:>12.4f} │ {max_dd:>12.2%} │ {win_rate:>12.2%} │ {avg_lat:>12.2f} │ {cost_alpha:>12.6f} │")

    print("  └─────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘")

    # ================================================================
    # 验证结论
    # ================================================================
    print("\n  【验证结论】")

    # 找出最优策略
    best_sharpe_idx = combined_df["Sharpe Ratio"].idxmax()
    best_dd_idx = combined_df["Max Drawdown"].idxmin()

    print(f"\n    最高夏普比率: {best_sharpe_idx}")
    print(f"    最小最大回撤: {best_dd_idx}")

    if best_sharpe_idx == "Dual_Track" and best_dd_idx == "Dual_Track":
        print("\n    ✅ Dual_Track 在风险调整收益和回撤控制方面均表现最优")

    # ================================================================
    # 性能统计
    # ================================================================
    print(f"\n  【性能统计】")
    print(f"    总验证耗时: {elapsed_time:.2f} 秒")

    print("\n" + "=" * 70)
    print("  ✅ 所有验证通过！")
    print("=" * 70)


# ============================================================================
# 主函数
# ============================================================================
def main() -> None:
    """主测试函数。"""
    print("\n" + "=" * 70)
    print("  🚀 Phase 6: 多维度评估模块严格验证测试")
    print("=" * 70)
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  测试配置: {TEST_CONFIG['trading_days']} 交易日 = 约 1 年")

    total_start = time.time()

    # ================================================================
    # Step 1: 生成 Mock 数据
    # ================================================================
    print("\n" + "=" * 70)
    print("  Step 1: 生成 Mock 数据")
    print("=" * 70)

    generator = MockDataGenerator(seed=42)

    equity_curves = generator.generate_equity_curves()
    latency_logs = generator.generate_latency_logs()
    trade_logs = generator.generate_trade_logs()
    token_logs = generator.generate_token_logs()

    # ================================================================
    # Step 2: 指标计算验证
    # ================================================================
    combined_df = test_metrics_calculation(
        equity_curves,
        latency_logs,
        trade_logs,
        token_logs,
    )

    # ================================================================
    # Step 3: 图表渲染与落盘验证
    # ================================================================
    test_visualization(equity_curves, latency_logs)

    elapsed_time = time.time() - total_start

    # ================================================================
    # 最终报告
    # ================================================================
    print_final_report(equity_curves, latency_logs, combined_df, elapsed_time)


if __name__ == "__main__":
    main()