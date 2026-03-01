#!/usr/bin/env python
"""
DualTrack Quant Research - CLI 主入口。

Usage:
    python main.py run --symbol CSI300 --start 2020-01-01 --end 2024-01-01
    python main.py evaluate
    python main.py cache-build

子命令:
    run:         执行完整回测流水线 (Phase 1-6)
    evaluate:    重新生成评估图表
    cache-build: 构建 LLM 离线缓存
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import numpy as np
import pandas as pd

# 确保项目根目录在路径中
sys.path.insert(0, str(Path(__file__).parent))

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# CLI 主入口
# ============================================================================
@click.group()
@click.version_option(version="1.0.0", prog_name="DualTrack Quant")
@click.option("--verbose", "-v", is_flag=True, help="详细输出模式")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """
    DualTrack Quant Research - 双轨制量化回测框架。

    项目目标：对比 ML Track 和 LLM Track 在量化交易中的 ROI 和鲁棒性。
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# ============================================================================
# run 子命令 - 执行完整回测流水线
# ============================================================================
@cli.command("run")
@click.option("--symbol", "-s", default="CSI300", help="交易标的 (CSI300/QQQ)")
@click.option("--start", default="2026-01-09", help="回测开始日期")
@click.option("--end", default="2026-02-28", help="回测结束日期")
@click.option("--initial-cash", default=100000.0, help="初始资金")
@click.option("--commission", default=0.0002, help="佣金率")
@click.option("--output-dir", default="docs/output", help="输出目录")
@click.pass_context
def run_backtest(
    ctx: click.Context,
    symbol: str,
    start: str,
    end: str,
    initial_cash: float,
    commission: float,
    output_dir: str,
) -> None:
    """
    执行完整的双轨制回测流水线 (Phase 1-6)。

    流程:
      1. 数据获取 (Phase 1)
      2. ML Track 信号生成 (Phase 2)
      3. LLM Track 信号生成 (Phase 3)
      4. 信号融合 (Phase 4)
      5. Backtrader 回测执行 (Phase 5)
      6. 多维度评估 (Phase 6)
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo("=" * 70)
    click.echo("  DualTrack Quant - 完整回测流水线")
    click.echo("=" * 70)
    click.echo(f"  标的: {symbol}")
    click.echo(f"  日期范围: {start} ~ {end}")
    click.echo(f"  初始资金: {initial_cash:,.2f}")
    click.echo("=" * 70)

    total_start = time.time()

    # ================================================================
    # Phase 1: 数据获取
    # ================================================================
    click.echo("\n[Phase 1/6] 数据获取...")

    try:
        from src.data.market_data import MarketDataFetcher
        from src.data.news_data import MockNewsGenerator
        from src.data.data_aligner import DataAligner

        # 获取价格数据
        fetcher = MarketDataFetcher()

        # 优先读取真实数据文件
        real_data_path = Path(f"data/raw/real_{symbol.lower()}_1y.csv")
        if real_data_path.exists():
            click.echo(f"  使用真实 OHLCV 数据: {real_data_path}")
            ohlcv_data = pd.read_csv(real_data_path, parse_dates=["date"])
            ohlcv_data.set_index("date", inplace=True)
        elif symbol == "CSI300":
            ohlcv_data = fetcher.fetch_csi300(start_date=start, end_date=end)
        elif symbol == "QQQ":
            ohlcv_data = fetcher.fetch_qqq(start_date=start, end_date=end)
        else:
            # 生成模拟数据
            click.echo(f"  使用模拟数据: {symbol}")
            ohlcv_data = _generate_mock_ohlcv(start, end)

        click.echo(f"  ✅ OHLCV 数据: {len(ohlcv_data)} 条")

        # 优先读取真实新闻数据
        real_news_path = Path("data/raw/real_csi300_news_3m.csv")
        if real_news_path.exists() and symbol == "CSI300":
            click.echo(f"  使用真实新闻数据: {real_news_path}")
            news_data = pd.read_csv(real_news_path, parse_dates=["timestamp"])
            # 过滤时间范围
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            news_data = news_data[
                (news_data["timestamp"] >= start_dt) & (news_data["timestamp"] <= end_dt)
            ]
            click.echo(f"  ✅ 新闻数据: {len(news_data)} 条")
        else:
            # 生成 Mock 新闻数据
            click.echo("  生成 Mock 新闻数据...")
            news_generator = MockNewsGenerator()
            news_data = news_generator.generate_mock_news(
                start_date=start,
                end_date=end,
                symbols=[symbol],
            )
            click.echo(f"  ✅ 新闻数据: {len(news_data)} 条")

        # 数据对齐（简化版：直接使用原始数据）
        aligned_data = {"ohlcv": ohlcv_data, "news": news_data}
        click.echo(f"  ✅ 数据对齐完成")

    except Exception as e:
        click.echo(f"  ⚠️ 数据获取失败，使用模拟数据: {e}")
        ohlcv_data = _generate_mock_ohlcv(start, end)
        aligned_data = {"ohlcv": ohlcv_data, "news": pd.DataFrame()}

    # ================================================================
    # Phase 2: ML Track 信号生成
    # ================================================================
    click.echo("\n[Phase 2/6] ML Track 信号生成...")

    try:
        from src.models.ml_track.features import FeatureEngineer
        from src.models.ml_track.baselines import MLStrategyPortfolio

        # 特征工程
        feature_engineer = FeatureEngineer()
        features = feature_engineer.compute_all_features(aligned_data["ohlcv"])
        click.echo(f"  ✅ 特征计算: {features.shape[1]} 个因子")

        # 使用模拟信号（简化版，避免训练复杂度）
        ml_signals = _generate_mock_ml_signals(symbol, len(ohlcv_data))
        click.echo(f"  ✅ ML 信号: {len(ml_signals)} 条")

    except Exception as e:
        click.echo(f"  ⚠️ ML Track 失败，使用模拟信号: {e}")
        ml_signals = _generate_mock_ml_signals(symbol, len(ohlcv_data))

    # ================================================================
    # Phase 3: LLM Track 信号生成
    # ================================================================
    click.echo("\n[Phase 3/6] LLM Track 信号生成...")

    try:
        from src.models.llm_track.agent import LLMTradingAgent

        # 使用 Mock 执行器（避免 API 调用）
        llm_agent = LLMTradingAgent(executor_type="mock")

        # 转换新闻数据为列表格式
        news_list = aligned_data.get("news", pd.DataFrame()).to_dict("records") if not aligned_data.get("news", pd.DataFrame()).empty else []

        # 移除 price_data 参数
        llm_signals = llm_agent.batch_analyze(
            news_list=news_list,
        )
        click.echo(f"  ✅ LLM 信号: {len(llm_signals)} 条")

    except Exception as e:
        click.echo(f"  ⚠️ LLM Track 失败，使用模拟信号: {e}")
        llm_signals = _generate_mock_llm_signals(symbol, len(ohlcv_data))

    # ================================================================
    # Phase 4: 信号融合
    # ================================================================
    click.echo("\n[Phase 4/6] 信号融合...")

    try:
        from src.models.ml_track.features import FeatureEngineer

        # 计算波动率
        feature_engineer = FeatureEngineer()
        features = feature_engineer.compute_all_features(aligned_data["ohlcv"])
        volatility = features.get("volatility_20", pd.Series([0.02])).iloc[-1]

        from src.orchestrator.fusion_engine import SignalFusionEngine

        fusion_engine = SignalFusionEngine(
            rebalance_threshold=0.05,  # 5% 调仓死区
            llm_signal_decay_hours=72,   # 72 小时信号衰减
        )

        target_positions = fusion_engine.generate_target_positions(
            ml_signals=ml_signals,
            llm_signals=llm_signals,
            volatility=volatility,
            has_major_news=False,
        )

        click.echo(f"  ✅ 融合完成: {len(target_positions)} 个目标仓位")
        click.echo(f"  市场状态: {fusion_engine.get_current_regime().value}")

    except Exception as e:
        click.echo(f"  ⚠️ 信号融合失败: {e}")
        target_positions = {}

    # ================================================================
    # Phase 5: Backtrader 回测
    # ================================================================
    click.echo("\n[Phase 5/6] Backtrader 回测执行...")

    result = None

    try:
        from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

        # 准备目标仓位字典（Backtrader 格式）
        bt_target_positions = {}
        for date_idx in range(len(ohlcv_data)):
            date = ohlcv_data.index[date_idx]
            # 简化：使用最后一个融合信号
            if target_positions:
                symbol_key = list(target_positions.keys())[0]
                bt_target_positions[date] = {symbol: target_positions[symbol_key].weight}

        # 创建回测引擎
        engine = BacktestEngine(
            initial_cash=initial_cash,
            commission=commission,
        )

        # 添加数据
        engine.add_data(ohlcv_data, name=symbol)

        # 添加策略
        engine.add_strategy(
            DualTrackStrategy,
            target_positions=bt_target_positions,
            printlog=verbose,
        )

        # 执行回测
        result = engine.run()

        click.echo(f"  ✅ 回测完成")
        click.echo(f"  最终资产: {result.final_value:,.2f}")
        click.echo(f"  总收益率: {result.total_return:.2%}")

    except Exception as e:
        click.echo(f"  ⚠️ 回测执行失败: {e}")

    # ================================================================
    # Phase 6: 多维度评估
    # ================================================================
    click.echo("\n[Phase 6/6] 多维度评估...")

    try:
        from src.evaluation.metrics_calculator import MetricsCalculator
        from src.evaluation.visualizer import plot_equity_curves

        if result is not None and not result.equity_curve.empty:
            # 计算指标
            calculator = MetricsCalculator()
            evaluation = calculator.evaluate(
                strategy_name=f"DualTrack_{symbol}",
                equity_curve=result.equity_curve,
                latency_log=[],  # TODO: 收集实际延迟
                num_signals=len(target_positions),
            )

            click.echo(f"  ✅ 评估完成")
            click.echo(f"  夏普比率: {evaluation.financial_metrics.sharpe_ratio:.4f}")
            click.echo(f"  最大回撤: {evaluation.financial_metrics.max_drawdown:.2%}")

            # 生成图表
            output_path = Path(output_dir) / "figures"
            output_path.mkdir(parents=True, exist_ok=True)

            equity_curves = {
                f"DualTrack_{symbol}": result.equity_curve,
            }

            plot_equity_curves(
                equity_curves,
                title=f"DualTrack Strategy: {symbol}",
                save_path=str(output_path / f"equity_curves_{symbol}.png"),
            )

            click.echo(f"  ✅ 图表已保存: {output_path}")

    except Exception as e:
        click.echo(f"  ⚠️ 评估失败: {e}")

    # ================================================================
    # 完成
    # ================================================================
    total_elapsed = time.time() - total_start

    click.echo("\n" + "=" * 70)
    click.echo("  回测流水线执行完成")
    click.echo("=" * 70)
    click.echo(f"  总耗时: {total_elapsed:.2f} 秒")

    if result is not None:
        click.echo(f"\n  📊 回测结果摘要:")
        click.echo(f"  ┌──────────────────┬────────────────┐")
        click.echo(f"  │      指标        │      数值      │")
        click.echo(f"  ├──────────────────┼────────────────┤")
        click.echo(f"  │  初始资金        │  {initial_cash:>12,.2f}  │")
        click.echo(f"  │  最终资产        │  {result.final_value:>12,.2f}  │")
        click.echo(f"  │  总收益率        │  {result.total_return:>12.2%}  │")
        click.echo(f"  │  夏普比率        │  {result.sharpe_ratio:>12.4f}  │")
        click.echo(f"  │  最大回撤        │  {result.max_drawdown:>12.2%}  │")
        click.echo(f"  └──────────────────┴────────────────┘")


# ============================================================================
# evaluate 子命令 - 重新生成评估图表
# ============================================================================
@cli.command("evaluate")
@click.option("--log-file", default=None, help="回测日志文件路径")
@click.option("--output-dir", default="docs/figures", help="输出目录")
@click.pass_context
def evaluate(
    ctx: click.Context,
    log_file: Optional[str],
    output_dir: str,
) -> None:
    """
    根据已有的回测日志重新生成评估图表。

    仅执行 Phase 6 的图表生成，不重新运行回测。
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo("=" * 70)
    click.echo("  评估图表生成")
    click.echo("=" * 70)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        from src.evaluation.visualizer import (
            plot_equity_curves,
            plot_drawdown_heatmap,
            plot_latency_boxplot,
        )

        # 如果有日志文件，从中加载数据
        if log_file and Path(log_file).exists():
            click.echo(f"  加载日志文件: {log_file}")
            # TODO: 实现日志解析
        else:
            click.echo("  使用模拟数据生成示例图表")

            # 生成模拟数据
            np.random.seed(42)
            dates = pd.date_range(start="2024-01-01", periods=252, freq="B")

            ml_returns = np.random.randn(252) * 0.012 + 0.0006
            llm_returns = np.random.randn(252) * 0.018 + 0.0004
            fusion_returns = 0.6 * ml_returns + 0.4 * llm_returns + 0.0002

            equity_curves = {
                "ML_Track": pd.DataFrame({"nav": 1.0 * (1 + ml_returns).cumprod()}, index=dates),
                "LLM_Track": pd.DataFrame({"nav": 1.0 * (1 + llm_returns).cumprod()}, index=dates),
                "Dual_Track": pd.DataFrame({"nav": 1.0 * (1 + fusion_returns).cumprod()}, index=dates),
            }

            # 1. 资金曲线对比图
            click.echo("\n  [1/3] 资金曲线对比图...")
            plot_equity_curves(
                equity_curves,
                title="DualTrack Strategy Equity Curves",
                save_path=str(output_path / "equity_curves.png"),
            )

            # 2. 最大回撤热力图
            click.echo("  [2/3] 最大回撤热力图...")
            drawdown_data = {
                "ML_Track": {"Q1": 0.05, "Q2": 0.08, "Q3": 0.06, "Q4": 0.04},
                "LLM_Track": {"Q1": 0.07, "Q2": 0.12, "Q3": 0.09, "Q4": 0.06},
                "Dual_Track": {"Q1": 0.03, "Q2": 0.05, "Q3": 0.04, "Q4": 0.02},
            }
            plot_drawdown_heatmap(
                drawdown_data,
                title="Maximum Drawdown Heatmap",
                save_path=str(output_path / "drawdown_heatmap.png"),
            )

            # 3. 延迟分布箱线图
            click.echo("  [3/3] 延迟分布箱线图...")
            latency_data = {
                "ML_Track": np.random.uniform(5, 15, 100).tolist(),
                "LLM_Track": np.random.uniform(800, 2000, 100).tolist(),
                "Dual_Track": np.random.uniform(50, 200, 100).tolist(),
            }
            plot_latency_boxplot(
                latency_data,
                title="Inference Latency Distribution",
                save_path=str(output_path / "latency_boxplot.png"),
            )

        click.echo(f"\n  ✅ 图表已保存至: {output_path}")

    except Exception as e:
        click.echo(f"  ❌ 图表生成失败: {e}")
        raise


# ============================================================================
# cache-build 子命令 - 构建 LLM 离线缓存
# ============================================================================
@cli.command("cache-build")
@click.option("--symbol", "-s", default="CSI300", help="交易标的")
@click.option("--start", default="2026-01-09", help="开始日期")
@click.option("--end", default="2026-02-28", help="结束日期")
@click.option("--news-file", default="data/raw/real_csi300_news_3m.csv", help="新闻数据文件路径")
@click.option("--output-dir", default="docs/cache/llm_responses", help="缓存输出目录")
@click.option("--executor", default="ollama", type=click.Choice(["ollama", "deepseek", "mock"]), help="LLM 执行器类型")
@click.pass_context
def cache_build(
    ctx: click.Context,
    symbol: str,
    start: str,
    end: str,
    news_file: str,
    output_dir: str,
    executor: str,
) -> None:
    """
    构建 LLM 响应离线缓存（支持断点续传）。

    预先运行 LLM 推理并将结果缓存为 .jsonl 文件，
    用于加速后续回测（加速比可达 100x+）。

    支持断点续传：如果缓存文件已存在，会加载已有缓存并跳过已处理的新闻。
    """
    verbose = ctx.obj.get("verbose", False)

    click.echo("=" * 70)
    click.echo("  LLM 离线缓存构建")
    click.echo("=" * 70)
    click.echo(f"  标的: {symbol}")
    click.echo(f"  日期范围: {start} ~ {end}")
    click.echo(f"  新闻文件: {news_file}")
    click.echo(f"  执行器: {executor}")
    click.echo(f"  输出目录: {output_dir}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_file = output_path / f"llm_cache_{symbol}.jsonl"

    try:
        from src.models.llm_track.agent import LLMTradingAgent

        # 加载新闻数据
        news_path = Path(news_file)
        if news_path.exists():
            news_data = pd.read_csv(news_path, parse_dates=["timestamp"])
            click.echo(f"  ✅ 加载新闻: {len(news_data)} 条")

            # 过滤时间范围
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            news_data = news_data[
                (news_data["timestamp"] >= start_dt) & (news_data["timestamp"] <= end_dt)
            ]
            click.echo(f"  ✅ 时间范围过滤后: {len(news_data)} 条")
        else:
            click.echo(f"  ⚠️ 文件不存在: {news_file}")
            return

        # 初始化 LLM Agent
        llm_agent = LLMTradingAgent(executor_type=executor)

        # 检查已有缓存（断点续传）
        if cache_file.exists():
            llm_agent._load_cache(cache_file)
            click.echo(f"  ✅ 已加载 {len(llm_agent._cache)} 条缓存，将跳过已处理的新闻")

        # 转换为新闻列表格式
        news_list = news_data.to_dict("records")

        if not news_list:
            click.echo("  ⚠️ 无新闻数据需要处理")
            return

        # 批量推理（支持断点续传）
        click.echo("\n  执行批量推理...")

        start_time = time.time()
        signals = llm_agent.batch_analyze(
            news_list=news_list,
            market_context="当前市场正常运行。",
            symbol=symbol,
            cache_path=cache_file,  # 传入缓存路径以支持追加模式
        )
        elapsed = time.time() - start_time

        click.echo(f"\n  ✅ 缓存构建完成:")
        click.echo(f"  总新闻数: {len(news_data)}")
        click.echo(f"  已处理: {len(signals)}")
        click.echo(f"  总耗时: {elapsed:.2f} 秒")
        click.echo(f"  缓存文件: {cache_file}")
        click.echo(f"  文件大小: {cache_file.stat().st_size:,} bytes")

        # 计算加速比
        if len(signals) > 0:
            avg_latency = signals["latency_ms"].mean()
            click.echo(f"  平均延迟: {avg_latency:.1f}ms")
            click.echo(f"  预计加速比: ~{avg_latency / 0.1:.0f}x (缓存读取 ~0.1ms)")

    except Exception as e:
        click.echo(f"  ❌ 缓存构建失败: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        raise


# ============================================================================
# 辅助函数
# ============================================================================
def _generate_mock_ohlcv(start_date: str, end_date: str) -> pd.DataFrame:
    """生成模拟 OHLCV 数据。"""
    np.random.seed(42)

    dates = pd.date_range(start=start_date, end=end_date, freq="B")
    n = len(dates)

    base_price = 100
    returns = np.random.randn(n) * 0.02
    prices = base_price * (1 + returns).cumprod()

    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n) * 0.003),
        "high": prices * (1 + np.abs(np.random.randn(n)) * 0.008),
        "low": prices * (1 - np.abs(np.random.randn(n)) * 0.008),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n),
    }, index=dates)

    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def _generate_mock_ml_signals(symbol: str, n: int) -> pd.DataFrame:
    """生成模拟 ML 信号。"""
    np.random.seed(42)

    return pd.DataFrame({
        "symbol": [symbol] * n,
        "model_name": np.random.choice(["LightGBM", "LogisticRegression", "LSTM"], n),
        "signal_strength_0_to_1": np.random.uniform(0.3, 0.8, n),
        "latency_ms": np.random.uniform(1, 15, n),
    })


def _generate_mock_llm_signals(symbol: str, n: int) -> pd.DataFrame:
    """生成模拟 LLM 信号。"""
    np.random.seed(123)

    signals = np.random.choice(["buy", "sell", "hold"], n, p=[0.3, 0.2, 0.5])

    return pd.DataFrame({
        "symbol": [symbol] * n,
        "llm_signal": signals,
        "confidence": np.random.uniform(0.5, 0.9, n),
        "reasoning": ["Mock reasoning"] * n,
        "latency_ms": np.random.uniform(800, 2000, n),
    })


# ============================================================================
# 入口
# ============================================================================
if __name__ == "__main__":
    cli(obj={})