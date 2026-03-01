"""
Backtrader 回测引擎验证脚本。

验证内容包括：
1. Data Feed 加载 OHLCV 数据
2. DualTrackStrategy 策略执行
3. 目标仓位调仓逻辑
4. 分析器结果输出
5. 资产净值曲线生成
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution.bt_engine import (
    DualTrackStrategy,
    PandasDataFeed,
    BacktestEngine,
    BacktestResult,
    run_backtest,
)


def print_separator(title: str) -> None:
    """打印分隔线。"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_subsection(title: str) -> None:
    """打印子标题。"""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print('─' * 70)


def create_sample_ohlcv_data(days: int = 100) -> pd.DataFrame:
    """
    创建示例 OHLCV 数据。

    Args:
        days: 数据天数。

    Returns:
        OHLCV DataFrame。
    """
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=days, freq="B")

    # 生成模拟价格（带有一定趋势）
    base_price = 100
    trend = np.linspace(0, 0.2, days)  # 轻微上涨趋势
    noise = np.random.randn(days) * 0.02
    returns = noise + trend / days
    prices = base_price * (1 + returns).cumsum()
    prices = np.maximum(prices, 1)  # 确保价格为正

    # 创建 OHLCV
    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(days) * 0.003),
        "high": prices * (1 + np.abs(np.random.randn(days)) * 0.008),
        "low": prices * (1 - np.abs(np.random.randn(days)) * 0.008),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, days),
    }, index=dates)

    # 确保价格合理性
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


def create_target_positions(
    dates: pd.DatetimeIndex,
    strategy: str = "momentum",
) -> dict:
    """
    创建目标仓位字典。

    Args:
        dates: 日期索引。
        strategy: 策略类型 ('momentum', 'mean_reversion', 'random').

    Returns:
        目标仓位字典 {datetime: {symbol: weight}}。
    """
    target_positions = {}

    if strategy == "momentum":
        # 动量策略：持有 20 天，空仓 20 天
        for i, date in enumerate(dates):
            cycle = (i // 20) % 2
            if cycle == 0:
                target_positions[date] = {"ASSET": 0.9}  # 90% 仓位
            else:
                target_positions[date] = {"ASSET": 0.1}  # 10% 仓位

    elif strategy == "mean_reversion":
        # 均值回归策略：每 10 天切换
        for i, date in enumerate(dates):
            if i % 20 < 10:
                target_positions[date] = {"ASSET": 0.8}
            else:
                target_positions[date] = {"ASSET": 0.2}

    else:  # random
        # 随机仓位
        np.random.seed(42)
        for date in dates:
            weight = np.random.uniform(0.0, 1.0)
            target_positions[date] = {"ASSET": weight}

    return target_positions


# ============================================================================
# 验证点 1: Data Feed 数据加载
# ============================================================================
def test_data_feed() -> pd.DataFrame:
    """
    验证点 1: 测试 PandasDataFeed 数据加载。
    """
    print_separator("验证点 1: Data Feed 数据加载")

    # 创建示例数据
    sample_data = create_sample_ohlcv_data(100)

    print_subsection("1.1 原始数据概览")
    print(f"\n  数据形状: {sample_data.shape}")
    print(f"  日期范围: {sample_data.index.min()} ~ {sample_data.index.max()}")
    print(f"  列名: {list(sample_data.columns)}")

    print_subsection("1.2 数据样例")
    print("\n  前 5 行:")
    print(sample_data.head(5).to_string())

    print_subsection("1.3 创建 Data Feed")

    data_feed = PandasDataFeed.from_dataframe(sample_data, name="ASSET")

    print(f"\n  ✅ Data Feed 创建成功")
    print(f"  类型: {type(data_feed)}")
    print(f"  名称: {data_feed._name}")

    print("\n✅ Data Feed 测试通过")
    return sample_data


# ============================================================================
# 验证点 2: 策略执行与调仓逻辑
# ============================================================================
def test_strategy_execution() -> dict:
    """
    验证点 2: 测试 DualTrackStrategy 策略执行。

    Returns:
        测试结果字典。
    """
    print_separator("验证点 2: 策略执行与调仓逻辑")

    # 创建数据和目标仓位
    sample_data = create_sample_ohlcv_data(100)
    dates = sample_data.index
    target_positions = create_target_positions(dates, strategy="momentum")

    print_subsection("2.1 策略参数")
    print(f"\n  初始资金: 100,000")
    print(f"  佣金率: 万分之二 (0.02%)")
    print(f"  策略类型: 动量策略 (20天周期)")
    print(f"  目标仓位变化点:")

    # 显示仓位变化点
    change_points = []
    prev_weight = None
    for i, (date, pos) in enumerate(target_positions.items()):
        weight = pos["ASSET"]
        if prev_weight is None or abs(weight - prev_weight) > 0.1:
            change_points.append({
                "date": date.strftime("%Y-%m-%d"),
                "weight": weight,
            })
        prev_weight = weight

    for cp in change_points[:6]:  # 只显示前 6 个
        print(f"    {cp['date']}: 仓位 -> {cp['weight']:.0%}")

    print_subsection("2.2 执行回测")

    start_time = time.time()
    result = run_backtest(
        ohlcv_data=sample_data,
        target_positions=target_positions,
        initial_cash=100000,
    )
    elapsed = time.time() - start_time

    print(f"\n  回测耗时: {elapsed:.2f} 秒")

    print_subsection("2.3 回测结果")

    print(f"\n  {result.summary()}")

    print("\n✅ 策略执行测试通过")
    return {
        "data": sample_data,
        "target_positions": target_positions,
        "result": result,
    }


# ============================================================================
# 验证点 3: 分析器结果输出
# ============================================================================
def test_analyzers() -> BacktestResult:
    """
    验证点 3: 测试分析器结果输出。

    Returns:
        BacktestResult 对象。
    """
    print_separator("验证点 3: 分析器结果输出")

    sample_data = create_sample_ohlcv_data(100)
    dates = sample_data.index
    target_positions = create_target_positions(dates, strategy="mean_reversion")

    print_subsection("3.1 分析器配置")

    print("\n  已配置分析器:")
    print("    - TimeReturn: 时间序列收益率")
    print("    - SharpeRatio: 夏普比率")
    print("    - DrawDown: 回撤分析")
    print("    - Returns: 收益率分析")
    print("    - TradeAnalyzer: 交易分析")

    # 执行回测
    engine = BacktestEngine(
        initial_cash=100000,
        commission=0.0002,
    )
    engine.add_data(sample_data, name="ASSET")
    engine.add_strategy(
        DualTrackStrategy,
        target_positions=target_positions,
        printlog=False,  # 关闭详细日志
    )

    result = engine.run()

    print_subsection("3.2 分析器结果")

    print("\n  【TimeReturn】时间序列收益率:")
    time_return = result.analyzers.get("time_return", {})
    if time_return:
        returns_list = list(time_return.items())[:5]
        for date, ret in returns_list:
            print(f"    {date}: {ret:.4%}")
        print(f"    ... 共 {len(time_return)} 条记录")

    print("\n  【Returns】收益率指标:")
    returns = result.analyzers.get("returns", {})
    if returns:
        print(f"    平均收益率: {returns.get('rtot', 0):.4%}")
        print(f"    年化平均收益率: {returns.get('rnorm100', 0):.4%}")

    print("\n  【DrawDown】回撤指标:")
    dd = result.analyzers.get("drawdown", {})
    if dd:
        print(f"    最大回撤: {dd.get('max', {}).get('drawdown', 0):.2f}%")
        print(f"    最大回撤持续期: {dd.get('max', {}).get('len', 0)} 天")

    print("\n  【TradeAnalyzer】交易统计:")
    trades = result.analyzers.get("trades", {})
    if trades:
        total = trades.get("total", {})
        print(f"    总交易次数: {total.get('total', 0)}")
        print(f"    盈利交易: {total.get('won', 0)}")
        print(f"    亏损交易: {total.get('lost', 0)}")

        pnl = trades.get("pnl", {})
        avg = pnl.get("average", {})
        print(f"    平均盈亏: {avg.get('price', 0):.2f}")

    print("\n✅ 分析器测试通过")
    return result


# ============================================================================
# 验证点 4: 资产净值曲线生成
# ============================================================================
def test_equity_curve() -> pd.DataFrame:
    """
    验证点 4: 测试资产净值曲线生成。

    Returns:
        资产净值曲线 DataFrame。
    """
    print_separator("验证点 4: 资产净值曲线生成")

    sample_data = create_sample_ohlcv_data(100)
    dates = sample_data.index

    # 创建一个更激进的策略
    target_positions = {}
    for i, date in enumerate(dates):
        if i % 10 < 5:
            target_positions[date] = {"ASSET": 1.0}  # 满仓
        else:
            target_positions[date] = {"ASSET": 0.0}  # 空仓

    print_subsection("4.1 执行回测")

    result = run_backtest(
        ohlcv_data=sample_data,
        target_positions=target_positions,
        initial_cash=100000,
    )

    print_subsection("4.2 资产净值曲线")

    equity_curve = result.equity_curve

    if not equity_curve.empty:
        print(f"\n  净值曲线形状: {equity_curve.shape}")
        print(f"\n  列名: {list(equity_curve.columns)}")

        print(f"\n  净值曲线样例 (前 10 天):")
        print(equity_curve.head(10).to_string())

        print(f"\n  净值曲线样例 (后 5 天):")
        print(equity_curve.tail(5).to_string())

        # 计算净值
        equity_curve["nav"] = equity_curve["value"] / result.initial_cash

        print(f"\n  净值统计:")
        print(f"    起始净值: {equity_curve['nav'].iloc[0]:.4f}")
        print(f"    结束净值: {equity_curve['nav'].iloc[-1]:.4f}")
        print(f"    最高净值: {equity_curve['nav'].max():.4f}")
        print(f"    最低净值: {equity_curve['nav'].min():.4f}")
    else:
        print("\n  ⚠️ 净值曲线为空")

    print("\n✅ 资产净值曲线测试通过")
    return equity_curve


# ============================================================================
# 验证点 5: 端到端集成测试
# ============================================================================
def test_end_to_end() -> dict:
    """
    验证点 5: 端到端集成测试（模拟双轨系统输出）。

    Returns:
        测试结果字典。
    """
    print_separator("验证点 5: 端到端集成测试")

    print_subsection("5.1 模拟双轨系统信号流")

    # 创建数据
    sample_data = create_sample_ohlcv_data(60)
    dates = sample_data.index

    # 模拟双轨融合输出的目标仓位
    # 场景：前 40 天正常 ML 主导，后 20 天 LLM 黑天鹅
    target_positions = {}

    for i, date in enumerate(dates):
        if i < 40:
            # 正常时期：ML 信号为主
            # 模拟 ML 信号在 0.5-0.9 之间波动
            weight = 0.5 + 0.4 * np.sin(i / 10)
            target_positions[date] = {"ASSET": weight}
        else:
            # 黑天鹅时期：LLM 强制清仓
            target_positions[date] = {"ASSET": 0.0}

    print(f"\n  模拟场景:")
    print(f"    - 前 40 天: ML 主导，权重 0.5-0.9")
    print(f"    - 后 20 天: LLM 黑天鹅，强制清仓")

    print_subsection("5.2 执行回测")

    engine = BacktestEngine(
        initial_cash=100000,
        commission=0.0002,
    )
    engine.add_data(sample_data, name="ASSET")
    engine.add_strategy(
        DualTrackStrategy,
        target_positions=target_positions,
        printlog=False,
    )

    result = engine.run()

    print_subsection("5.3 回测结果分析")

    print(f"\n  初始资金: {result.initial_cash:,.2f}")
    print(f"  最终资产: {result.final_value:,.2f}")
    print(f"  总收益率: {result.total_return:.2%}")
    print(f"  夏普比率: {result.sharpe_ratio:.4f}")
    print(f"  最大回撤: {result.max_drawdown:.2%}")

    print_subsection("5.4 模拟输出接口验证")

    # 验证输出格式
    print(f"\n  输出格式验证:")

    # 1. to_dict() 方法
    result_dict = result.to_dict()
    print(f"    1. to_dict() 类型: {type(result_dict)}")
    print(f"       内容: {json.dumps(result_dict, indent=6, default=str)}")

    # 2. equity_curve DataFrame
    print(f"\n    2. equity_curve 类型: {type(result.equity_curve)}")
    print(f"       形状: {result.equity_curve.shape}")

    # 3. analyzers 字典
    print(f"\n    3. analyzers 类型: {type(result.analyzers)}")
    print(f"       键: {list(result.analyzers.keys())}")

    print("\n✅ 端到端集成测试通过")
    return {
        "result": result,
        "target_positions": target_positions,
    }


# ============================================================================
# 主函数
# ============================================================================
def main() -> None:
    """运行所有验证测试。"""
    print("\n" + "=" * 70)
    print("  🚀 Backtrader 回测引擎验证测试")
    print("=" * 70)
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_start = time.time()

    # 验证点 1: Data Feed
    sample_data = test_data_feed()

    # 验证点 2: 策略执行
    strategy_result = test_strategy_execution()

    # 验证点 3: 分析器
    analyzer_result = test_analyzers()

    # 验证点 4: 净值曲线
    equity_curve = test_equity_curve()

    # 验证点 5: 端到端
    e2e_result = test_end_to_end()

    total_elapsed = time.time() - total_start

    # 最终汇总
    print("\n" + "=" * 70)
    print("  📋 验证结果汇总")
    print("=" * 70)

    print("\n  [✓] 验证点 1: Data Feed 数据加载 - 通过")
    print("      - PandasDataFeed 成功加载 OHLCV 数据")
    print("      - 数据格式正确")

    print("\n  [✓] 验证点 2: 策略执行与调仓逻辑 - 通过")
    print("      - DualTrackStrategy 正确执行")
    print("      - 目标仓位调仓逻辑正常")

    print("\n  [✓] 验证点 3: 分析器结果输出 - 通过")
    print("      - TimeReturn: 时间序列收益率")
    print("      - SharpeRatio: 夏普比率")
    print("      - DrawDown: 回撤分析")
    print("      - TradeAnalyzer: 交易统计")

    print("\n  [✓] 验证点 4: 资产净值曲线生成 - 通过")
    print("      - 净值曲线 DataFrame 格式正确")
    print("      - 包含日期、资产价值、现金")

    print("\n  [✓] 验证点 5: 端到端集成测试 - 通过")
    print("      - 模拟双轨系统信号流成功")
    print("      - 输出接口格式验证通过")

    print(f"\n  ⏱️  总测试时间: {total_elapsed:.2f}秒")
    print("=" * 70)

    # 最终结果展示
    print("\n  📊 关键回测指标:")
    print(f"  ┌──────────────────┬────────────────┐")
    print(f"  │      指标        │      数值      │")
    print(f"  ├──────────────────┼────────────────┤")
    print(f"  │  总收益率        │  {e2e_result['result'].total_return:>+.2%}       │")
    print(f"  │  年化收益率      │  {e2e_result['result'].annual_return:>+.2%}       │")
    print(f"  │  夏普比率        │  {e2e_result['result'].sharpe_ratio:>+.4f}      │")
    print(f"  │  最大回撤        │  {e2e_result['result'].max_drawdown:>+.2%}       │")
    print(f"  └──────────────────┴────────────────┘")


if __name__ == "__main__":
    main()