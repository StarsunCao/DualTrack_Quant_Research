"""
Backtrader 引擎订单执行验证测试。

验证重点：
1. 订单是否真实成交（而非幽灵订单）
2. notify_order/notify_trade 回调是否正确触发
3. 分析器结果提取
4. 净值曲线生成

测试场景：
- 20天 OHLCV 数据
- 第5天满仓(1.0)，第10天清仓(0.0)，第15天做空(-0.5)
"""

import sys
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.execution.bt_engine import (
    DualTrackStrategy,
    PandasDataFeed,
    BacktestEngine,
    BacktestResult,
    run_backtest,
)

# ============================================================================
# Python 3.10+ 兼容性修复
# ============================================================================
import collections
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Callable = collections.abc.Callable

import backtrader as bt


# ============================================================================
# 订单记录数据类
# ============================================================================
@dataclass
class OrderRecord:
    """订单记录。"""
    date: str
    status: str
    direction: str
    price: float
    size: float
    value: float
    commission: float


@dataclass
class TradeRecord:
    """交易记录。"""
    date: str
    direction: str
    price: float
    size: float
    pnl: float
    pnl_comm: float


# ============================================================================
# 带订单记录的策略类
# ============================================================================
class VerifiableStrategy(bt.Strategy):
    """
    可验证的策略类。

    记录所有订单和交易，用于验证订单是否真实成交。
    """

    params = (
        ("target_positions", None),
        ("rebalance_freq", 1),
        ("printlog", True),
    )

    def __init__(self) -> None:
        """初始化策略。"""
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.last_rebalance = None

        # 订单和交易记录
        self.order_records: List[OrderRecord] = []
        self.trade_records: List[TradeRecord] = []

        # 资产净值曲线
        self.equity_curve: List[dict] = []

    def log(self, txt: str, dt: Optional[datetime] = None) -> None:
        """打印日志。"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"  [{dt.isoformat()}] {txt}")

    def notify_order(self, order: bt.Order) -> None:
        """
        订单状态通知 - 关键验证点。

        记录订单状态变化，验证订单是否成交。
        """
        current_date = self.datas[0].datetime.date(0).isoformat()

        # 订单状态
        status_map = {
            order.Submitted: "Submitted",
            order.Accepted: "Accepted",
            order.Completed: "Completed",
            order.Canceled: "Canceled",
            order.Margin: "Margin",
            order.Rejected: "Rejected",
        }
        status = status_map.get(order.status, f"Unknown({order.status})")

        # 记录订单
        record = OrderRecord(
            date=current_date,
            status=status,
            direction="BUY" if order.isbuy() else "SELL",
            price=order.executed.price if order.status == order.Completed else 0.0,
            size=order.executed.size if order.status == order.Completed else 0.0,
            value=order.executed.value if order.status == order.Completed else 0.0,
            commission=order.executed.comm if order.status == order.Completed else 0.0,
        )
        self.order_records.append(record)

        # 打印详细日志
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status == order.Completed:
            if order.isbuy():
                self.log(
                    f"✅ 买入成交: 价格={order.executed.price:.2f}, "
                    f"数量={order.executed.size:.2f}, "
                    f"成本={order.executed.value:.2f}, "
                    f"佣金={order.executed.comm:.2f}"
                )
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log(
                    f"✅ 卖出成交: 价格={order.executed.price:.2f}, "
                    f"数量={order.executed.size:.2f}, "
                    f"成本={order.executed.value:.2f}, "
                    f"佣金={order.executed.comm:.2f}"
                )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"⚠️ 订单失败: {status}")

        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """
        交易通知 - 验证盈亏计算。

        记录每笔已关闭的交易。
        """
        if not trade.isclosed:
            return

        current_date = self.datas[0].datetime.date(0).isoformat()

        # 安全获取交易方向
        # trade.size > 0 表示买入开仓, < 0 表示卖出开仓
        direction = "LONG" if trade.size > 0 else "SHORT"

        record = TradeRecord(
            date=current_date,
            direction=direction,
            price=trade.price,
            size=abs(trade.size),
            pnl=trade.pnl,
            pnl_comm=trade.pnlcomm,
        )
        self.trade_records.append(record)

        self.log(
            f"💰 交易盈亏: 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}"
        )

    def next(self) -> None:
        """每个 bar 执行的逻辑。"""
        # 记录资产净值
        current_date = self.datas[0].datetime.date(0)
        current_value = self.broker.getvalue()
        current_cash = self.broker.getcash()

        self.equity_curve.append({
            "date": current_date,
            "value": current_value,
            "cash": current_cash,
            "position": self.position.size,
        })

        # 检查未完成订单
        if self.order:
            return

        # 获取目标仓位
        target_positions = self.params.target_positions
        if not target_positions:
            return

        # 当前日期
        current_dt = self.datas[0].datetime.datetime(0)

        # 查找目标仓位
        target = None
        for date_key, positions in target_positions.items():
            if isinstance(date_key, str):
                date_key = pd.to_datetime(date_key)

            if hasattr(date_key, 'date'):
                key_date = date_key.date()
            else:
                key_date = pd.to_datetime(date_key).date()

            if key_date == current_date:
                target = positions
                break

        if not target:
            return

        # 检查调仓频率
        if self.last_rebalance is not None:
            days_since_last = (current_date - self.last_rebalance).days
            if days_since_last < self.params.rebalance_freq:
                return

        self.last_rebalance = current_date

        # 执行调仓
        for data in self.datas:
            symbol = data._name if hasattr(data, '_name') else "default"

            if symbol in target:
                weight = target[symbol]
                self.log(f"📈 调仓信号: {symbol} -> {weight:.1%}")
                self.order_target_percent(data=data, target=weight)
            else:
                self.log(f"📉 调仓信号: {symbol} -> 清仓")
                self.order_target_percent(data=data, target=0.0)

    def stop(self) -> None:
        """策略结束时调用。"""
        final_value = self.broker.getvalue()
        self.log(f"🏁 策略结束, 最终资产: {final_value:,.2f}")


# ============================================================================
# 测试函数
# ============================================================================
def create_ohlcv_data(days: int = 20) -> pd.DataFrame:
    """
    创建 20 天 OHLCV 数据。

    Args:
        days: 数据天数。

    Returns:
        OHLCV DataFrame。
    """
    np.random.seed(42)
    dates = pd.date_range(start="2024-01-01", periods=days, freq="B")

    # 生成模拟价格（有一定波动）
    base_price = 100
    returns = np.random.randn(days) * 0.02
    prices = base_price * (1 + returns).cumprod()
    prices = np.maximum(prices, 10)  # 确保价格为正

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


def create_target_positions(dates: pd.DatetimeIndex) -> dict:
    """
    创建目标仓位字典。

    策略：
    - 第5天: 满仓 (1.0)
    - 第10天: 清仓 (0.0)
    - 第15天: 做空 (-0.5)

    Args:
        dates: 日期索引。

    Returns:
        目标仓位字典。
    """
    target_positions = {}

    for i, date in enumerate(dates):
        # 索引从 0 开始
        if i == 4:  # 第5天 (索引4)
            target_positions[date] = {"ASSET": 1.0}
        elif i == 9:  # 第10天 (索引9)
            target_positions[date] = {"ASSET": 0.0}
        elif i == 14:  # 第15天 (索引14)
            target_positions[date] = {"ASSET": -0.5}
        else:
            # 其他日期保持上一个仓位（通过不设置实现）
            pass

    return target_positions


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


def main() -> None:
    """主测试函数。"""
    print_separator("Backtrader 订单执行验证测试")

    # =========================================================================
    # 步骤 1: 构建 20 天 OHLCV 数据
    # =========================================================================
    print_subsection("步骤 1: 构建 20 天 OHLCV 数据")

    ohlcv_data = create_ohlcv_data(20)
    print(f"\n  数据形状: {ohlcv_data.shape}")
    print(f"  日期范围: {ohlcv_data.index.min().date()} ~ {ohlcv_data.index.max().date()}")
    print("\n  数据预览 (前5行):")
    print(ohlcv_data.head().to_string())

    # =========================================================================
    # 步骤 2: 设置目标仓位
    # =========================================================================
    print_subsection("步骤 2: 设置目标仓位")

    target_positions = create_target_positions(ohlcv_data.index)

    print("\n  目标仓位计划:")
    print("  ┌────────────┬────────────┬─────────────────────┐")
    print("  │    日期    │    动作    │       目标仓位      │")
    print("  ├────────────┼────────────┼─────────────────────┤")

    for date, pos in target_positions.items():
        date_str = date.strftime("%Y-%m-%d")
        weight = pos["ASSET"]
        if weight == 1.0:
            action = "满仓买入"
        elif weight == 0.0:
            action = "清仓"
        elif weight < 0:
            action = "做空"
        else:
            action = "持仓"
        print(f"  │ {date_str} │ {action:^8} │ {weight:^19.1%} │")

    print("  └────────────┴────────────┴─────────────────────┘")

    # =========================================================================
    # 步骤 3: 创建回测引擎
    # =========================================================================
    print_subsection("步骤 3: 创建回测引擎")

    cerebro = bt.Cerebro()

    # 设置初始资金
    initial_cash = 100000.0
    cerebro.broker.setcash(initial_cash)

    # 设置佣金
    cerebro.broker.setcommission(commission=0.0002)  # 万分之二

    # 添加数据
    data_feed = PandasDataFeed.from_dataframe(ohlcv_data, name="ASSET")
    cerebro.adddata(data_feed)

    # 添加策略
    cerebro.addstrategy(
        VerifiableStrategy,
        target_positions=target_positions,
        printlog=True,
    )

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    print(f"\n  初始资金: {initial_cash:,.2f}")
    print(f"  佣金率: 0.02%")
    print(f"  数据: ASSET (20天)")

    # =========================================================================
    # 步骤 4: 执行回测
    # =========================================================================
    print_subsection("步骤 4: 执行回测")

    print("\n  执行日志:")
    print("  " + "─" * 60)

    results = cerebro.run()
    strategy = results[0]

    final_value = cerebro.broker.getvalue()
    total_return = (final_value - initial_cash) / initial_cash

    # =========================================================================
    # 步骤 5: 打印成交记录
    # =========================================================================
    print_separator("步骤 5: 成交记录验证")

    print_subsection("5.1 订单记录 (Order Records)")

    order_records = strategy.order_records
    completed_orders = [r for r in order_records if r.status == "Completed"]
    failed_orders = [r for r in order_records if r.status in ["Canceled", "Margin", "Rejected"]]

    print(f"\n  总订单数: {len(order_records)}")
    print(f"  成交订单: {len(completed_orders)}")
    print(f"  失败订单: {len(failed_orders)}")

    if completed_orders:
        print("\n  成交明细:")
        print("  ┌────────────┬──────────┬───────────┬────────────┬───────────────┬──────────────┐")
        print("  │    日期    │   方向   │   价格    │    数量    │     成交额    │     佣金     │")
        print("  ├────────────┼──────────┼───────────┼────────────┼───────────────┼──────────────┤")

        for r in completed_orders:
            print(f"  │ {r.date} │ {r.direction:^8} │ {r.price:>9.2f} │ {r.size:>10.2f} │ {r.value:>13.2f} │ {r.commission:>12.2f} │")

        print("  └────────────┴──────────┴───────────┴────────────┴───────────────┴──────────────┘")
    else:
        print("\n  ⚠️ 无成交订单！可能存在问题。")

    if failed_orders:
        print("\n  ⚠️ 失败订单明细:")
        for r in failed_orders:
            print(f"    {r.date}: {r.direction} - {r.status}")

    print_subsection("5.2 交易记录 (Trade Records)")

    trade_records = strategy.trade_records

    print(f"\n  已关闭交易数: {len(trade_records)}")

    if trade_records:
        print("\n  交易明细:")
        print("  ┌────────────┬──────────┬───────────┬───────────┬────────────┬────────────┐")
        print("  │    日期    │   方向   │   价格    │   数量    │    毛盈亏   │   净盈亏   │")
        print("  ├────────────┼──────────┼───────────┼───────────┼────────────┼────────────┤")

        for r in trade_records:
            print(f"  │ {r.date} │ {r.direction:^8} │ {r.price:>9.2f} │ {r.size:>9.2f} │ {r.pnl:>10.2f} │ {r.pnl_comm:>10.2f} │")

        print("  └────────────┴──────────┴───────────┴───────────┴────────────┴────────────┘")
    else:
        print("\n  ℹ️ 无已关闭的交易记录")

    # =========================================================================
    # 步骤 6: 提取分析器结果
    # =========================================================================
    print_separator("步骤 6: 分析器结果")

    # Sharpe Ratio
    sharpe_analysis = strategy.analyzers.sharpe_ratio.get_analysis()
    sharpe_ratio = sharpe_analysis.get("sharperatio", 0.0) or 0.0

    print("\n  【SharpeRatio - 夏普比率】")
    print(f"    夏普比率: {sharpe_ratio:.4f}")
    print(f"    分析器原始输出: {sharpe_analysis}")

    # Max Drawdown
    dd_analysis = strategy.analyzers.drawdown.get_analysis()
    max_dd = dd_analysis.get("max", {}).get("drawdown", 0.0) / 100
    max_dd_len = dd_analysis.get("max", {}).get("len", 0)

    print("\n  【MaxDrawdown - 最大回撤】")
    print(f"    最大回撤: {max_dd:.2%}")
    print(f"    回撤持续期: {max_dd_len} 天")
    print(f"    分析器原始输出: {dd_analysis}")

    # Returns
    returns_analysis = strategy.analyzers.returns.get_analysis()

    print("\n  【Returns - 收益率】")
    print(f"    总收益率: {returns_analysis.get('rtot', 0):.4%}")
    print(f"    平均收益率: {returns_analysis.get('ravg', 0):.4%}")
    print(f"    年化收益率: {returns_analysis.get('rnorm100', 0):.4%}")

    # Trade Analyzer
    trade_analysis = strategy.analyzers.trades.get_analysis()

    print("\n  【TradeAnalyzer - 交易统计】")
    total_trades = trade_analysis.get("total", {}).get("total", 0)
    won_trades = trade_analysis.get("total", {}).get("won", 0)
    lost_trades = trade_analysis.get("total", {}).get("lost", 0)

    print(f"    总交易次数: {total_trades}")
    print(f"    盈利交易: {won_trades}")
    print(f"    亏损交易: {lost_trades}")

    if total_trades > 0:
        win_rate = won_trades / total_trades
        print(f"    胜率: {win_rate:.2%}")

    # =========================================================================
    # 步骤 7: 净值曲线
    # =========================================================================
    print_separator("步骤 7: 净值曲线")

    equity_curve = pd.DataFrame(strategy.equity_curve)

    if not equity_curve.empty:
        equity_curve["date"] = pd.to_datetime(equity_curve["date"])
        equity_curve.set_index("date", inplace=True)
        equity_curve["nav"] = equity_curve["value"] / initial_cash

        print(f"\n  净值曲线形状: {equity_curve.shape}")
        print(f"  列名: {list(equity_curve.columns)}")

        print("\n  净值曲线 - 头部3行:")
        print(equity_curve.head(3).to_string())

        print("\n  净值曲线 - 尾部3行:")
        print(equity_curve.tail(3).to_string())

        print("\n  净值统计:")
        print(f"    起始净值: {equity_curve['nav'].iloc[0]:.4f}")
        print(f"    结束净值: {equity_curve['nav'].iloc[-1]:.4f}")
        print(f"    最高净值: {equity_curve['nav'].max():.4f}")
        print(f"    最低净值: {equity_curve['nav'].min():.4f}")

    # =========================================================================
    # 步骤 8: 验证总结
    # =========================================================================
    print_separator("验证总结")

    print("\n  ✅ 订单执行验证:")
    print(f"     - 成交订单数: {len(completed_orders)}")
    print(f"     - 失败订单数: {len(failed_orders)}")

    if len(completed_orders) > 0 and len(failed_orders) == 0:
        print("     - 状态: 所有订单成功成交，无幽灵订单 ✓")
    else:
        print("     - 状态: ⚠️ 存在问题，请检查订单状态")

    print("\n  ✅ 回调函数验证:")
    print(f"     - notify_order 记录数: {len(order_records)}")
    print(f"     - notify_trade 记录数: {len(trade_records)}")

    print("\n  ✅ 分析器验证:")
    print(f"     - SharpeRatio: {sharpe_ratio:.4f}")
    print(f"     - MaxDrawdown: {max_dd:.2%}")

    print("\n  ✅ 净值曲线验证:")
    print(f"     - 数据点数: {len(equity_curve)}")
    print(f"     - 最终资产: {final_value:,.2f}")
    print(f"     - 总收益率: {total_return:.2%}")

    print("\n" + "=" * 70)
    print("  测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()