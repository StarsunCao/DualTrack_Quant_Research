"""
策略抽象基类。

定义统一的策略接口，提取不同市场的通用逻辑。
支持 A 股和美股市场的差异化实现。
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import backtrader as bt


class BaseMarketStrategy(bt.Strategy):
    """
    市场策略抽象基类。

    定义不同市场策略的统一接口，提取通用逻辑。

    参数:
        target_positions: 目标仓位字典
        rebalance_freq: 调仓频率（天数）
        printlog: 是否打印日志
        rebalance_threshold: 调仓死区阈值
    """

    params = (
        ("target_positions", None),
        ("rebalance_freq", 1),
        ("printlog", True),
        ("rebalance_threshold", 0.05),  # 调仓死区：5%
    )

    def __init__(self) -> None:
        """初始化策略。"""
        self.dataclose = self.datas[0].close
        self.order = None
        self.trade_count = 0
        self.last_rebalance = None

        # 记录每日资产价值
        self.equity_curve: list[dict] = []

        # 详细交易记录
        self.trade_records: list[dict] = []
        self.position_records: list[dict] = []
        self.rebalance_records: list[dict] = []

    def log(self, txt: str, dt: Optional[datetime] = None) -> None:
        """打印日志。"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"  [{dt.isoformat()}] {txt}")

    def notify_order(self, order: bt.Order) -> None:
        """订单状态通知。"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0)
            symbol = self.datas[0]._name or "ASSET"

            record = {
                "date": dt.isoformat(),
                "type": "买入" if order.isbuy() else "卖出",
                "price": order.executed.price,
                "size": order.executed.size,
                "value": order.executed.value,
                "commission": order.executed.comm,
                "symbol": symbol,
            }
            self.trade_records.append(record)

            if order.isbuy():
                self.log(
                    f"买入执行: 价格={order.executed.price:.2f}, "
                    f"数量={order.executed.size:.0f}, "
                    f"成本={order.executed.value:.2f}, "
                    f"手续费={order.executed.comm:.2f}"
                )
            else:
                self.log(
                    f"卖出执行: 价格={order.executed.price:.2f}, "
                    f"数量={order.executed.size:.0f}, "
                    f"成本={order.executed.value:.2f}, "
                    f"手续费={order.executed.comm:.2f}"
                )
            self.trade_count += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单取消/拒绝: {order.status}")

        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """交易通知。"""
        if not trade.isclosed:
            return
        self.log(f"交易盈亏: 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}")

    @abstractmethod
    def adjust_target_weight(self, weight: float) -> float:
        """
        调整目标仓位（处理做空限制等市场规则）。

        Args:
            weight: 原始目标权重

        Returns:
            调整后的目标权重
        """
        raise NotImplementedError

    @abstractmethod
    def _get_target_size(self, data, target_weight: float) -> int:
        """
        计算目标持仓数量（处理整手限制等市场规则）。

        Args:
            data: 数据源
            target_weight: 目标权重

        Returns:
            目标持仓数量
        """
        raise NotImplementedError

    @abstractmethod
    def check_trading_rules(self, symbol: str, target_weight: float) -> bool:
        """
        检查交易规则（T+1限制等）。

        Args:
            symbol: 股票代码
            target_weight: 目标权重

        Returns:
            True: 可以交易
            False: 不能交易（违反规则）
        """
        raise NotImplementedError

    def _execute_order(self, data, target_weight: float) -> None:
        """
        执行订单（通用逻辑，子类可覆盖）。

        Args:
            data: 数据源
            target_weight: 目标权重
        """
        symbol = data._name if hasattr(data, '_name') else "ASSET"

        # 调整目标权重（处理做空限制）
        target_weight = self.adjust_target_weight(target_weight)

        # 检查交易规则
        if not self.check_trading_rules(symbol, target_weight):
            return

        # 获取当前持仓
        pos = self.getposition(data)
        current_size = pos.size

        # 计算目标股数
        target_size = self._get_target_size(data, target_weight)

        # 计算交易股数
        trade_size = target_size - current_size

        if trade_size > 0:
            # 买入
            self.buy(data=data, size=trade_size)
            self.log(f"  {symbol}: 买入 {trade_size}股")

        elif trade_size < 0:
            # 卖出
            sell_size = abs(trade_size)
            self.sell(data=data, size=sell_size)
            self.log(f"  {symbol}: 卖出 {sell_size}股")

    def next(self) -> None:
        """每个 bar 执行的逻辑。"""
        current_date = self.datas[0].datetime.date(0)
        current_value = self.broker.getvalue()

        # 记录每日资产价值
        self.equity_curve.append({
            "date": current_date,
            "value": current_value,
            "cash": self.broker.getcash(),
        })

        # 记录每日持仓
        for data in self.datas:
            pos = self.getposition(data)
            self.position_records.append({
                "date": current_date,
                "symbol": data._name or "ASSET",
                "position_size": pos.size,
                "position_value": pos.size * data.close[0] if pos else 0,
                "cash": self.broker.getcash(),
                "total_value": current_value,
                "close_price": data.close[0],
            })

        # 获取目标仓位
        target_positions = self.params.target_positions
        if not target_positions:
            return

        # 查找当前日期对应的目标仓位
        target = None
        for date_key, positions in target_positions.items():
            # date_key可能是Timestamp或datetime
            if hasattr(date_key, 'date'):
                key_date = date_key.date() if hasattr(date_key, 'date') else date_key
            else:
                key_date = date_key

            if key_date == current_date:
                target = positions
                break

        if not target:
            return

        # 检查调仓频率
        if self.last_rebalance is not None:
            try:
                days_since_last = (current_date - self.last_rebalance).days
                if days_since_last < self.params.rebalance_freq:
                    return
            except (AttributeError, TypeError):
                pass

        self.last_rebalance = current_date

        # 记录调仓信息
        for data in self.datas:
            symbol = data._name if hasattr(data, '_name') else "default"
            if symbol in target:
                pos = self.getposition(data)
                current_weight = (pos.size * data.close[0] / current_value) if current_value > 0 else 0.0

                rebalance_record = {
                    "date": current_date.isoformat(),
                    "symbol": symbol,
                    "target_weight": target[symbol],
                    "current_weight": current_weight,
                    "portfolio_value": current_value,
                    "close_price": data.close[0],
                }
                self.rebalance_records.append(rebalance_record)

        # 执行调仓
        self.log(f"执行调仓, 目标仓位: {target}")

        for data in self.datas:
            symbol = data._name if hasattr(data, '_name') else "default"

            if symbol in target:
                weight = target[symbol]

                # 调仓死区检查
                pos = self.getposition(data)
                current_weight = (pos.size * data.close[0] / current_value) if current_value > 0 else 0.0
                weight_diff = abs(weight - current_weight)

                if weight_diff < self.params.rebalance_threshold:
                    self.log(f"  {symbol}: 死区跳过（变化{weight_diff:.2%}<{self.params.rebalance_threshold:.0%}）")
                    continue

                self.log(f"  {symbol}: 调整仓位 {current_weight:.2%}→{weight:.2%}")

                # 执行订单
                self._execute_order(data, weight)

            else:
                # 清仓
                self.log(f"  {symbol}: 清仓")
                pos = self.getposition(data)
                if pos.size > 0:
                    # 检查交易规则
                    if self.check_trading_rules(symbol, 0.0):
                        self.sell(data=data, size=pos.size)
                    else:
                        self.log(f"  ⚠️ {symbol}: 交易规则限制，不能清仓")

    def stop(self) -> None:
        """策略结束时调用。"""
        final_value = self.broker.getvalue()
        self.log(f"策略结束, 最终资产: {final_value:,.2f}", dt=self.datas[0].datetime.date(0))


if __name__ == "__main__":
    print("=" * 80)
    print("BaseMarketStrategy 抽象基类")
    print("=" * 80)
    print("\n这是一个抽象基类，不能直接实例化。")
    print("请使用具体实现：")
    print("  - AShareStrategy (A股策略)")
    print("  - USMarketStrategy (美股策略)")