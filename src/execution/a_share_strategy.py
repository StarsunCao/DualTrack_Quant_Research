"""
A股专用策略类。

实现A股市场特有的交易规则：
1. T+1交易制度（当天买入不能当天卖出）
2. 整手交易（买入必须是100股整数倍）
3. 调仓死区（避免频繁小额交易）
4. 最低佣金5元
5. 印花税（仅卖出收取0.05%）
6. 禁止做空
"""

from datetime import datetime
from typing import Optional

import backtrader as bt

from .base_strategy import BaseMarketStrategy


class AShareStrategy(BaseMarketStrategy):
    """
    A股专用策略类。

    在DualTrackStrategy基础上增加A股特有规则。

    参数:
        target_positions: 目标仓位字典
        rebalance_freq: 调仓频率（天数）
        printlog: 是否打印日志
        rebalance_threshold: 调仓死区阈值（默认5%）
    """

    params = (
        ("target_positions", None),
        ("rebalance_freq", 1),
        ("printlog", True),
        ("rebalance_threshold", 0.05),  # 调仓死区：5%
    )

    def __init__(self) -> None:
        """初始化策略。"""
        super().__init__()

        # A股特有：T+1锁（记录每只股票的买入bar索引）
        self.buy_bar_index: dict[str, int] = {}

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
            symbol = self.datas[0]._name or "CSI300"

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

            # A股特有：记录买入bar索引（用于T+1检查）
            if order.isbuy():
                self.buy_bar_index[symbol] = len(self)
                self.log(
                    f"买入执行: 价格={order.executed.price:.2f}, "
                    f"数量={order.executed.size:.0f}股, "
                    f"成本={order.executed.value:.2f}, "
                    f"手续费={order.executed.comm:.2f}"
                )
            else:
                self.log(
                    f"卖出执行: 价格={order.executed.price:.2f}, "
                    f"数量={order.executed.size:.0f}股, "
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

    def adjust_target_weight(self, weight: float) -> float:
        """
        调整目标仓位（A股规则：禁止做空）。

        Args:
            weight: 原始目标权重

        Returns:
            调整后的目标权重（负权重转为0）
        """
        if weight < 0:
            # A股规则：SELL信号（负权重）→ 减仓而非做空
            # 负权重转换为"保留仓位"
            # weight = -0.95 → 保留仓位 = 1 + (-0.95) = 0.05 (5%)
            # weight = -0.70 → 保留仓位 = 1 + (-0.70) = 0.30 (30%)
            # 置信度越高，保留仓位越少（减仓越多）
            target_weight = 1 + weight  # 负权重转正
            target_weight = max(0.0, target_weight)  # 确保非负
            self.log(f"  ⚠️ 做空信号 {weight:.2%} → 保留仓位 {target_weight:.2%}（减仓{-weight:.2%}）")
            return target_weight

        return weight

    def _get_target_size(self, data, target_weight: float) -> int:
        """
        计算目标股数（A股规则：买入必须整手）。

        Args:
            data: 数据源
            target_weight: 目标权重

        Returns:
            目标股数（已取整到100的倍数）
        """
        portfolio_value = self.broker.getvalue()
        current_price = data.close[0]

        # 计算目标市值
        target_value = portfolio_value * target_weight

        # 计算目标股数
        target_size = int(target_value / current_price)

        # A股规则：向下取整到最接近的整手数（100股）
        target_size = (target_size // 100) * 100

        return target_size

    def _get_target_size(self, data, target_weight: float) -> int:
        """
        计算目标股数（A股规则：买入必须整手）。

        Args:
            data: 数据源
            target_weight: 目标权重

        Returns:
            目标股数（已取整到100的倍数）
        """
        portfolio_value = self.broker.getvalue()
        current_price = data.close[0]

        # 计算目标市值
        target_value = portfolio_value * target_weight

        # 计算目标股数
        target_size = int(target_value / current_price)

        # A股规则：向下取整到最接近的整手数（100股）
        target_size = (target_size // 100) * 100

        return target_size

    def check_trading_rules(self, symbol: str, target_weight: float) -> bool:
        """
        检查交易规则（T+1限制）。

        Args:
            symbol: 股票代码
            target_weight: 目标权重

        Returns:
            True: 可以交易
            False: 不能交易（违反规则）
        """
        # T+1规则仅对卖出生效
        # 如果是买入（weight > current_weight），无需检查
        # 如果是卖出，需要检查 T+1 锁
        if target_weight < 0 or (symbol in self.buy_bar_index):
            return self._check_t1_lock(symbol)

        return True

    def _check_t1_lock(self, symbol: str) -> bool:
        """
        检查是否满足T+1规则（买入后至少隔一个交易日才能卖出）。

        Args:
            symbol: 股票代码

        Returns:
            True: 可以卖出
            False: 不能卖出（T+1限制）
        """
        if symbol not in self.buy_bar_index:
            return True  # 没有买入记录，可以卖出

        buy_bar = self.buy_bar_index[symbol]
        current_bar = len(self)

        # T+1规则：必须至少跨越一个交易日
        return current_bar > buy_bar

    def _check_t1_lock(self, symbol: str) -> bool:
        """
        检查是否满足T+1规则（买入后至少隔一个交易日才能卖出）。

        Args:
            symbol: 股票代码

        Returns:
            True: 可以卖出
            False: 不能卖出（T+1限制）
        """
        if symbol not in self.buy_bar_index:
            return True  # 没有买入记录，可以卖出

        buy_bar = self.buy_bar_index[symbol]
        current_bar = len(self)

        # T+1规则：必须至少跨越一个交易日
        return current_bar > buy_bar

    def notify_order(self, order: bt.Order) -> None:
        """订单状态通知（A股特有：记录买入bar索引）。"""
        super().notify_order(order)

        # A股特有：记录买入bar索引（用于T+1检查）
        if order.status in [order.Completed] and order.isbuy():
            symbol = self.datas[0]._name or "CSI300"
            self.buy_bar_index[symbol] = len(self)

    def _execute_order(self, data, target_weight: float) -> None:
        """
        执行A股订单（考虑整手、T+1、做空限制）。

        Args:
            data: 数据源
            target_weight: 目标权重
        """
        symbol = data._name if hasattr(data, '_name') else "CSI300"

        # A股规则：禁止做空
        if target_weight < 0:
            self.log(f"  ⚠️ {symbol}: 做空信号已限制为0")
            target_weight = 0.0

        # 获取当前持仓
        pos = self.getposition(data)
        current_size = pos.size

        # 计算目标股数（整手）
        target_size = self._get_target_size(data, target_weight)

        # 计算交易股数
        trade_size = target_size - current_size

        if trade_size > 0:
            # 买入：必须是整手
            lot_size = (trade_size // 100) * 100
            if lot_size >= 100:
                self.buy(data=data, size=lot_size)
                self.log(f"  {symbol}: 买入 {lot_size}股 ({lot_size//100}手)")
            else:
                self.log(f"  {symbol}: 买入量不足1手，跳过")

        elif trade_size < 0:
            # 卖出：检查T+1规则
            if not self._check_t1_lock(symbol):
                bars_held = len(self) - self.buy_bar_index.get(symbol, 0)
                self.log(f"  ⚠️ {symbol}: T+1限制，持有{bars_held}天，不能卖出")
                return

            # 卖出：可以卖出零股（清仓时）
            sell_size = abs(trade_size)
            if sell_size > 0:
                self.sell(data=data, size=sell_size)
                self.log(f"  {symbol}: 卖出 {sell_size}股")
