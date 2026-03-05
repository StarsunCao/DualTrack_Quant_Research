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


class AShareStrategy(bt.Strategy):
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
        self.dataclose = self.datas[0].close
        self.order = None
        self.trade_count = 0
        self.last_rebalance = None

        # 记录每日资产价值
        self.equity_curve: list[dict] = []

        # 交易记录
        self.trade_records: list[dict] = []
        self.position_records: list[dict] = []
        self.rebalance_records: list[dict] = []

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

    def next(self) -> None:
        """每个bar执行的逻辑。"""
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
                "symbol": data._name or "CSI300",
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

        # 执行调仓
        self.log(f"执行调仓, 目标仓位: {target}")

        for data in self.datas:
            symbol = data._name if hasattr(data, '_name') else "default"

            if symbol in target:
                weight = target[symbol]

                # A股特有：调仓死区检查
                pos = self.getposition(data)
                current_weight = (pos.size * data.close[0] / current_value) if current_value > 0 else 0.0
                weight_diff = abs(weight - current_weight)

                if weight_diff < self.params.rebalance_threshold:
                    self.log(f"  {symbol}: 死区跳过（变化{weight_diff:.2%}<{self.params.rebalance_threshold:.0%}）")
                    continue

                self.log(f"  {symbol}: 调整仓位 {current_weight:.2%}→{weight:.2%}")

                # A股特有：执行订单
                self._execute_order(data, weight)

            else:
                # 清仓
                self.log(f"  {symbol}: 清仓")
                pos = self.getposition(data)
                if pos.size > 0:
                    # 检查T+1
                    if self._check_t1_lock(symbol):
                        self.sell(data=data, size=pos.size)
                    else:
                        self.log(f"  ⚠️ {symbol}: T+1限制，不能清仓")

    def stop(self) -> None:
        """策略结束时调用。"""
        final_value = self.broker.getvalue()
        self.log(f"策略结束, 最终资产: {final_value:,.2f}", dt=self.datas[0].datetime.date(0))
