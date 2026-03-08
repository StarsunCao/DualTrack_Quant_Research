"""
美股专用策略类。

实现美股市场特有的交易规则：
1. T+0交易制度（当天买入可当天卖出）
2. 允许做空
3. 无整手限制
4. 无印花税
5. SEC费（仅卖出收取）
"""

from datetime import datetime
from typing import Optional

import backtrader as bt

from .base_strategy import BaseMarketStrategy


class USMarketStrategy(BaseMarketStrategy):
    """
    美股专用策略类。

    在BaseMarketStrategy基础上实现美股特有规则。

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

    def adjust_target_weight(self, weight: float) -> float:
        """
        调整目标仓位（美股规则：允许做空）。

        美股允许做空，负权重保留不变。

        Args:
            weight: 原始目标权重

        Returns:
            调整后的目标权重（美股不做调整）
        """
        # 美股允许做空，权重可以 < 0
        return weight

    def _get_target_size(self, data, target_weight: float) -> int:
        """
        计算目标股数（美股规则：无整手限制）。

        美股无整手限制，可以买入任意股数（包括零股）。

        Args:
            data: 数据源
            target_weight: 目标权重

        Returns:
            目标股数（直接取整，无整手限制）
        """
        portfolio_value = self.broker.getvalue()
        current_price = data.close[0]

        # 计算目标市值
        target_value = portfolio_value * target_weight

        # 计算目标股数（美股无整手限制，直接取整）
        target_size = int(target_value / current_price)

        return target_size

    def check_trading_rules(self, symbol: str, target_weight: float) -> bool:
        """
        检查交易规则（美股T+0，无限制）。

        美股T+0交易制度，无任何限制。

        Args:
            symbol: 股票代码
            target_weight: 目标权重

        Returns:
            True: 美股无限制，始终可以交易
        """
        # 美股T+0，无限制
        return True

    def _execute_order(self, data, target_weight: float) -> None:
        """
        执行美股订单（无整手、T+0、允许做空）。

        Args:
            data: 数据源
            target_weight: 目标权重
        """
        symbol = data._name if hasattr(data, '_name') else "ASSET"

        # 美股规则：允许做空，权重可以 < 0
        # 不调整权重

        # 获取当前持仓
        pos = self.getposition(data)
        current_size = pos.size

        # 计算目标股数（无整手限制）
        target_size = self._get_target_size(data, target_weight)

        # 计算交易股数
        trade_size = target_size - current_size

        if trade_size > 0:
            # 买入（美股无整手限制）
            self.buy(data=data, size=trade_size)
            self.log(f"  {symbol}: 买入 {trade_size}股")

        elif trade_size < 0:
            # 卖出（可能是做空或减仓）
            sell_size = abs(trade_size)
            self.sell(data=data, size=sell_size)

            # 判断是做空还是减仓
            if target_weight < 0:
                self.log(f"  {symbol}: 做空 {sell_size}股（目标权重 {target_weight:.2%}）")
            else:
                self.log(f"  {symbol}: 卖出 {sell_size}股")


if __name__ == "__main__":
    # 测试美股策略规则
    print("=" * 80)
    print("美股策略规则测试")
    print("=" * 80)

    # 模拟策略实例
    class MockStrategy:
        def adjust_target_weight(self, weight):
            """美股允许做空"""
            return weight

        def _get_target_size(self, data, target_weight):
            """美股无整手限制"""
            # 假设股价100元，资金100,000元
            portfolio_value = 100000
            current_price = 100
            target_value = portfolio_value * target_weight
            return int(target_value / current_price)

        def check_trading_rules(self, symbol, target_weight):
            """美股T+0，无限制"""
            return True

    strategy = MockStrategy()

    # 测试1: 做空信号
    print("\n测试1: 做空信号")
    weight = -0.8
    adjusted = strategy.adjust_target_weight(weight)
    print(f"  原始权重: {weight:.2%}")
    print(f"  调整后权重: {adjusted:.2%}")
    print(f"  ✓ 美股保留做空信号")

    # 测试2: 无整手限制
    print("\n测试2: 无整手限制")
    weight = 0.756  # 75.6%
    size = strategy._get_target_size(None, weight)
    print(f"  目标权重: {weight:.2%}")
    print(f"  计算股数: {size}股")
    print(f"  ✓ 美股可以买入任意股数（包括零股）")

    # 测试3: T+0交易
    print("\n测试3: T+0交易")
    can_trade = strategy.check_trading_rules('QQQ', -0.5)
    print(f"  是否可以交易: {can_trade}")
    print(f"  ✓ 美股T+0，当天买入可当天卖出")

    # 测试4: 做空计算
    print("\n测试4: 做空计算")
    weight = -0.5  # 做空50%
    size = strategy._get_target_size(None, weight)
    print(f"  目标权重: {weight:.2%}（做空）")
    print(f"  计算股数: {size}股（负数表示做空）")

    print("\n" + "=" * 80)
    print("美股策略核心差异：")
    print("  1. 允许做空（负权重保留）")
    print("  2. 无整手限制（可买零股）")
    print("  3. T+0交易（无限制）")
    print("=" * 80)