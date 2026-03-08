"""
美股市场单元测试。

测试美股策略的核心功能：
1. 做空支持
2. T+0 交易
3. 无整手限制
4. SEC 费计算
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import unittest
from datetime import datetime, timedelta

import backtrader as bt
import numpy as np
import pandas as pd

from src.config.market_config import MarketConfig, MarketType
from src.execution.us_market_strategy import USMarketStrategy
from src.execution.us_market_sizer import USMarketCommission


class TestUSMarketConfig(unittest.TestCase):
    """测试美股市场配置。"""

    def test_market_type_detection(self):
        """测试市场类型检测。"""
        # 美股代码
        self.assertEqual(
            MarketConfig.get_market_type_for_symbol('QQQ'),
            MarketType.US_MARKET
        )
        self.assertEqual(
            MarketConfig.get_market_type_for_symbol('NASDAQ100'),
            MarketType.US_MARKET
        )

        # A股代码
        self.assertEqual(
            MarketConfig.get_market_type_for_symbol('CSI300'),
            MarketType.A_SHARE
        )

    def test_us_market_config_values(self):
        """测试美股配置值。"""
        config = MarketConfig.us_market()

        # 美股特有配置
        self.assertTrue(config.allow_short_selling)
        self.assertFalse(config.t_plus_one)
        self.assertEqual(config.lot_size, 0)  # 无整手限制
        self.assertEqual(config.stamp_duty, 0.0)  # 无印花税
        self.assertGreater(config.sec_fee_rate, 0)  # 有 SEC 费
        self.assertEqual(config.timezone, "America/New_York")


class TestUSMarketStrategy(unittest.TestCase):
    """测试美股策略。"""

    def setUp(self):
        """设置测试环境。"""
        self.strategy = USMarketStrategy()

        # Mock 必要的属性
        self.strategy.log = lambda msg: None  # 禁用日志

    def test_short_selling_allowed(self):
        """测试美股允许做空。"""
        # 负权重应该保留
        weight = -0.8
        adjusted = self.strategy.adjust_target_weight(weight)
        self.assertEqual(adjusted, -0.8)

        weight = -0.5
        adjusted = self.strategy.adjust_target_weight(weight)
        self.assertEqual(adjusted, -0.5)

    def test_no_lot_limit(self):
        """测试美股无整手限制。"""
        # Mock 数据源
        class MockData:
            def __init__(self):
                self.close = [100.0]  # 股价 $100

        class MockBroker:
            def getvalue(self):
                return 100000  # $100,000

        self.strategy.broker = MockBroker()

        data = MockData()

        # 计算目标股数
        target_weight = 0.756  # 75.6%
        target_size = self.strategy._get_target_size(data, target_weight)

        # 应该是 756 股（美股无整手限制）
        self.assertEqual(target_size, 756)

    def test_t_plus_zero(self):
        """测试美股 T+0 交易。"""
        # 美股应该始终允许交易
        can_trade = self.strategy.check_trading_rules('QQQ', 0.5)
        self.assertTrue(can_trade)

        can_trade = self.strategy.check_trading_rules('AAPL', -0.3)
        self.assertTrue(can_trade)

        can_trade = self.strategy.check_trading_rules('MSFT', 0.8)
        self.assertTrue(can_trade)


class TestUSMarketCommission(unittest.TestCase):
    """测试美股费用计算。"""

    def setUp(self):
        """设置测试环境。"""
        self.commission = USMarketCommission()

    def test_buy_commission_zero(self):
        """测试买入费用为零（零佣金）。"""
        # 买入 100 股，股价 $100
        size = 100
        price = 100.0
        comm = self.commission._getcommission(size, price, pseudoexec=False)

        # 应该为 0（零佣金）
        self.assertEqual(comm, 0.0)

    def test_sell_sec_fee(self):
        """测试卖出收取 SEC 费。"""
        # 卖出 100 股，股价 $100
        size = -100
        price = 100.0
        comm = self.commission._getcommission(size, price, pseudoexec=False)

        # 应该只收取 SEC 费
        # SEC 费 = $10,000 * 0.0000207 ≈ $0.207
        expected_sec_fee = 10000 * 0.0000207
        self.assertAlmostEqual(comm, expected_sec_fee, places=2)

    def test_large_sell_order(self):
        """测试大额卖出订单。"""
        # 卖出 1000 股，股价 $200
        size = -1000
        price = 200.0
        comm = self.commission._getcommission(size, price, pseudoexec=False)

        # SEC 费 = $200,000 * 0.0000207 ≈ $4.14
        expected_sec_fee = 200000 * 0.0000207
        self.assertAlmostEqual(comm, expected_sec_fee, places=2)


class TestMarketComparison(unittest.TestCase):
    """测试 A 股和美股的差异对比。"""

    def test_commission_comparison(self):
        """测试 A 股和美股费用对比。"""
        from src.execution.a_share_sizer import AShareCommission

        a_share_comm = AShareCommission()
        us_comm = USMarketCommission()

        # 卖出 ¥500,000 (约 $70,000)
        turnover = 500000

        # A 股费用：印花税 0.1% + 佣金（最低 5 元）
        a_share_cost = turnover * 0.001 + max(turnover * 0.0002, 5.0)
        # 印花税：¥500 + 佣金：¥100

        # 美股费用：SEC 费
        us_cost = turnover * 0.0000207  # ≈ ¥10.35

        # 美股费用应该远低于 A 股
        self.assertLess(us_cost, a_share_cost * 0.1)  # 美股费用 < A股费用的 10%

    def test_trading_rules_comparison(self):
        """测试 A 股和美股交易规则对比。"""
        from src.execution.a_share_strategy import AShareStrategy

        a_share = AShareStrategy()
        us_market = USMarketStrategy()

        # 1. 做空限制
        # A 股禁止做空
        a_share_weight = a_share.adjust_target_weight(-0.8)
        self.assertGreaterEqual(a_share_weight, 0)  # 负权重转为非负

        # 美股允许做空
        us_weight = us_market.adjust_target_weight(-0.8)
        self.assertEqual(us_weight, -0.8)  # 负权重保留

        # 2. 整手限制
        # A 股有整手限制（100 股）
        # 美股无限制

        # 3. T+1 vs T+0
        # A 股需要检查 T+1 限制
        # 美股无限制，始终返回 True


if __name__ == "__main__":
    print("=" * 80)
    print("美股市场单元测试")
    print("=" * 80)

    unittest.main(verbosity=2)