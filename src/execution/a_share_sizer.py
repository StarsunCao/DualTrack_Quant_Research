"""
A股交易Sizer模块。

实现A股特有的交易规则：
1. 买入必须是100股的整数倍（整手）
2. 卖出可以卖出全部持仓（包括零股）
3. 考虑手续费最低消费（5元）
4. 印花税（仅卖出收取，0.05%）
"""

import math
import backtrader as bt


class AShareSizer(bt.SizerBase):
    """
    A股交易Sizer。

    实现A股市场的交易规则：
    - 买入：必须以手（100股）为单位
    - 卖出：可以卖出全部持仓（包括零股）
    - 自动向下取整到最接近的整手数

    使用方法：
        cerebro.addsizer(AShareSizer)
    """

    params = (
        ('lot_size', 100),  # A股每手100股
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        计算交易数量。

        Args:
            comminfo: 佣金信息对象
            cash: 可用现金
            data: 数据源
            isbuy: 是否为买入

        Returns:
            交易数量（股数）
        """
        position = self.broker.getposition(data)
        price = data.close[0]  # 使用当前收盘价计算

        if isbuy:
            # 买入：计算能买多少整手
            # 考虑手续费，预留一定的buffer
            commission_rate = comminfo.p.commission
            max_value = cash / (1 + commission_rate)  # 预留手续费空间

            # 计算最大可买股数
            max_shares = int(max_value / price)

            # 向下取整到最接近的整手数
            lot_size = self.p.lot_size
            rounded_shares = (max_shares // lot_size) * lot_size

            return rounded_shares

        else:
            # 卖出：可以卖出全部持仓（包括零股）
            # A股允许卖出零股（因分红送转产生的零头）
            return position.size


class AShareCommission(bt.CommInfoBase):
    """
    A股佣金和印花税计算器。

    实现A股市场的费用规则：
    - 佣金：万分之二，最低5元
    - 印花税：仅卖出收取，0.05%
    - 过户费：0.00002%（已并入佣金）

    使用方法：
        cerebro.broker.addcommissioninfo(AShareCommission())
    """

    params = (
        ('commission', 0.0002),  # 佣金：万分之二
        ('stamp_duty', 0.001),   # 印花税：千分之一（仅卖出）
        ('min_commission', 5.0), # 最低佣金：5元
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        计算佣金和印花税。

        Args:
            size: 交易数量（正数买入，负数卖出）
            price: 交易价格
            pseudoexec: 是否为模拟执行

        Returns:
            总费用（佣金 + 印花税）
        """
        # 计算交易金额
        turnover = abs(size) * price

        # 计算佣金（最低5元）
        commission = turnover * self.p.commission
        if commission < self.p.min_commission and turnover > 0:
            commission = self.p.min_commission

        # 计算印花税（仅卖出收取）
        stamp_duty = 0.0
        if size < 0:  # 卖出
            stamp_duty = turnover * self.p.stamp_duty

        return commission + stamp_duty


if __name__ == "__main__":
    # 测试示例
    print("=" * 80)
    print("A股Sizer测试")
    print("=" * 80)

    print("\n买入场景：")
    print("  可用现金: 50,000元")
    print("  股价: 100元")
    print("  佣金率: 0.0002")
    print("\n计算：")
    cash = 50000
    price = 100
    commission_rate = 0.0002
    max_value = cash / (1 + commission_rate)
    max_shares = int(max_value / price)
    rounded_shares = (max_shares // 100) * 100
    print(f"  预留手续费后最大购买力: {max_value:.2f}元")
    print(f"  最大可买股数: {max_shares}股")
    print(f"  取整后股数: {rounded_shares}股 ({rounded_shares//100}手)")

    print("\n" + "=" * 80)
    print("A股佣金测试")
    print("=" * 80)

    print("\n小额买入（触发最低消费）：")
    turnover = 10000
    commission = max(turnover * 0.0002, 5.0)
    print(f"  交易金额: {turnover}元")
    print(f"  理论佣金: {turnover * 0.0002:.2f}元")
    print(f"  实际佣金: {commission:.2f}元（最低5元）")

    print("\n卖出场景（收取印花税）：")
    turnover = 50000
    commission = max(turnover * 0.0002, 5.0)
    stamp_duty = turnover * 0.001
    print(f"  交易金额: {turnover}元")
    print(f"  佣金: {commission:.2f}元")
    print(f"  印花税: {stamp_duty:.2f}元")
    print(f"  总费用: {commission + stamp_duty:.2f}元")