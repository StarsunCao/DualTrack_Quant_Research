"""
美股交易Sizer和费用计算模块。

实现美股特有的交易规则：
1. 无整手限制（可以买卖零股）
2. SEC费（仅卖出收取，2024年标准：0.0000207）
3. 佣金（大部分券商已实现零佣金）
"""

import backtrader as bt


class USMarketSizer(bt.SizerBase):
    """
    美股交易Sizer。

    实现美股市场的交易规则：
    - 无整手限制，可以买卖任意股数（包括零股）
    - 自动计算最大可买/卖股数

    使用方法：
        cerebro.addsizer(USMarketSizer)
    """

    params = (
        ('lot_size', 0),  # 0表示无整手限制
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
            # 买入：计算能买多少股
            # 考虑佣金，预留一定的buffer
            commission_rate = comminfo.p.commission if hasattr(comminfo.p, 'commission') else 0.0

            # 预留手续费空间（如果有佣金的话）
            if commission_rate > 0:
                max_value = cash / (1 + commission_rate)
            else:
                max_value = cash

            # 计算最大可买股数（美股无整手限制）
            max_shares = int(max_value / price)

            return max_shares

        else:
            # 卖出：可以卖出全部持仓
            return position.size


class USMarketCommission(bt.CommInfoBase):
    """
    美股佣金和SEC费计算器。

    实现美股市场的费用规则：
    - 佣金：大部分券商已实现零佣金
    - SEC费：仅卖出收取，费率 0.0000207（2024年标准）
    - 无印花税

    使用方法：
        cerebro.broker.addcommissioninfo(USMarketCommission())
    """

    params = (
        ('commission', 0.0),  # 佣金：大部分券商零佣金
        ('sec_fee_rate', 0.0000207),  # SEC费率（2024年标准）
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_PERC),
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        计算佣金和SEC费。

        Args:
            size: 交易数量（正数买入，负数卖出）
            price: 交易价格
            pseudoexec: 是否为模拟执行

        Returns:
            总费用（佣金 + SEC费）
        """
        # 计算交易金额
        turnover = abs(size) * price

        # 计算佣金（大部分券商零佣金）
        commission = turnover * self.p.commission

        # 计算SEC费（仅卖出收取）
        sec_fee = 0.0
        if size < 0:  # 卖出
            sec_fee = turnover * self.p.sec_fee_rate

        return commission + sec_fee


if __name__ == "__main__":
    # 测试示例
    print("=" * 80)
    print("美股Sizer测试")
    print("=" * 80)

    print("\n买入场景：")
    print("  可用现金: 50,000美元")
    print("  股价: 100美元")
    print("  佣金率: 0.0（零佣金）")
    print("\n计算：")
    cash = 50000
    price = 100
    commission_rate = 0.0

    if commission_rate > 0:
        max_value = cash / (1 + commission_rate)
    else:
        max_value = cash

    max_shares = int(max_value / price)
    print(f"  最大购买力: {max_value:.2f}美元")
    print(f"  最大可买股数: {max_shares}股（无整手限制）")

    print("\n" + "=" * 80)
    print("美股佣金和SEC费测试")
    print("=" * 80)

    print("\n买入场景（零佣金）：")
    turnover = 50000
    commission = turnover * 0.0
    sec_fee = 0.0  # 买入无SEC费
    print(f"  交易金额: ${turnover:,.2f}")
    print(f"  佣金: ${commission:.2f}")
    print(f"  SEC费: ${sec_fee:.2f}")
    print(f"  总费用: ${commission + sec_fee:.2f}")

    print("\n卖出场景（收取SEC费）：")
    turnover = 50000
    commission = turnover * 0.0
    sec_fee_rate = 0.0000207
    sec_fee = turnover * sec_fee_rate
    print(f"  交易金额: ${turnover:,.2f}")
    print(f"  佣金: ${commission:.2f}")
    print(f"  SEC费: ${sec_fee:.4f}（费率 {sec_fee_rate}）")
    print(f"  总费用: ${commission + sec_fee:.4f}")

    print("\n对比A股和美股费用：")
    print("  A股（卖出）: 印花税 0.1% + 佣金（最低5元） = 较高")
    print("  美股（卖出）: SEC费 0.0000207 + 零佣金 = 极低")

    # 计算示例
    a_share_cost = 50000 * 0.001 + 5  # 印花税 + 最低佣金
    us_cost = 50000 * 0.0000207

    print(f"\n  卖出$50,000（¥360,000）:")
    print(f"    A股费用: ¥{a_share_cost:.2f}（约${a_share_cost/7.2:.2f}）")
    print(f"    美股费用: ${us_cost:.2f}")
    print(f"    美股费用仅为A股的 {us_cost/(a_share_cost/7.2)*100:.1f}%")