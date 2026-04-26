"""
市场配置系统。

定义不同市场的交易规则、费用结构、交易日历等配置。
支持 A 股和美股市场的差异化配置。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarketType(Enum):
    """市场类型枚举。"""
    A_SHARE = "a_share"  # A股市场
    US_MARKET = "us_market"  # 美股市场


@dataclass
class MarketConfig:
    """
    市场配置数据类。

    Attributes:
        market_type: 市场类型（A股/美股）
        allow_short_selling: 是否允许做空
        t_plus_one: 是否实行 T+1 制度
        lot_size: 每手股数（0 表示无限制）
        stamp_duty: 印花税率（仅卖出收取）
        min_commission: 最低佣金
        commission_rate: 佣金率
        sec_fee_rate: SEC 费率（美股专用，仅卖出收取）
        trading_calendar: 交易日历标识
        timezone: 市场时区
        settlement_days: 结算天数
        ema_alpha: EMA 平滑系数（信号→仓位转换时使用）
        decay_rate: 中性信号衰减速率（hold/neutral 时逐步减仓）
    """
    market_type: MarketType
    allow_short_selling: bool
    t_plus_one: bool
    lot_size: int  # 0 表示无限制
    stamp_duty: float
    min_commission: float
    commission_rate: float
    sec_fee_rate: float = 0.0  # 美股专用
    ema_alpha: float = 0.50  # EMA 平滑系数
    decay_rate: float = 0.80  # 中性信号衰减速率
    trading_calendar: str = "SSE"  # 'SSE' (上交所) 或 'NASDAQ'
    timezone: str = "Asia/Shanghai"
    settlement_days: int = 1  # T+1 结算

    @classmethod
    def a_share(cls) -> "MarketConfig":
        """
        创建 A 股市场配置。

        Returns:
            A 股市场配置实例
        """
        return cls(
            market_type=MarketType.A_SHARE,
            allow_short_selling=False,  # 禁止做空
            t_plus_one=True,  # T+1 制度
            lot_size=100,  # 每手 100 股
            stamp_duty=0.001,  # 印花税 0.1%（千分之一）
            min_commission=5.0,  # 最低佣金 5 元
            commission_rate=0.0002,  # 佣金万分之二
            ema_alpha=0.30,  # 强 EMA 平滑
            decay_rate=0.70,  # 快速衰减
            trading_calendar="SSE",  # 上交所交易日历
            timezone="Asia/Shanghai",
            settlement_days=1,
        )

    @classmethod
    def us_market(cls) -> "MarketConfig":
        """
        创建美股市场配置。

        Returns:
            美股市场配置实例
        """
        return cls(
            market_type=MarketType.US_MARKET,
            allow_short_selling=True,  # 允许做空
            t_plus_one=False,  # T+0 制度
            lot_size=0,  # 无整手限制
            stamp_duty=0.0,  # 无印花税
            min_commission=0.0,  # 无最低佣金
            commission_rate=0.0,  # 大部分券商零佣金
            sec_fee_rate=0.0000207,  # SEC 费率（2024年标准）
            ema_alpha=0.50,  # 中等 EMA 平滑
            decay_rate=0.80,  # 缓慢衰减
            trading_calendar="NASDAQ",  # 纳斯达克交易日历
            timezone="America/New_York",
            settlement_days=1,
        )

    def get_market_type_for_symbol(symbol: str) -> MarketType:
        """
        根据股票代码判断市场类型。

        Args:
            symbol: 股票代码

        Returns:
            市场类型枚举值
        """
        # 美股代码列表
        us_symbols = ['QQQ', 'NASDAQ100', 'SPY', 'SPX', 'AAPL', 'GOOGL', 'MSFT']

        if symbol.upper() in us_symbols:
            return MarketType.US_MARKET
        else:
            return MarketType.A_SHARE

    @classmethod
    def get_config_for_symbol(cls, symbol: str) -> "MarketConfig":
        """
        根据股票代码获取市场配置。

        Args:
            symbol: 股票代码

        Returns:
            对应的市场配置实例
        """
        market_type = cls.get_market_type_for_symbol(symbol)

        if market_type == MarketType.US_MARKET:
            return cls.us_market()
        else:
            return cls.a_share()


if __name__ == "__main__":
    # 测试市场配置
    print("=" * 80)
    print("市场配置测试")
    print("=" * 80)

    # A股配置
    a_share_config = MarketConfig.a_share()
    print("\nA股市场配置：")
    print(f"  市场类型: {a_share_config.market_type.value}")
    print(f"  允许做空: {a_share_config.allow_short_selling}")
    print(f"  T+1制度: {a_share_config.t_plus_one}")
    print(f"  整手限制: {a_share_config.lot_size}股/手")
    print(f"  印花税率: {a_share_config.stamp_duty:.4f}")
    print(f"  最低佣金: {a_share_config.min_commission}元")
    print(f"  佣金率: {a_share_config.commission_rate:.4f}")
    print(f"  交易日历: {a_share_config.trading_calendar}")
    print(f"  时区: {a_share_config.timezone}")

    # 美股配置
    us_config = MarketConfig.us_market()
    print("\n美股市场配置：")
    print(f"  市场类型: {us_config.market_type.value}")
    print(f"  允许做空: {us_config.allow_short_selling}")
    print(f"  T+1制度: {us_config.t_plus_one}")
    print(f"  整手限制: {'无限制' if us_config.lot_size == 0 else f'{us_config.lot_size}股/手'}")
    print(f"  印花税率: {us_config.stamp_duty:.4f}")
    print(f"  最低佣金: {us_config.min_commission}元")
    print(f"  佣金率: {us_config.commission_rate:.4f}")
    print(f"  SEC费率: {us_config.sec_fee_rate:.7f}")
    print(f"  交易日历: {us_config.trading_calendar}")
    print(f"  时区: {us_config.timezone}")

    # 根据代码自动判断
    print("\n自动判断市场：")
    for symbol in ['CSI300', 'QQQ', 'SPY', 'NASDAQ100']:
        config = MarketConfig.get_config_for_symbol(symbol)
        print(f"  {symbol}: {config.market_type.value}")