"""
回测执行引擎模块。

将融合后的交易信号接入 Backtrader 回测框架。
"""

from .bt_engine import (
    DualTrackStrategy,
    PandasDataFeed,
    BacktestEngine,
    BacktestResult,
)

__all__ = [
    "DualTrackStrategy",
    "PandasDataFeed",
    "BacktestEngine",
    "BacktestResult",
]