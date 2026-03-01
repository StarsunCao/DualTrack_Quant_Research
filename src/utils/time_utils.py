"""
时间处理工具模块。

提供统一的时间戳格式化和交易日对齐功能。
"""

import pandas as pd
from datetime import datetime
from typing import Union, Optional


def normalize_timestamp(
    ts: Union[str, datetime, pd.Timestamp],
    fmt: Optional[str] = None
) -> pd.Timestamp:
    """
    统一时间戳格式。

    Args:
        ts: 输入时间戳（字符串、datetime或Timestamp）
        fmt: 字符串格式（可选）

    Returns:
        标准化的 pandas Timestamp（去除时间部分）

    Examples:
        >>> normalize_timestamp("2024-01-15")
        Timestamp('2024-01-15 00:00:00')
        >>> normalize_timestamp("2024-01-15 14:30:00")
        Timestamp('2024-01-15 00:00:00')
    """
    if isinstance(ts, pd.Timestamp):
        return ts.normalize()  # 去除时间部分

    if isinstance(ts, str):
        # 尝试多种格式
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y/%m/%d",
        ]
        if fmt:
            formats.insert(0, fmt)

        for f in formats:
            try:
                return pd.to_datetime(ts, format=f).normalize()
            except:
                continue

        # 最后尝试pandas自动解析
        return pd.to_datetime(ts).normalize()

    if isinstance(ts, datetime):
        return pd.Timestamp(ts).normalize()

    raise ValueError(f"无法解析时间戳: {ts}")


def align_to_trading_days(
    signals: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    date_col: str = 'timestamp'
) -> pd.DataFrame:
    """
    将信号对齐到交易日。

    对于每个交易日，使用最近的前序信号（包括当日）。
    这是防止未来函数的关键：信号时间 <= 交易日

    Args:
        signals: 信号DataFrame，必须包含 date_col 列
        trading_days: 交易日索引
        date_col: 日期列名

    Returns:
        对齐后的信号DataFrame，日期与 trading_days 一致

    Examples:
        >>> signals = pd.DataFrame({
        ...     'timestamp': ['2024-01-10', '2024-01-12'],
        ...     'signal': [0.5, -0.3]
        ... })
        >>> trading_days = pd.DatetimeIndex(['2024-01-10', '2024-01-11', '2024-01-12'])
        >>> aligned = align_to_trading_days(signals, trading_days)
        # 2024-01-10: signal=0.5 (当日)
        # 2024-01-11: signal=0.5 (前序)
        # 2024-01-12: signal=-0.3 (当日)
    """
    if signals.empty:
        return pd.DataFrame()

    signals = signals.copy()

    # 标准化日期列
    signals[date_col] = pd.to_datetime(signals[date_col]).dt.normalize()

    # 标准化交易日
    trading_days = pd.to_datetime(trading_days).normalize().unique()

    # 按日期排序
    signals = signals.sort_values(date_col)

    # 创建交易日映射
    aligned_data = []

    for trade_day in trading_days:
        # 找到该交易日或之前的最新信号（关键：不晚于交易日，防止未来函数）
        past_signals = signals[signals[date_col] <= trade_day]

        if not past_signals.empty:
            # 使用最近日期的信号
            latest = past_signals.iloc[-1:].copy()
            latest[date_col] = trade_day
            aligned_data.append(latest)
        else:
            # 无信号时返回空（调用方应处理为 hold/0）
            pass

    if aligned_data:
        return pd.concat(aligned_data, ignore_index=True)
    else:
        return pd.DataFrame()


def fill_missing_trading_days(
    positions: dict,
    trading_days: pd.DatetimeIndex,
    default_weight: float = 0.0,
    symbol: str = "CSI300"
) -> dict:
    """
    填充缺失的交易日，确保每个交易日都有仓位信号。

    Args:
        positions: 原始仓位字典 {datetime: {symbol: weight}}
        trading_days: 交易日索引
        default_weight: 缺失日的默认权重（通常为 0 = hold）
        symbol: 交易标的

    Returns:
        完整的仓位字典，包含所有交易日
    """
    trading_days = pd.to_datetime(trading_days).normalize()

    filled_positions = {}
    last_valid_position = default_weight

    for trade_day in trading_days:
        # 查找该交易日的信号
        if trade_day in positions:
            weight = positions[trade_day].get(symbol, default_weight)
            last_valid_position = weight
            filled_positions[trade_day] = {symbol: weight}
        else:
            # 使用上一日信号或默认
            filled_positions[trade_day] = {symbol: last_valid_position}

    return filled_positions


def aggregate_daily_signals(
    signals: pd.DataFrame,
    date_col: str = 'timestamp',
    signal_col: str = 'llm_signal',
    confidence_col: str = 'confidence',
    symbol_col: str = 'symbol'
) -> pd.DataFrame:
    """
    将日内多条信号聚合成日频信号。

    用于 LLM Track：一天可能有多条新闻，需要聚合成单一交易信号。

    Args:
        signals: 信号DataFrame
        date_col: 日期列名
        signal_col: 信号列名（如 'llm_signal'）
        confidence_col: 置信度列名
        symbol_col: 标的列名

    Returns:
        日频信号DataFrame
    """
    if signals.empty:
        return pd.DataFrame()

    signals = signals.copy()
    signals[date_col] = pd.to_datetime(signals[date_col]).dt.normalize()

    daily_signals = []

    for date, group in signals.groupby(date_col):
        # 计算平均置信度
        avg_confidence = group[confidence_col].mean() if confidence_col in group.columns else 0.5

        # 信号投票（取多数）
        if signal_col in group.columns:
            signal_counts = group[signal_col].value_counts()
            dominant_signal = signal_counts.index[0]
        else:
            dominant_signal = 'hold'

        # 获取标的
        symbol = group[symbol_col].iloc[0] if symbol_col in group.columns else 'CSI300'

        daily_signals.append({
            'timestamp': date,
            'symbol': symbol,
            'signal': dominant_signal,
            'confidence': avg_confidence,
            'news_count': len(group)
        })

    return pd.DataFrame(daily_signals)
