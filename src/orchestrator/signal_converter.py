"""
信号转换器模块。

将 ML Track 或 LLM Track 的信号转换为目标仓位。
这是主实验使用的核心模块，用于独立运行各个轨道。

项目核心目标：对比 ML Track 和 LLM Track，回答"谁更好"。
此模块提供信号到仓位的转换功能，不进行融合。
"""

from datetime import datetime
from typing import Optional

import pandas as pd


class SignalConverter:
    """
    信号转换器。

    将 ML Track 或 LLM Track 的信号转换为目标仓位。
    不进行融合，只进行格式转换。

    这是主实验使用的核心类，用于独立运行各个轨道。
    """

    @staticmethod
    def ml_signals_to_positions(
        ml_signals: pd.DataFrame,
        ohlcv_dates: pd.DatetimeIndex = None
    ) -> dict:
        """
        将 ML 信号转换为目标仓位。

        Args:
            ml_signals: ML Track 信号 DataFrame
            ohlcv_dates: OHLCV数据的日期索引，用于对齐

        Returns:
            目标仓位字典 {datetime: {symbol: weight}}
        """
        positions = {}

        if ml_signals.empty:
            return positions

        # 确保timestamp列存在
        if 'timestamp' not in ml_signals.columns:
            # 尝试使用索引
            if isinstance(ml_signals.index, pd.DatetimeIndex):
                ml_signals = ml_signals.copy()
                ml_signals['timestamp'] = ml_signals.index
            else:
                return positions

        if "signal_strength_0_to_1" in ml_signals.columns:
            grouped = ml_signals.groupby("timestamp")
            for timestamp, group in grouped:
                avg_signal = group["signal_strength_0_to_1"].mean()
                weight = (avg_signal - 0.5) * 2  # 0-1 → -1到1
                symbol = group["symbol"].iloc[0] if "symbol" in group.columns else "CSI300"
                positions[pd.Timestamp(timestamp)] = {symbol: weight}

        # 如果提供了OHLCV日期，对齐到交易日
        if ohlcv_dates is not None and positions:
            from ..utils.time_utils import align_to_trading_days, fill_missing_trading_days

            # 转换为DataFrame进行对齐
            pos_df = pd.DataFrame([
                {'timestamp': ts, 'symbol': list(pos.keys())[0], 'weight': list(pos.values())[0]}
                for ts, pos in positions.items()
            ])

            aligned = align_to_trading_days(pos_df, ohlcv_dates)

            if not aligned.empty:
                positions = {
                    row['timestamp']: {row['symbol']: row['weight']}
                    for _, row in aligned.iterrows()
                }

            # 填充缺失交易日
            symbol = list(list(positions.values())[0].keys())[0] if positions else "CSI300"
            positions = fill_missing_trading_days(positions, ohlcv_dates, symbol=symbol)

        return positions

    @staticmethod
    def llm_signals_to_positions(
        llm_signals: pd.DataFrame,
        ohlcv_dates: pd.DatetimeIndex = None
    ) -> dict:
        """
        将 LLM 信号转换为目标仓位（支持交易日对齐，T-1信号做T决策）。

        关键修正：LLM信号的timestamp是决策日期（T），实际应该使用T-1的新闻和
        数据做出T日的决策。因此，信号timestamp对应的是T日的仓位。

        Args:
            llm_signals: LLM Track 信号 DataFrame
            ohlcv_dates: OHLCV数据的日期索引，用于对齐

        Returns:
            目标仓位字典 {datetime: {symbol: weight}}
        """
        positions = {}
        # 信号映射（语义解耦版本）
        # 注意：实际使用 signal_to_weight() 函数处理置信度
        signal_map = {
            "buy": 1.0,      # 建多仓
            "neutral": 0.0,  # 清仓/持有现金（防御）
            "short": -1.0,   # 主动做空（美股专用）
            # 旧信号类型（向后兼容）
            "sell": 0.0,     # 映射为清仓，避免过度做空
            "hold": None,    # 保持现状
        }

        if llm_signals.empty:
            return positions

        # 1. 确保timestamp列存在且为datetime类型
        if 'timestamp' not in llm_signals.columns:
            return positions

        llm_signals = llm_signals.copy()
        llm_signals['timestamp'] = pd.to_datetime(llm_signals['timestamp'])

        # 2. 按日期聚合多条新闻信号（取平均）
        from ..utils.time_utils import aggregate_daily_signals, align_to_trading_days, fill_missing_trading_days

        daily_df = aggregate_daily_signals(
            llm_signals,
            date_col='timestamp',
            signal_col='signal',  # LLM缓存中的列名是'signal'
            confidence_col='confidence',
            symbol_col='symbol'
        )

        if daily_df.empty:
            return positions

        # 3. 信号到仓位映射（语义解耦版本）
        # ====================================================================
        # 新信号类型（推荐）：
        # - BUY: weight = +confidence（按置信度建多仓）
        # - NEUTRAL: weight = 0（清仓/持有现金，防御性操作）
        # - SHORT: weight = -confidence（主动做空，美股专用）
        #
        # 旧信号类型（向后兼容）：
        # - HOLD: 不产生目标仓位（保持现状）
        # - SELL: 负权重（美股触发做空，A股转换为减仓）
        # ====================================================================

        def signal_to_weight(signal: str, confidence: float) -> Optional[float]:
            """
            信号到权重的映射函数（语义解耦版本）。

            核心改进：将"风险规避"（neutral）和"主动做空"（short）分离，
            避免 LLM 对数值阈值感知迟钝导致的语义混乱。

            Args:
                signal: 信号类型 (buy/neutral/short/hold/sell)
                confidence: 置信度 (0.0-1.0)

            Returns:
                目标权重（hold返回None，表示不调整仓位）
            """
            signal_lower = signal.lower().strip()

            # 新信号类型（语义解耦）
            if signal_lower == "buy":
                return confidence  # BUY: 正权重，按置信度建多仓
            elif signal_lower == "neutral":
                # NEUTRAL: 清仓/持有现金，防御性操作
                # 与 short 的区别：neutral 是风险规避，short 是主动做空
                return 0.0
            elif signal_lower == "short":
                # SHORT: 主动做空（仅美股有效）
                # 这是一种激进的看空赌注，仅在有系统性风险时使用
                return -confidence

            # 旧信号类型（向后兼容）
            elif signal_lower == "sell":
                # SELL: 旧版信号，保持兼容性
                # 映射为 0（清仓）而非 -confidence，避免过度做空
                # 注意：这是关键的语义修正！
                return 0.0
            else:  # hold 或其他
                return None  # HOLD: 不产生目标仓位，保持现状

        # 应用映射
        daily_df['weight'] = daily_df.apply(
            lambda row: signal_to_weight(row['signal'], row['confidence']),
            axis=1
        )

        # ====================================================================
        # 关键修正：时间对齐
        # LLM缓存中的timestamp是决策日期（T），即使用T-1数据做出T日决策
        # 因此，信号timestamp对应的权重应该应用于T日
        # 不需要额外调整，因为timestamp已经是决策日期
        # ====================================================================

        # 4. 如果提供了OHLCV日期，对齐到交易日
        if ohlcv_dates is not None:
            aligned = align_to_trading_days(daily_df, ohlcv_dates)

            if not aligned.empty:
                # 过滤掉HOLD信号（weight为NaN），只保留BUY和SELL信号
                # HOLD信号不产生目标仓位，回测引擎会自动保持当前持仓
                positions = {
                    row['timestamp']: {row['symbol']: row['weight']}
                    for _, row in aligned.iterrows()
                    if pd.notna(row['weight'])  # 使用pd.notna()过滤NaN（HOLD信号）
                }

            # ⚠️ 关键修改：不填充缺失交易日
            # 让回测引擎自己处理：如果某天没有目标仓位，就保持当前持仓
            # 这才是HOLD信号的正确含义："保持现状"
        else:
            # 不对齐，直接使用信号日期（决策日期）
            for _, row in daily_df.iterrows():
                if pd.notna(row['weight']):  # 使用pd.notna()过滤NaN（HOLD信号）
                    positions[row['timestamp']] = {row['symbol']: row['weight']}

        return positions