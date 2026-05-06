"""
信号转换器模块。

将 ML Track 或 LLM Track 的信号转换为目标仓位。
增加 EMA 平滑和动态衰减，减少信号跳跃导致的过度换手。

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
    包含 EMA 平滑和动态衰减，使仓位变化更渐进。
    """

    @staticmethod
    def _apply_ema_smoothing(
        daily_df: pd.DataFrame,
        symbol: str,
        ema_alpha: float = 0.50,
        decay_rate: float = 0.80,
        weak_buy_threshold: float = 0.55,
        weak_short_threshold: float = 0.60,
    ) -> pd.DataFrame:
        """
        对信号权重应用 EMA 平滑和动态衰减。

        - buy/short 信号：EMA 平滑 smoothed = alpha * raw + (1-alpha) * prev
        - hold/neutral 信号：维持当前仓位（不衰减）
        - sell 信号：清仓（动态衰减归零）
        - 当 abs(smoothed) < 0.01 时归零，避免无穷小尾部

        Args:
            daily_df: 包含 weight 列的信号 DataFrame，已按日期排序
            symbol: 资产代码
            ema_alpha: EMA 平滑系数 (0-1)，越大越灵敏
            decay_rate: 衰减速率 (0-1)，越小衰减越快
            weak_buy_threshold: weak buy 置信度阈值，低于此值视为观望
            weak_short_threshold: weak short 置信度阈值，低于此值视为清仓

        Returns:
            添加了 smoothed_weight 列的 DataFrame
        """
        daily_df = daily_df.copy()
        daily_df = daily_df.sort_values("timestamp").reset_index(drop=True)

        # 判断仓位意图
        # ML 信号可能只有 weight 列（无 signal/confidence），此时统一用 EMA 平滑
        has_signal_col = "signal" in daily_df.columns
        if has_signal_col:
            signal_col = daily_df["signal"].astype(str).str.lower()
            conf_col = daily_df.get("confidence", pd.Series([0.5] * len(daily_df)))

            # weak buy：信号是 buy 但 conf < threshold → 视为观望，维持当前仓位
            # weak short：信号是 short 但 conf < threshold → 视为"勉强做空，清仓观望"
            is_weak_buy = (signal_col == "buy") & (conf_col < weak_buy_threshold)
            is_weak_short = (signal_col == "short") & (conf_col < weak_short_threshold)
            is_hold = signal_col.isin(["hold", "neutral"]) | is_weak_buy  # 观望
            is_sell = signal_col.isin(["sell"]) | is_weak_short  # 清仓
        else:
            # ML 信号（只有 weight 列）：正常 buy/short 用 EMA，零值用衰减
            is_hold = daily_df["weight"] == 0.0
            is_sell = pd.Series([False] * len(daily_df))

        smoothed = []
        prev_weight = 0.0

        for i, (_, row) in enumerate(daily_df.iterrows()):
            raw_weight = row.get("weight", 0.0)
            if pd.isna(raw_weight):
                raw_weight = 0.0

            if is_hold.iloc[i]:
                # hold/neutral：维持当前仓位不变
                new_weight = prev_weight
            elif is_sell.iloc[i]:
                # sell 或 weak buy/short：动态衰减归零
                new_weight = prev_weight * decay_rate
            else:
                # 正常 buy/short：EMA 平滑
                new_weight = ema_alpha * raw_weight + (1 - ema_alpha) * prev_weight

            # 截断到 [-1, 1]
            new_weight = max(-1.0, min(1.0, new_weight))

            # 微小值归零
            if abs(new_weight) < 0.01:
                new_weight = 0.0

            smoothed.append(new_weight)
            prev_weight = new_weight

        daily_df["smoothed_weight"] = smoothed
        return daily_df

    @staticmethod
    def _compute_turnover(
        positions: dict,
        ohlcv_dates: pd.DatetimeIndex,
        symbol: str = "CSI300",
    ) -> float:
        """
        计算平均日换手率。

        Args:
            positions: 目标仓位字典 {datetime: {symbol: weight}}
            ohlcv_dates: OHLCV 日期索引
            symbol: 资产代码

        Returns:
            平均日换手率（相邻日仓位差的绝对值之和 / 总天数）
        """
        if len(ohlcv_dates) < 2:
            return 0.0

        weights = []
        for dt in ohlcv_dates:
            w = positions.get(dt, {}).get(symbol, 0.0)
            weights.append(w)

        turnovers = [abs(weights[i] - weights[i - 1]) for i in range(1, len(weights))]
        return sum(turnovers) / (len(weights) - 1)

    @staticmethod
    def ml_signals_to_positions(
        ml_signals: pd.DataFrame,
        ohlcv_dates: pd.DatetimeIndex = None,
        ema_alpha: float = 0.50,
        decay_rate: float = 0.80,
        signal_threshold: float = 0.55,
    ) -> dict:
        """
        将 ML 信号转换为目标仓位。

        Args:
            ml_signals: ML Track 信号 DataFrame
            ohlcv_dates: OHLCV数据的日期索引，用于对齐
            ema_alpha: EMA 平滑系数
            decay_rate: 动态衰减速率
            signal_threshold: 信号阈值，只有 signal_strength > threshold 或
                < (1-threshold) 才触发仓位变化，中间视为 neutral

        Returns:
            目标仓位字典 {datetime: {symbol: weight}}
        """
        positions = {}

        if ml_signals.empty:
            return positions

        # 确保timestamp列存在
        if 'timestamp' not in ml_signals.columns:
            if isinstance(ml_signals.index, pd.DatetimeIndex):
                ml_signals = ml_signals.copy()
                ml_signals['timestamp'] = ml_signals.index
            else:
                return positions

        if "signal_strength_0_to_1" in ml_signals.columns:
            grouped = ml_signals.groupby("timestamp")
            for timestamp, group in grouped:
                avg_signal = group["signal_strength_0_to_1"].mean()
                # 阈值过滤：弱信号视为 neutral（维持仓位）
                if (1 - signal_threshold) < avg_signal < signal_threshold:
                    weight = 0.0  # neutral，EMA 会维持当前仓位
                else:
                    weight = (avg_signal - 0.5) * 2  # 0-1 → -1到1
                symbol = group["symbol"].iloc[0] if "symbol" in group.columns else "CSI300"
                positions[pd.Timestamp(timestamp)] = {symbol: weight}

            # EMA 平滑
            pos_list = [{"timestamp": ts, "symbol": list(pos.keys())[0], "weight": list(pos.values())[0]}
                        for ts, pos in positions.items()]
            if pos_list:
                daily_df = pd.DataFrame(pos_list)
                symbol = pos_list[0]["symbol"]
                daily_df = SignalConverter._apply_ema_smoothing(
                    daily_df, symbol, ema_alpha, decay_rate
                )
                positions = {
                    pd.Timestamp(row["timestamp"]): {row["symbol"]: row["smoothed_weight"]}
                    for _, row in daily_df.iterrows()
                }

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
        ohlcv_dates: pd.DatetimeIndex = None,
        confidence_mode: str = "linear",
        ema_alpha: float = 0.50,
        decay_rate: float = 0.80,
    ) -> dict:
        """
        将 LLM 信号转换为目标仓位（支持交易日对齐，T-1信号做T决策）。

        关键修正：LLM信号的timestamp是决策日期（T），实际应该使用T-1的新闻和
        数据做出T日的决策。因此，信号timestamp对应的是T日的仓位。

        Args:
            llm_signals: LLM Track 信号 DataFrame
            ohlcv_dates: OHLCV数据的日期索引，用于对齐
            confidence_mode: 置信度映射模式
                - "linear": (C-0.5)*2，0.55→10%, 0.90→80% (新映射)
                - "direct": C，0.55→55%, 0.90→90% (旧映射)
            ema_alpha: EMA 平滑系数
            decay_rate: 动态衰减速率

        Returns:
            目标仓位字典 {datetime: {symbol: weight}}
        """
        positions = {}
        signal_map = {
            "buy": 1.0,
            "neutral": 0.0,
            "short": -1.0,
            "sell": 0.0,
            "hold": None,
        }

        if llm_signals.empty:
            return positions

        if 'timestamp' not in llm_signals.columns:
            return positions

        llm_signals = llm_signals.copy()
        llm_signals['timestamp'] = pd.to_datetime(llm_signals['timestamp'])

        # 按日期聚合多条新闻信号
        from ..utils.time_utils import aggregate_daily_signals, align_to_trading_days, fill_missing_trading_days

        daily_df = aggregate_daily_signals(
            llm_signals,
            date_col='timestamp',
            signal_col='signal',
            confidence_col='confidence',
            symbol_col='symbol'
        )

        if daily_df.empty:
            return positions

        def signal_to_weight(signal: str, confidence: float) -> Optional[float]:
            signal_lower = signal.lower().strip()

            if signal_lower == "buy":
                if confidence < 0.5:
                    # buy 但确信度低 → 维持仓位（视为 hold）
                    return 0.0
                if confidence_mode == "linear":
                    return (confidence - 0.5) * 2
                else:
                    return confidence
            elif signal_lower == "neutral":
                return 0.0
            elif signal_lower == "short":
                if confidence < 0.5:
                    # short 但确信度低 → 维持仓位（视为 hold）
                    return 0.0
                if confidence_mode == "linear":
                    return -(confidence - 0.5) * 2
                else:
                    return -confidence
            elif signal_lower == "sell":
                return 0.0
            else:
                return None

        daily_df['weight'] = daily_df.apply(
            lambda row: signal_to_weight(row['signal'], row['confidence']),
            axis=1
        )

        # EMA 平滑 + 动态衰减
        symbol = daily_df['symbol'].iloc[0] if 'symbol' in daily_df.columns else "CSI300"
        from ..config.market_config import MarketConfig
        market_config = MarketConfig.get_config_for_symbol(symbol)
        daily_df = SignalConverter._apply_ema_smoothing(
            daily_df, symbol,
            ema_alpha=market_config.ema_alpha,
            decay_rate=market_config.decay_rate,
            weak_buy_threshold=market_config.weak_buy_threshold,
            weak_short_threshold=market_config.weak_short_threshold,
        )
        # 用 smoothed_weight 覆盖 weight
        daily_df['weight'] = daily_df['smoothed_weight']

        # 时间对齐
        if ohlcv_dates is not None:
            aligned = align_to_trading_days(daily_df, ohlcv_dates)

            if not aligned.empty:
                positions = {
                    row['timestamp']: {row['symbol']: row['weight']}
                    for _, row in aligned.iterrows()
                    if pd.notna(row['weight']) and abs(row['weight']) >= 0.01
                }
        else:
            for _, row in daily_df.iterrows():
                if pd.notna(row['weight']) and abs(row['weight']) >= 0.01:
                    positions[row['timestamp']] = {row['symbol']: row['weight']}

        return positions
