"""
信号融合引擎模块（可选探索）。

⚠️ 注意：此模块不是项目的核心目标！

项目核心：对比 ML Track 和 LLM Track，回答"谁更好"。
此模块提供的融合功能仅作为附加研究，不应在主实验中使用。

适用场景：
  - 探索性研究：融合是否能带来额外收益？
  - 实际应用：如果实验证明两个轨道互补，可考虑融合。

使用建议：
  - 主实验中禁用此模块
  - 在 Exp-C（可选实验）中探索融合效果
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """市场状态枚举。"""
    NORMAL = "normal"  # 正常波动率
    HIGH_VOLATILITY = "high_volatility"  # 高波动率
    BLACK_SWAN = "black_swan"  # 黑天鹅事件


class SignalSource(Enum):
    """信号来源枚举。"""
    ML_TRACK = "ml_track"
    LLM_TRACK = "llm_track"
    FUSED = "fused"


class SignalConverter:
    """
    信号转换器。

    将 ML Track 或 LLM Track 的信号转换为目标仓位。
    不进行融合，只进行格式转换。

    这是主实验使用的核心类，用于独立运行两个轨道。
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
        from datetime import datetime
        import pandas as pd

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
        from datetime import datetime
        import pandas as pd
        from ..utils.time_utils import aggregate_daily_signals, align_to_trading_days, fill_missing_trading_days

        positions = {}
        signal_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}

        if llm_signals.empty:
            return positions

        # 1. 确保timestamp列存在且为datetime类型
        if 'timestamp' not in llm_signals.columns:
            return positions

        llm_signals = llm_signals.copy()
        llm_signals['timestamp'] = pd.to_datetime(llm_signals['timestamp'])

        # 2. 按日期聚合多条新闻信号（取平均）
        daily_df = aggregate_daily_signals(
            llm_signals,
            date_col='timestamp',
            signal_col='llm_signal',
            confidence_col='confidence',
            symbol_col='symbol'
        )

        if daily_df.empty:
            return positions

        # 3. 转换为权重
        daily_df['weight'] = daily_df['signal'].map(signal_map) * daily_df['confidence']

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
                positions = {
                    row['timestamp']: {row['symbol']: row['weight']}
                    for _, row in aligned.iterrows()
                }

            # 填充缺失交易日（使用前向填充）
            symbol = daily_df['symbol'].iloc[0] if 'symbol' in daily_df.columns else "CSI300"
            positions = fill_missing_trading_days(positions, ohlcv_dates, default_weight=0.0, symbol=symbol)
        else:
            # 不对齐，直接使用信号日期（决策日期）
            for _, row in daily_df.iterrows():
                positions[row['timestamp']] = {row['symbol']: row['weight']}

        return positions


@dataclass
class LatencyMetrics:
    """
    延迟度量数据类。

    记录系统各环节的延迟。
    """
    ml_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    fusion_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "ml_latency_ms": self.ml_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "fusion_latency_ms": self.fusion_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass
class TargetPosition:
    """
    目标仓位数据类。

    表示最终发送给执行引擎的统一目标仓位。
    """
    symbol: str
    weight: float  # -1 到 1，负数表示做空
    signal_source: str
    confidence: float
    reasoning: str
    timestamp: datetime
    latency_metrics: LatencyMetrics
    market_regime: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "symbol": self.symbol,
            "weight": self.weight,
            "signal_source": self.signal_source,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "market_regime": self.market_regime,
            "latency": self.latency_metrics.to_dict(),
            "metadata": self.metadata,
        }


@dataclass
class FusedSignal:
    """
    融合信号数据类。

    表示 ML 和 LLM 轨道融合后的信号。
    """
    symbol: str
    ml_signal: float  # ML 轨道信号 (-1 到 1)
    llm_signal: float  # LLM 轨道信号 (-1 到 1)
    ml_confidence: float
    llm_confidence: float
    ml_latency_ms: float
    llm_latency_ms: float
    fused_weight: float
    fusion_source: str
    market_regime: str
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "symbol": self.symbol,
            "ml_signal": self.ml_signal,
            "llm_signal": self.llm_signal,
            "ml_confidence": self.ml_confidence,
            "llm_confidence": self.llm_confidence,
            "ml_latency_ms": self.ml_latency_ms,
            "llm_latency_ms": self.llm_latency_ms,
            "fused_weight": self.fused_weight,
            "fusion_source": self.fusion_source,
            "market_regime": self.market_regime,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


class SignalFusionEngine:
    """
    信号融合引擎。

    订阅 ML Track 和 LLM Track 的信号，根据市场状态动态融合。

    融合规则：
    - 正常波动率期间：ML Track 信号为主导
    - 高波动率期间：增加 LLM Track 权重
    - 黑天鹅事件：LLM Agent 具有一票否决权或强制清仓权

    优化特性：
    - 调仓死区（Rebalancing Dead Zone）：避免微小调仓带来的手续费磨损
    - LLM 信号衰减：超过一定时间无新新闻，LLM 信号权重逐渐衰减

    Attributes:
        volatility_threshold: 高波动率阈值。
        llm_veto_threshold: LLM 否决阈值。
        ml_weight_normal: 正常时期 ML 权重。
        ml_weight_high_vol: 高波动时期 ML 权重。
        rebalance_threshold: 调仓死区阈值（默认 5%）。
        llm_signal_decay_hours: LLM 信号衰减小时数（默认 72 小时 = 3 天）。
        decay_curve: 衰减曲线类型 ('linear' 或 'exponential')。
    """

    def __init__(
        self,
        volatility_threshold: float = 0.03,
        llm_veto_threshold: float = 0.8,
        llm_force_sell_threshold: float = 0.9,
        ml_weight_normal: float = 0.7,
        ml_weight_high_vol: float = 0.4,
        ml_latency_threshold_ms: float = 10.0,
        llm_latency_threshold_ms: float = 2000.0,
        rebalance_threshold: float = 0.05,
        llm_signal_decay_hours: int = 72,
        decay_curve: str = "linear",
    ) -> None:
        """
        初始化信号融合引擎。

        Args:
            volatility_threshold: 高波动率阈值（日波动率），默认 3%。
            llm_veto_threshold: LLM 否决阈值（确信度），默认 0.8。
            llm_force_sell_threshold: LLM 强制清仓阈值（确信度），默认 0.9。
            ml_weight_normal: 正常时期 ML 权重，默认 0.7。
            ml_weight_high_vol: 高波动时期 ML 权重，默认 0.4。
            ml_latency_threshold_ms: ML 延迟阈值（毫秒），默认 10ms。
            llm_latency_threshold_ms: LLM 延迟阈值（毫秒），默认 2000ms。
            rebalance_threshold: 调仓死区阈值，默认 0.05（5%）。
            llm_signal_decay_hours: LLM 信号衰减小时数，默认 72 小时（3 天）。
            decay_curve: 衰减曲线类型，可选 'linear' 或 'exponential'，默认 'linear'。
        """
        self.volatility_threshold = volatility_threshold
        self.llm_veto_threshold = llm_veto_threshold
        self.llm_force_sell_threshold = llm_force_sell_threshold
        self.ml_weight_normal = ml_weight_normal
        self.ml_weight_high_vol = ml_weight_high_vol
        self.ml_latency_threshold_ms = ml_latency_threshold_ms
        self.llm_latency_threshold_ms = llm_latency_threshold_ms
        self.rebalance_threshold = rebalance_threshold
        self.llm_signal_decay_hours = llm_signal_decay_hours
        self.decay_curve = decay_curve

        # 状态变量
        self._current_regime = MarketRegime.NORMAL
        self._last_update_time: Optional[datetime] = None
        self._signal_history: list[FusedSignal] = []
        self._latency_history: list[LatencyMetrics] = []

        # LLM 信号衰减追踪
        self._last_llm_signal_time: Optional[datetime] = None
        self._last_llm_signals: dict[str, dict] = {}  # 缓存上次的 LLM 信号

        # 上一次的目标仓位（用于调仓死区判断）
        self._last_target_positions: dict[str, float] = {}

    def detect_market_regime(
        self,
        volatility: float,
        has_major_news: bool = False,
        llm_confidence: float = 0.0,
        llm_signal: float = 0.0,
    ) -> MarketRegime:
        """
        检测市场状态。

        Args:
            volatility: 当前波动率（日波动率）。
            has_major_news: 是否有重大新闻输入。
            llm_confidence: LLM 确信度。
            llm_signal: LLM 信号。

        Returns:
            市场状态枚举值。
        """
        # 黑天鹅检测：高确信度的极端信号
        if has_major_news and llm_confidence >= self.llm_force_sell_threshold:
            if llm_signal <= -0.8:  # 强烈卖出信号
                return MarketRegime.BLACK_SWAN

        # 高波动率检测
        if volatility >= self.volatility_threshold:
            return MarketRegime.HIGH_VOLATILITY

        # 重大新闻但未达到黑天鹅级别
        if has_major_news and llm_confidence >= self.llm_veto_threshold:
            return MarketRegime.HIGH_VOLATILITY

        return MarketRegime.NORMAL

    def _signal_to_weight(self, signal: str, confidence: float) -> float:
        """
        将信号转换为权重。

        Args:
            signal: 信号类型 ('buy', 'sell', 'hold')。
            confidence: 确信度。

        Returns:
            权重值 (-1 到 1)。
        """
        signal_map = {
            "buy": 1.0,
            "sell": -1.0,
            "hold": 0.0,
        }
        return signal_map.get(signal.lower(), 0.0) * confidence

    def _numeric_signal_to_weight(self, signal: float) -> float:
        """
        将数值信号标准化为权重。

        Args:
            signal: 数值信号 (-1 到 1)。

        Returns:
            权重值 (-1 到 1)。
        """
        return max(-1.0, min(1.0, float(signal)))

    def fuse_signals(
        self,
        ml_signals: pd.DataFrame,
        llm_signals: pd.DataFrame,
        volatility: float = 0.02,
        has_major_news: bool = False,
        current_time: Optional[datetime] = None,
    ) -> list[FusedSignal]:
        """
        融合 ML Track 和 LLM Track 的信号。

        Args:
            ml_signals: ML Track 信号 DataFrame，需包含列：
                       - symbol: 资产代码
                       - model_name: 模型名称
                       - signal_strength_0_to_1: 信号强度
                       - 可选 latency_ms: 延迟
            llm_signals: LLM Track 信号 DataFrame，需包含列：
                        - symbol: 资产代码
                        - llm_signal: 信号 ('buy', 'sell', 'hold')
                        - confidence: 确信度
                        - reasoning: 推理过程
                        - latency_ms: 延迟
            volatility: 当前波动率。
            has_major_news: 是否有重大新闻输入。
            current_time: 当前时间（用于 LLM 信号衰减计算）。

        Returns:
            融合信号列表。
        """
        fusion_start = time.time()
        fused_signals: list[FusedSignal] = []
        timestamp = current_time or datetime.now()

        # 按 symbol 分组 ML 信号
        ml_grouped = self._group_ml_signals(ml_signals)

        # 按 symbol 分组 LLM 信号（包含衰减逻辑）
        llm_grouped = self._group_llm_signals(llm_signals, current_time=timestamp)

        # 获取所有唯一的 symbols
        all_symbols = set(ml_grouped.keys()) | set(llm_grouped.keys())

        for symbol in all_symbols:
            # 获取 ML 信号
            ml_data = ml_grouped.get(symbol, {})
            ml_signal = ml_data.get("signal", 0.0)
            ml_confidence = ml_data.get("confidence", 0.5)
            ml_latency = ml_data.get("latency_ms", 0.0)

            # 获取 LLM 信号
            llm_data = llm_grouped.get(symbol, {})
            llm_signal = llm_data.get("signal", 0.0)
            llm_confidence = llm_data.get("confidence", 0.0)
            llm_latency = llm_data.get("latency_ms", 0.0)
            llm_reasoning = llm_data.get("reasoning", "")

            # 检测市场状态
            regime = self.detect_market_regime(
                volatility=volatility,
                has_major_news=has_major_news,
                llm_confidence=llm_confidence,
                llm_signal=llm_signal,
            )

            # 执行融合
            fused_weight, fusion_source, reasoning = self._perform_fusion(
                ml_signal=ml_signal,
                ml_confidence=ml_confidence,
                llm_signal=llm_signal,
                llm_confidence=llm_confidence,
                llm_reasoning=llm_reasoning,
                regime=regime,
            )

            # 创建融合信号
            fused = FusedSignal(
                symbol=symbol,
                ml_signal=ml_signal,
                llm_signal=llm_signal,
                ml_confidence=ml_confidence,
                llm_confidence=llm_confidence,
                ml_latency_ms=ml_latency,
                llm_latency_ms=llm_latency,
                fused_weight=fused_weight,
                fusion_source=fusion_source,
                market_regime=regime.value,
                reasoning=reasoning,
                timestamp=timestamp,
            )
            fused_signals.append(fused)

        # 记录延迟
        fusion_latency = (time.time() - fusion_start) * 1000

        # 更新状态
        self._current_regime = self._detect_overall_regime(fused_signals)
        self._last_update_time = timestamp
        self._signal_history.extend(fused_signals)

        return fused_signals

    def _group_ml_signals(self, ml_signals: pd.DataFrame) -> dict[str, dict]:
        """
        分组 ML 信号。

        Args:
            ml_signals: ML Track 信号 DataFrame。

        Returns:
            按 symbol 分组的信号字典。
        """
        result: dict[str, dict] = {}

        if ml_signals.empty:
            return result

        # 检查必需列
        if "symbol" not in ml_signals.columns:
            return result

        for symbol, group in ml_signals.groupby("symbol"):
            # 获取信号强度
            if "signal_strength_0_to_1" in group.columns:
                # 将 0-1 转换为 -1 到 1
                signal = group["signal_strength_0_to_1"].mean() * 2 - 1
            else:
                signal = 0.0

            # 获取延迟
            latency = group.get("latency_ms", pd.Series([0.0])).mean()

            result[symbol] = {
                "signal": self._numeric_signal_to_weight(signal),
                "confidence": abs(signal),  # 使用信号强度作为置信度
                "latency_ms": latency if not pd.isna(latency) else 0.0,
            }

        return result

    def _group_llm_signals(
        self,
        llm_signals: pd.DataFrame,
        current_time: Optional[datetime] = None,
    ) -> dict[str, dict]:
        """
        分组 LLM 信号，并应用信号衰减逻辑。

        当超过 llm_signal_decay_hours 小时无新新闻时，LLM 信号权重逐渐衰减。

        Args:
            llm_signals: LLM Track 信号 DataFrame。
            current_time: 当前时间（用于测试）。

        Returns:
            按 symbol 分组的信号字典。
        """
        result: dict[str, dict] = {}

        if llm_signals.empty:
            # 没有新的 LLM 信号，检查是否需要应用衰减
            if self._last_llm_signals and current_time and self._last_llm_signal_time:
                # 计算小时级别的时间差
                hours_since_last = (current_time - self._last_llm_signal_time).total_seconds() / 3600

                if hours_since_last > self.llm_signal_decay_hours:
                    # 应用衰减
                    if self.decay_curve == "linear":
                        # 线性衰减：每超时 24 小时衰减 30%
                        decay_factor = max(0.0, 1.0 - (hours_since_last - self.llm_signal_decay_hours) / 24 * 0.3)
                    elif self.decay_curve == "exponential":
                        # 指数衰减：更平滑
                        import math
                        decay_factor = max(0.0, math.exp(-(hours_since_last - self.llm_signal_decay_hours) / 24))
                    else:
                        decay_factor = 1.0

                    for symbol, data in self._last_llm_signals.items():
                        result[symbol] = {
                            "signal": data["signal"] * decay_factor,
                            "confidence": data["confidence"] * decay_factor,
                            "latency_ms": 0.0,
                            "reasoning": f"[衰减 {decay_factor:.0%}] {data.get('reasoning', '')}",
                        }
            return result

        # 检查必需列
        if "symbol" not in llm_signals.columns:
            return result

        # 更新最后 LLM 信号时间
        # 关键改进：优先使用 timestamp 列，支持多种格式
        if "timestamp" in llm_signals.columns:
            # 找到最新的信号时间戳
            try:
                # 尝试多种时间戳格式
                timestamps = []
                for ts in llm_signals["timestamp"]:
                    if pd.isna(ts):
                        continue
                    # 尝试解析不同格式的时间戳
                    parsed = False
                    for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"]:
                        try:
                            if isinstance(ts, str):
                                parsed_ts = datetime.strptime(ts, fmt)
                            else:
                                parsed_ts = pd.to_datetime(ts)
                            timestamps.append(parsed_ts)
                            parsed = True
                            break
                        except (ValueError, TypeError):
                            continue

                    if not parsed:
                        print(f"⚠️ 无法解析时间戳: {ts}")

                if timestamps:
                    latest_timestamp = max(timestamps)
                    self._last_llm_signal_time = latest_timestamp
            except Exception as e:
                print(f"⚠️ 时间戳解析失败: {e}")
                self._last_llm_signal_time = current_time or datetime.now()
        else:
            self._last_llm_signal_time = current_time or datetime.now()

        for symbol, group in llm_signals.groupby("symbol"):
            # 关键改进：按时间戳排序，取最新信号
            if "timestamp" in group.columns:
                group = group.sort_values("timestamp", ascending=False)
                latest = group.iloc[0] if len(group) > 0 else None
            else:
                latest = group.iloc[0] if len(group) > 0 else None

            if latest is None:
                continue

            # 获取信号
            if "llm_signal" in group.columns:
                signal_str = latest.get("llm_signal", "hold")
                if "confidence" in group.columns:
                    confidence = latest.get("confidence", 0.5)
                    if pd.isna(confidence):
                        confidence = 0.5
                else:
                    confidence = 0.5
                signal = self._signal_to_weight(signal_str, confidence)
            else:
                signal = 0.0
                confidence = 0.0

            # 获取延迟
            latency = latest.get("latency_ms", 0.0)
            if pd.isna(latency):
                latency = 0.0

            # 获取推理
            reasoning = latest.get("reasoning", "")
            if pd.isna(reasoning):
                reasoning = ""

            signal_data = {
                "signal": signal,
                "confidence": confidence,
                "latency_ms": latency,
                "reasoning": reasoning,
            }

            result[symbol] = signal_data
            # 缓存 LLM 信号
            self._last_llm_signals[symbol] = signal_data

        return result

    def _perform_fusion(
        self,
        ml_signal: float,
        ml_confidence: float,
        llm_signal: float,
        llm_confidence: float,
        llm_reasoning: str,
        regime: MarketRegime,
    ) -> tuple[float, str, str]:
        """
        执行信号融合。

        Args:
            ml_signal: ML 信号 (-1 到 1)。
            ml_confidence: ML 置信度。
            llm_signal: LLM 信号 (-1 到 1)。
            llm_confidence: LLM 置信度。
            llm_reasoning: LLM 推理过程。
            regime: 市场状态。

        Returns:
            (融合权重, 融合来源, 推理过程) 元组。
        """
        # 黑天鹅事件：LLM 具有强制清仓权
        if regime == MarketRegime.BLACK_SWAN:
            if llm_signal <= -0.8 and llm_confidence >= self.llm_force_sell_threshold:
                reasoning = f"【黑天鹅事件】LLM 强制清仓: {llm_reasoning[:100]}..."
                return -1.0, "llm_veto", reasoning

        # 高波动率：增加 LLM 权重
        if regime == MarketRegime.HIGH_VOLATILITY:
            # LLM 否决权检查
            if llm_confidence >= self.llm_veto_threshold:
                if abs(llm_signal) > abs(ml_signal) * 1.5:
                    reasoning = f"【高波动】LLM 否决 ML: ML={ml_signal:.2f}, LLM={llm_signal:.2f}"
                    return llm_signal, "llm_veto", reasoning

            # 加权融合
            ml_weight = self.ml_weight_high_vol
            llm_weight = 1 - ml_weight

            fused = ml_signal * ml_weight + llm_signal * llm_weight
            reasoning = f"【高波动】加权融合: ML({ml_weight:.0%})={ml_signal:.2f}, LLM({llm_weight:.0%})={llm_signal:.2f}"
            return fused, "weighted_high_vol", reasoning

        # 正常时期：ML 为主导
        ml_weight = self.ml_weight_normal
        llm_weight = 1 - ml_weight

        # 只有在有 LLM 信号时才进行加权
        if llm_confidence > 0:
            fused = ml_signal * ml_weight + llm_signal * llm_weight * llm_confidence
            reasoning = f"【正常】加权融合: ML({ml_weight:.0%})={ml_signal:.2f}, LLM({llm_weight:.0%})={llm_signal:.2f}"
        else:
            fused = ml_signal
            reasoning = f"【正常】仅 ML 信号: {ml_signal:.2f}"

        return fused, "ml_dominant", reasoning

    def _detect_overall_regime(self, signals: list[FusedSignal]) -> MarketRegime:
        """
        检测整体市场状态。

        Args:
            signals: 融合信号列表。

        Returns:
            整体市场状态。
        """
        if not signals:
            return MarketRegime.NORMAL

        regimes = [s.market_regime for s in signals]

        # 如果任何信号处于黑天鹅状态，整体为黑天鹅
        if MarketRegime.BLACK_SWAN.value in regimes:
            return MarketRegime.BLACK_SWAN

        # 如果大多数信号处于高波动状态，整体为高波动
        high_vol_count = regimes.count(MarketRegime.HIGH_VOLATILITY.value)
        if high_vol_count > len(regimes) / 2:
            return MarketRegime.HIGH_VOLATILITY

        return MarketRegime.NORMAL

    def generate_target_positions(
        self,
        ml_signals: pd.DataFrame,
        llm_signals: pd.DataFrame,
        volatility: float = 0.02,
        has_major_news: bool = False,
        position_sizing: str = "equal_weight",
        current_time: Optional[datetime] = None,
    ) -> dict[str, TargetPosition]:
        """
        生成目标仓位字典。

        这是融合引擎的核心输出方法，产生最终发送给执行引擎的统一目标仓位。

        包含调仓死区逻辑：当新仓位与旧仓位的差异小于阈值时，保持原仓位不变，
        避免微调仓带来的手续费磨损。

        Args:
            ml_signals: ML Track 信号 DataFrame。
            llm_signals: LLM Track 信号 DataFrame。
            volatility: 当前波动率。
            has_major_news: 是否有重大新闻输入。
            position_sizing: 仓位分配策略，可选 'equal_weight', 'confidence_weighted', 'risk_parity'。
            current_time: 当前时间（用于 LLM 信号衰减计算）。

        Returns:
            目标仓位字典 {symbol: TargetPosition}。
        """
        current_time = current_time or datetime.now()

        # 融合信号（传入当前时间用于 LLM 衰减计算）
        fused_signals = self.fuse_signals(
            ml_signals=ml_signals,
            llm_signals=llm_signals,
            volatility=volatility,
            has_major_news=has_major_news,
            current_time=current_time,
        )

        # 生成目标仓位
        target_positions: dict[str, TargetPosition] = {}
        timestamp = current_time

        # 计算仓位权重
        if position_sizing == "equal_weight":
            weights = self._equal_weight_sizing(fused_signals)
        elif position_sizing == "confidence_weighted":
            weights = self._confidence_weighted_sizing(fused_signals)
        else:
            weights = self._risk_parity_sizing(fused_signals)

        for signal in fused_signals:
            symbol = signal.symbol
            new_weight = weights.get(symbol, signal.fused_weight)

            # ================================================================
            # 调仓死区检查 (Rebalancing Dead Zone)
            # ================================================================
            old_weight = self._last_target_positions.get(symbol, 0.0)

            # 判断是否需要调仓的条件：
            # 1. 仓位方向改变（正负号变化）
            # 2. 仓位幅度变化超过阈值
            sign_change = (old_weight * new_weight < 0)
            magnitude_change = abs(new_weight - old_weight)

            if sign_change or magnitude_change >= self.rebalance_threshold:
                # 需要调仓
                final_weight = new_weight
                dead_zone_applied = False
            elif old_weight == 0.0 and magnitude_change < self.rebalance_threshold:
                # 从空仓到小幅建仓，仍然执行
                final_weight = new_weight
                dead_zone_applied = False
            else:
                # 在死区内，保持原仓位（避免手续费磨损）
                final_weight = old_weight
                dead_zone_applied = True

            # 计算总延迟
            latency_metrics = LatencyMetrics(
                ml_latency_ms=signal.ml_latency_ms,
                llm_latency_ms=signal.llm_latency_ms,
                fusion_latency_ms=0.0,  # 稍后计算
                total_latency_ms=signal.ml_latency_ms + signal.llm_latency_ms,
            )

            # 确定信号来源
            if signal.fusion_source == "llm_veto":
                source = "llm_veto"
                confidence = signal.llm_confidence
                reasoning = signal.reasoning
            elif signal.fusion_source == "ml_dominant":
                source = "ml_dominant"
                confidence = signal.ml_confidence
                reasoning = signal.reasoning
            else:
                source = "fused"
                confidence = (signal.ml_confidence + signal.llm_confidence) / 2
                reasoning = signal.reasoning

            # 如果应用了调仓死区，在推理中标注
            if dead_zone_applied:
                reasoning = f"[死区保持] 变化 {magnitude_change:.2%} < 阈值 {self.rebalance_threshold:.0%}"

            target_positions[symbol] = TargetPosition(
                symbol=symbol,
                weight=final_weight,
                signal_source=source,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=timestamp,
                latency_metrics=latency_metrics,
                market_regime=signal.market_regime,
                metadata={
                    "ml_signal": signal.ml_signal,
                    "llm_signal": signal.llm_signal,
                    "fusion_source": signal.fusion_source,
                    "dead_zone_applied": dead_zone_applied,
                    "original_weight": new_weight,
                },
            )

            # 更新缓存
            self._last_target_positions[symbol] = final_weight

        # 记录延迟历史
        total_latency = sum(
            pos.latency_metrics.total_latency_ms for pos in target_positions.values()
        ) / max(len(target_positions), 1)
        self._latency_history.append(LatencyMetrics(
            total_latency_ms=total_latency,
        ))

        return target_positions

    def _equal_weight_sizing(self, signals: list[FusedSignal]) -> dict[str, float]:
        """
        等权重仓位分配。

        Args:
            signals: 融合信号列表。

        Returns:
            仓位权重字典。
        """
        return {s.symbol: s.fused_weight for s in signals}

    def _confidence_weighted_sizing(self, signals: list[FusedSignal]) -> dict[str, float]:
        """
        置信度加权仓位分配。

        Args:
            signals: 融合信号列表。

        Returns:
            仓位权重字典。
        """
        weights = {}
        for s in signals:
            # 使用置信度调整仓位大小
            avg_confidence = (s.ml_confidence + s.llm_confidence) / 2
            adjusted_weight = s.fused_weight * avg_confidence
            weights[s.symbol] = adjusted_weight

        return weights

    def _risk_parity_sizing(self, signals: list[FusedSignal]) -> dict[str, float]:
        """
        风险平价仓位分配（简化版）。

        Args:
            signals: 融合信号列表。

        Returns:
            仓位权重字典。
        """
        weights = {}
        total_abs_weight = sum(abs(s.fused_weight) for s in signals) + 1e-10

        for s in signals:
            # 按风险贡献分配
            risk_weight = abs(s.fused_weight) / total_abs_weight
            # 保持方向
            weights[s.symbol] = np.sign(s.fused_weight) * risk_weight

        return weights

    def get_current_regime(self) -> MarketRegime:
        """
        获取当前市场状态。

        Returns:
            当前市场状态。
        """
        return self._current_regime

    def get_latency_stats(self) -> dict:
        """
        获取延迟统计信息。

        Returns:
            延迟统计字典。
        """
        if not self._latency_history:
            return {
                "avg_total_latency_ms": 0.0,
                "max_total_latency_ms": 0.0,
                "min_total_latency_ms": 0.0,
                "sample_count": 0,
            }

        latencies = [l.total_latency_ms for l in self._latency_history]
        return {
            "avg_total_latency_ms": np.mean(latencies),
            "max_total_latency_ms": np.max(latencies),
            "min_total_latency_ms": np.min(latencies),
            "sample_count": len(latencies),
        }

    def get_signal_history(self, limit: int = 100) -> list[dict]:
        """
        获取信号历史记录。

        Args:
            limit: 最大返回数量。

        Returns:
            信号历史列表。
        """
        return [s.to_dict() for s in self._signal_history[-limit:]]

    def clear_history(self) -> None:
        """清空历史记录。"""
        self._signal_history.clear()
        self._latency_history.clear()
        self._last_llm_signals.clear()
        self._last_target_positions.clear()
        self._last_llm_signal_time = None


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("  信号融合引擎示例")
    print("=" * 60)

    # 创建示例 ML 信号
    ml_signals = pd.DataFrame([
        {"symbol": "CSI300", "model_name": "LightGBM", "signal_strength_0_to_1": 0.7, "latency_ms": 1.5},
        {"symbol": "CSI300", "model_name": "LogisticRegression", "signal_strength_0_to_1": 0.6, "latency_ms": 0.5},
        {"symbol": "CSI300", "model_name": "LSTM", "signal_strength_0_to_1": 0.55, "latency_ms": 15.0},
        {"symbol": "NASDAQ100", "model_name": "LightGBM", "signal_strength_0_to_1": 0.4, "latency_ms": 1.8},
    ])

    # 创建示例 LLM 信号
    llm_signals = pd.DataFrame([
        {"symbol": "CSI300", "llm_signal": "sell", "confidence": 0.85, "reasoning": "央行加息预期升温", "latency_ms": 1200},
        {"symbol": "NASDAQ100", "llm_signal": "hold", "confidence": 0.5, "reasoning": "市场观望", "latency_ms": 980},
    ])

    # 初始化融合引擎
    engine = SignalFusionEngine(
        volatility_threshold=0.03,
        llm_veto_threshold=0.8,
        ml_weight_normal=0.7,
    )

    print("\n【场景 1】正常波动率，无重大新闻")
    positions = engine.generate_target_positions(
        ml_signals=ml_signals,
        llm_signals=llm_signals,
        volatility=0.02,
        has_major_news=False,
    )
    print(f"市场状态: {engine.get_current_regime().value}")
    for symbol, pos in positions.items():
        print(f"  {symbol}: weight={pos.weight:.3f}, source={pos.signal_source}")
        print(f"    延迟: ML={pos.latency_metrics.ml_latency_ms:.1f}ms, LLM={pos.latency_metrics.llm_latency_ms:.1f}ms")

    print("\n【场景 2】高波动率")
    positions = engine.generate_target_positions(
        ml_signals=ml_signals,
        llm_signals=llm_signals,
        volatility=0.04,
        has_major_news=False,
    )
    print(f"市场状态: {engine.get_current_regime().value}")
    for symbol, pos in positions.items():
        print(f"  {symbol}: weight={pos.weight:.3f}, source={pos.signal_source}")
        print(f"    推理: {pos.reasoning[:60]}...")

    print("\n【场景 3】黑天鹅事件（LLM 强制清仓）")
    # 创建极端 LLM 信号
    extreme_llm = pd.DataFrame([
        {"symbol": "CSI300", "llm_signal": "sell", "confidence": 0.95, "reasoning": "【紧急】财务造假丑闻曝光，退市风险极高", "latency_ms": 850},
    ])
    positions = engine.generate_target_positions(
        ml_signals=ml_signals,
        llm_signals=extreme_llm,
        volatility=0.08,
        has_major_news=True,
    )
    print(f"市场状态: {engine.get_current_regime().value}")
    for symbol, pos in positions.items():
        print(f"  {symbol}: weight={pos.weight:.3f}, source={pos.signal_source}")
        print(f"    推理: {pos.reasoning}")

    print("\n【延迟统计】")
    stats = engine.get_latency_stats()
    print(f"  平均延迟: {stats['avg_total_latency_ms']:.2f}ms")
    print(f"  最大延迟: {stats['max_total_latency_ms']:.2f}ms")
    print(f"  最小延迟: {stats['min_total_latency_ms']:.2f}ms")
    print(f"  样本数量: {stats['sample_count']}")