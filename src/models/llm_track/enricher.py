"""
市场数据增强器模块。

预计算技术指标、价格序列、记忆上下文，供 SmartPromptAgent 一次性注入 prompt。
严格防止未来函数泄漏：所有数据仅使用 current_date 之前的历史信息。
"""

from typing import Optional

import pandas as pd

from src.models.llm_track.memory import DecisionMemoryStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MarketEnricher:
    """
    市场数据增强器。

    为每个交易日预计算技术指标摘要、近 N 日价格走势、决策记忆反馈，
    以自然语言结构化格式输出，直接注入 LLM prompt。

    所有数据获取严格限制在 current_date 之前，防止未来函数泄漏。

    Attributes:
        ohlcv: 完整 OHLCV 数据。
        historical: 截至 current_date 的历史数据（物理隔离未来数据）。
        current_date: 当前模拟日期。
    """

    def __init__(self, ohlcv_df: pd.DataFrame, current_date: str) -> None:
        """
        初始化数据增强器。

        Args:
            ohlcv_df: 包含 open/high/low/close/volume 列的 DataFrame，index 为日期。
            current_date: 当前模拟日期（格式 YYYY-MM-DD）。

        Raises:
            AssertionError: 如果 current_date 早于所有历史数据。
        """
        self.ohlcv = ohlcv_df.copy()
        self.current_date = pd.Timestamp(current_date)

        # 严格截取历史数据，物理隔离未来数据
        self.historical = self.ohlcv[self.ohlcv.index < self.current_date]

        if self.historical.empty:
            raise AssertionError(
                f"无历史数据: current_date={self.current_date} 早于所有 OHLCV 数据"
            )

    def _assert_no_lookahead(self) -> None:
        """
        内部断言：确保历史数据不包含当前日期及之后的数据。

        Raises:
            AssertionError: 如果检测到未来函数泄漏风险。
        """
        if not self.historical.empty:
            assert self.historical.index.max() < self.current_date, (
                f"未来函数泄漏: historical 最大日期 {self.historical.index.max()} "
                f">= 当前日期 {self.current_date}"
            )

    def get_technical_indicators(self) -> str:
        """
        计算并返回自然语言格式的技术指标摘要。

        内部复用 FeatureEngineer 计算 RSI、MACD、布林带、均线、量价关系，
        并生成可读判断（"死叉"、"超卖"、"空头排列" 等）。

        Returns:
            自然语言技术指标摘要，如：
            "技术面状态:\n- RSI(14): 42.3 (偏弱区间，未超卖)\n..."
        """
        self._assert_no_lookahead()
        df = self.historical.copy()

        if len(df) < 30:
            # 数据不足，返回有限信息
            return f"技术面状态: 历史数据不足 ({len(df)} 天)，仅供参考\n"

        # --- RSI ---
        from src.models.ml_track.features import FeatureEngineer

        fe = FeatureEngineer()

        # 计算 RSI(14)
        rsi_df = fe.compute_rsi(df, periods=[14])
        rsi_14 = rsi_df["rsi_14"].iloc[-1]

        if rsi_14 < 30:
            rsi_desc = "超卖区间，可能反弹"
        elif rsi_14 < 45:
            rsi_desc = "偏弱区间，未超卖"
        elif rsi_14 < 55:
            rsi_desc = "中性区间"
        elif rsi_14 < 70:
            rsi_desc = "偏强区间，未超买"
        else:
            rsi_desc = "超买区间，可能回调"

        # --- MACD ---
        macd_df = fe.compute_macd(df)
        macd_line = macd_df["macd"].iloc[-1]
        macd_signal = macd_df["macd_signal"].iloc[-1]
        macd_hist = macd_df["macd_histogram"].iloc[-1]

        if macd_line < macd_signal:
            macd_desc = f"死叉确认 ({macd_line:.1f} < {macd_signal:.1f})"
            if macd_hist < 0:
                macd_desc += "，空头动能增强"
        else:
            macd_desc = f"金叉确认 ({macd_line:.1f} > {macd_signal:.1f})"
            if macd_hist > 0:
                macd_desc += "，多头动能增强"

        # --- 布林带 ---
        bb_df = fe.compute_bollinger_bands(df, window=20)
        bb_upper = bb_df["bb_upper"].iloc[-1]
        bb_lower = bb_df["bb_lower"].iloc[-1]
        bb_pos = bb_df["bb_position"].iloc[-1]
        close = df["close"].iloc[-1]

        if bb_pos < 0.1:
            bb_desc = f"价格触及下轨 ({close:.0f} vs 下轨 {bb_lower:.0f})，短线超卖风险"
        elif bb_pos < 0.3:
            bb_desc = f"价格靠近下轨 ({bb_pos:.0%})，弱势运行"
        elif bb_pos < 0.7:
            bb_desc = f"价格位于布林带中部 ({bb_pos:.0%})，震荡整理"
        elif bb_pos < 0.9:
            bb_desc = f"价格靠近上轨 ({bb_pos:.0%})，偏强走势"
        else:
            bb_desc = f"价格触及上轨 ({close:.0f} vs 上轨 {bb_upper:.0f})，短线超买风险"

        # --- 均线排列 ---
        ma_cross_df = fe.compute_ma_cross(df, short_windows=[5, 10], long_windows=[20, 60])
        ma_5 = ma_cross_df["ma_5"].iloc[-1]
        ma_20 = ma_cross_df["ma_20"].iloc[-1]

        if ma_5 < ma_20:
            ma_desc = f"MA5({ma_5:.0f}) < MA20({ma_20:.0f})，空头排列"
        else:
            ma_desc = f"MA5({ma_5:.0f}) > MA20({ma_20:.0f})，多头排列"

        # --- 量价关系 ---
        vol_df = fe.compute_volume_features(df, windows=[5])
        vol_ratio = vol_df["volume_ratio_5d"].iloc[-1]
        recent_return = df["close"].pct_change().iloc[-1]

        if vol_ratio > 1.5 and recent_return < -0.01:
            vol_desc = "放量下跌，资金出逃"
        elif vol_ratio > 1.5 and recent_return > 0.01:
            vol_desc = "放量上涨，资金流入"
        elif vol_ratio < 0.7:
            vol_desc = f"缩量交易 (量比 {vol_ratio:.2f})，观望情绪浓"
        else:
            vol_desc = f"量能正常 (量比 {vol_ratio:.2f})"

        return (
            f"技术面状态:\n"
            f"- RSI(14): {rsi_14:.1f} ({rsi_desc})\n"
            f"- MACD: {macd_desc}\n"
            f"- 布林带: {bb_desc}\n"
            f"- 均线: {ma_desc}\n"
            f"- 量价: {vol_desc}"
        )

    def get_price_history(self, window: int = 5) -> str:
        """
        返回近 N 日价格走势的 Markdown 表格。

        Args:
            window: 回顾天数，默认 5。

        Returns:
            Markdown 格式的价格走势表格。
        """
        self._assert_no_lookahead()
        df = self.historical.tail(window)

        if df.empty or len(df) < 2:
            return ""

        lines = [f"近{window}日走势:"]
        lines.append("日期      收盘价    涨跌幅    成交量")

        for idx, row in df.iterrows():
            date_str = idx.strftime("%m-%d")
            close = row["close"]
            volume = row["volume"]

            # 计算涨跌幅（与前一天比较）
            if len(lines) > 2:  # 不是第一行
                prev_close = prev_row["close"]
                change_pct = (close - prev_close) / prev_close * 100
                change_str = f"{change_pct:+.1f}%"
            else:
                change_str = "N/A"

            # 量级判断
            vol_str = f"{volume/1e6:.0f}M" if volume >= 1e6 else f"{volume/1e3:.0f}K"

            # 标记放量
            note = ""
            if change_str != "N/A":
                try:
                    change_val = float(change_str.replace("+", "").replace("%", ""))
                    if change_val < -1.0 and volume > df["volume"].mean() * 1.3:
                        note = " (放量下跌)"
                    elif change_val > 1.0 and volume > df["volume"].mean() * 1.3:
                        note = " (放量上涨)"
                except ValueError:
                    pass

            lines.append(f"{date_str}    {close:>8.1f}   {change_str:>6s}    {vol_str}{note}")

            prev_row = row

        return "\n".join(lines)

    def get_memory_context(
        self,
        memory_store: Optional[DecisionMemoryStore] = None,
        n: int = 5,
    ) -> str:
        """
        生成决策记忆+反馈文本。

        遍历记忆中的每条记录，自动回填 actual_return 并附加结果评价。

        Args:
            memory_store: 决策记忆存储。
            n: 返回最近 N 条记录。

        Returns:
            紧凑的决策记忆文本，或空字符串（如果记忆为空）。
        """
        if memory_store is None or len(memory_store) == 0:
            return ""

        return memory_store.get_recent_context(
            n=n,
            ohlcv_df=self.ohlcv,
            current_date=self.current_date,
        )

    def get_enriched_context(
        self,
        memory_store: Optional[DecisionMemoryStore] = None,
        price_window: int = 5,
        memory_n: int = 5,
    ) -> str:
        """
        一次性获取所有增强上下文（技术指标 + 价格历史 + 记忆）。

        这是 SmartPromptAgent 调用的主要入口。

        Args:
            memory_store: 决策记忆存储。
            price_window: 价格回顾天数。
            memory_n: 记忆回顾条数。

        Returns:
            完整的增强上下文字符串。
        """
        parts = []

        # 技术指标（始终注入，除非数据不足）
        try:
            tech = self.get_technical_indicators()
            if tech:
                parts.append(tech)
        except Exception as e:
            logger.warning(f"技术指标计算失败: {e}")
            parts.append("技术面状态: 计算失败，仅供参考")

        # 价格历史（始终注入）
        try:
            price = self.get_price_history(window=price_window)
            if price:
                parts.append(price)
        except Exception as e:
            logger.warning(f"价格历史生成失败: {e}")

        # 决策记忆（冷启动时为空，不报错）
        try:
            memory = self.get_memory_context(memory_store, n=memory_n)
            if memory:
                parts.append(memory)
        except Exception as e:
            logger.warning(f"记忆上下文生成失败: {e}")

        if not parts:
            return ""

        return "\n\n".join(parts)
