"""
决策记忆模块。

记录 Agent 的交易决策及其真实市场结果，形成闭环反馈。
用于在后续决策的 prompt 中注入历史决策记录，使 LLM 能看到自己过去的判断是对是错。
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class DecisionRecord:
    """
    决策记录数据类。

    包含完整决策-结果生命周期：Agent 当时怎么想的 (signal/confidence/reasoning)，
    以及后续由 MarketEnricher 回填的真实市场结果 (actual_return/outcome_text)。
    """
    date: str
    symbol: str
    signal: str           # "buy" / "sell" / "hold"
    confidence: float     # 0.0 - 1.0
    reasoning: str        # Agent 当时的推理

    # 闭环反馈：T+1 日由 MarketEnricher 回填
    actual_return: Optional[float] = None   # T 日→T+1 日的真实收益率
    outcome_text: Optional[str] = None      # 自然语言描述，如 "决策正确 +1.2%"

    def to_dict(self) -> dict:
        """转换为字典。"""
        return {
            "date": self.date,
            "symbol": self.symbol,
            "signal": self.signal,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "actual_return": self.actual_return,
            "outcome_text": self.outcome_text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DecisionRecord":
        """从字典创建。"""
        return cls(
            date=data.get("date", ""),
            symbol=data.get("symbol", "UNKNOWN"),
            signal=data.get("signal", "hold"),
            confidence=data.get("confidence", 0.5),
            reasoning=data.get("reasoning", ""),
            actual_return=data.get("actual_return"),
            outcome_text=data.get("outcome_text"),
        )


class DecisionMemoryStore:
    """
    决策记忆存储。

    维护一个滑动窗口的决策历史记录，支持自动回填真实市场收益。
    用于在 prompt 中注入 Agent 过去的决策及其结果。

    Attributes:
        max_history: 滑动窗口大小，默认 20 条。
        records: 决策记录列表。
    """

    def __init__(self, max_history: int = 20) -> None:
        """
        初始化记忆存储。

        Args:
            max_history: 最大保留的历史记录数。
        """
        self.max_history = max_history
        self.records: list[DecisionRecord] = []

    def add_record(self, record: DecisionRecord) -> None:
        """
        添加新决策记录。

        新记录的 actual_return 初始为 None，由 MarketEnricher 在后续交易日回填。

        Args:
            record: 决策记录。
        """
        self.records.append(record)
        # 维护滑动窗口
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history:]

    def get_signal_streak(self) -> tuple[str, int]:
        """
        获取当前连续同向信号。

        Returns:
            (信号类型, 连续次数)。例如 ("buy", 3) 表示连续 3 次 buy。
        """
        if not self.records:
            return ("", 0)

        current_signal = self.records[-1].signal
        streak = 1
        for record in reversed(self.records[:-1]):
            if record.signal == current_signal:
                streak += 1
            else:
                break

        return (current_signal, streak)

    def get_avg_confidence(self, window: int = 10) -> float:
        """
        获取最近 N 条记录的平均置信度。

        Args:
            window: 窗口大小。

        Returns:
            平均置信度。
        """
        if not self.records:
            return 0.5

        recent = self.records[-window:]
        return sum(r.confidence for r in recent) / len(recent)

    def get_recent_context(
        self,
        n: int = 5,
        ohlcv_df: Optional[pd.DataFrame] = None,
        current_date: Optional[pd.Timestamp] = None,
    ) -> str:
        """
        生成最近 N 条决策记录的紧凑文本，用于注入 prompt。

        对每条记录，自动计算 actual_return（如果尚未回填）并附加结果评价。

        Args:
            n: 返回最近 N 条记录。
            ohlcv_df: OHLCV 数据，用于计算 actual_return。
            current_date: 当前模拟日期，确保不使用未来数据。

        Returns:
            紧凑文本，如：
            "近5日决策历史:\n  2天前: buy (conf: 0.72) - 政策利好预期 → 实际: 暴跌 -2.5% (决策失误)"
        """
        if not self.records:
            return ""

        recent = self.records[-n:]
        lines = [f"近{n}日决策历史:"]

        for i, record in enumerate(recent):
            date_str = record.date[:10] if len(record.date) > 10 else record.date
            base_text = (
                f"  {date_str}: {record.signal} "
                f"(conf: {record.confidence:.2f}) - {record.reasoning[:50]}"
            )

            # 回填 actual_return（如果尚未回填且有 OHLCV 数据）
            if record.actual_return is None and ohlcv_df is not None and current_date is not None:
                actual_ret = self._calculate_actual_return(
                    record.date, ohlcv_df, current_date
                )
                if actual_ret is not None:
                    record.actual_return = actual_ret
                    record.outcome_text = self._format_outcome(actual_ret, record.signal)

            # 附加结果评价
            if record.outcome_text:
                base_text += f" → {record.outcome_text}"
            elif record.actual_return is not None:
                base_text += f" → {self._format_outcome(record.actual_return, record.signal)}"

            lines.append(base_text)

        # 附加信号趋势
        streak_signal, streak_count = self.get_signal_streak()
        if streak_count > 1:
            lines.append(f"当前连续: {streak_signal} ×{streak_count}")

        return "\n".join(lines)

    def _calculate_actual_return(
        self,
        date: str,
        ohlcv_df: pd.DataFrame,
        current_date: pd.Timestamp,
    ) -> Optional[float]:
        """
        计算某决策记录对应的真实收益率。

        DecisionRecord 的 date 是执行日（T日），即该决策将在当天开盘建仓。
        实际收益 = T 日当天的开收盘涨跌 = (T收盘 - T开盘) / T开盘。

        时间线：
          T-1日收盘后: LLM 用 T-1 日数据做出针对 T 日的决策
          T  日开盘: 以开盘价建仓（record.date = T 日）
          T  日收盘: 盈亏已知
          T+1日: 生成决策历史，T 日已收盘，收益可知

        Args:
            date: 决策记录的执行日（即建仓日）。
            ohlcv_df: OHLCV 数据。
            current_date: 当前模拟日期。

        Returns:
            收益率，或 None（如果执行日尚未收盘）。
        """
        date_ts = pd.Timestamp(date)
        # 使用 <= current_date 的数据
        historical = ohlcv_df[ohlcv_df.index <= current_date]
        if historical.empty:
            return None

        # date 本身就是执行日，直接用当天的开收盘
        if date_ts not in historical.index:
            available = historical.index[historical.index <= date_ts]
            if len(available) == 0:
                return None
            date_ts = available[-1]

        open_price = historical.loc[date_ts, "open"]
        close = historical.loc[date_ts, "close"]

        return (close - open_price) / open_price

    def _format_outcome(self, actual_return: float, signal: str) -> str:
        """
        格式化实际收益为自然语言评价。

        Args:
            actual_return: 实际收益率。
            signal: 当时的决策信号。

        Returns:
            如 "实际: 上涨 +2.3% (决策正确)"。
        """
        pct = actual_return * 100
        direction = "上涨" if actual_return > 0 else "下跌"
        abs_pct = abs(pct)

        # 判断决策是否正确
        if signal == "buy":
            correct = actual_return > 0
        elif signal == "sell":
            correct = actual_return < 0
        else:  # hold
            correct = abs(actual_return) < 1.0  # 波动 <1% 认为 hold 正确

        verdict = "决策正确" if correct else "决策失误"
        return f"实际: {direction} {abs_pct:.1f}% ({verdict})"

    def clear(self) -> None:
        """清空所有记录。"""
        self.records.clear()

    def __len__(self) -> int:
        """返回记录数量。"""
        return len(self.records)

    def save_jsonl(self, path: Path) -> None:
        """
        保存记录到 JSONL 文件。

        Args:
            path: 输出文件路径。
        """
        if not self.records:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for record in self.records:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def load_jsonl(self, path: Path) -> None:
        """
        从 JSONL 文件加载记录。

        Args:
            path: 输入文件路径。
        """
        if not path.exists():
            return

        self.records.clear()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        self.records.append(DecisionRecord.from_dict(data))
                    except (json.JSONDecodeError, KeyError):
                        continue

        # 维护滑动窗口
        if len(self.records) > self.max_history:
            self.records = self.records[-self.max_history:]
