"""
交易质量分析器模块。

计算 MAE (Maximum Adverse Excursion) 和 MFE (Maximum Favorable Excursion)，
用于评估交易信号的质量和策略的非对称风险特征。

MAE: 持仓期间的最大不利偏移，衡量抗逆境能力。
MFE: 持仓期间的最大有利偏移，衡量机会捕获能力。
交易效率: 实现盈亏 / MFE，衡量离场时机质量。

学术价值:
- 对比 ML（高频试错型）vs LLM（狙击手型）的交易哲学差异
- 揭示 LLM 信号的抗逆境能力（低 MAE 证明宏观利空新闻过滤有效）
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TradeQualityMetrics:
    """
    单笔交易质量指标数据类。

    Attributes:
        trade_id: 交易ID。
        entry_date: 入场日期。
        exit_date: 出场日期。
        entry_price: 入场价格。
        exit_price: 出场价格。
        direction: 方向 (1=多头, -1=空头)。
        pnl: 实现盈亏（金额或百分比）。
        pnl_pct: 盈亏百分比。
        mae: 最大不利偏移 (Maximum Adverse Excursion)。
        mfe: 最大有利偏移 (Maximum Favorable Excursion)。
        mae_pct: MAE 百分比。
        mfe_pct: MFE 百分比。
        efficiency: 交易效率 (pnl_pct / mfe_pct)。
        hold_days: 持仓天数。
        is_winner: 是否盈利。
    """
    trade_id: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    direction: int  # 1=多头, -1=空头
    pnl: float
    pnl_pct: float
    mae: float = 0.0
    mfe: float = 0.0
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    efficiency: float = 0.0
    hold_days: int = 0
    is_winner: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "trade_id": self.trade_id,
            "entry_date": self.entry_date.isoformat() if pd.notna(self.entry_date) else None,
            "exit_date": self.exit_date.isoformat() if pd.notna(self.exit_date) else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "direction": "多头" if self.direction > 0 else "空头",
            "pnl": self.pnl,
            "pnl_pct": f"{self.pnl_pct:.2%}",
            "mae_pct": f"{self.mae_pct:.2%}",
            "mfe_pct": f"{self.mfe_pct:.2%}",
            "efficiency": f"{self.efficiency:.2%}",
            "hold_days": self.hold_days,
            "is_winner": self.is_winner,
        }


@dataclass
class TradeQualitySummary:
    """
    交易质量汇总统计数据类。

    Attributes:
        strategy_name: 策略名称。
        total_trades: 总交易次数。
        winning_trades: 盈利交易次数。
        losing_trades: 亏损交易次数。
        win_rate: 胜率。
        avg_pnl: 平均盈亏。
        avg_winner: 平均盈利交易盈亏。
        avg_loser: 平均亏损交易盈亏。
        payoff_ratio: 盈亏比 (avg_winner / abs(avg_loser))。
        avg_mae: 平均 MAE。
        avg_mfe: 平均 MFE。
        avg_mae_winner: 盈利交易平均 MAE。
        avg_mae_loser: 亏损交易平均 MAE。
        avg_mfe_winner: 盈利交易平均 MFE。
        avg_mfe_loser: 亏损交易平均 MFE。
        avg_efficiency: 平均交易效率。
        avg_hold_days: 平均持仓天数。
        profit_factor: 盈利因子 (总盈利 / 总亏损绝对值)。
    """
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_pnl: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    payoff_ratio: float = 0.0
    avg_mae: float = 0.0
    avg_mfe: float = 0.0
    avg_mae_winner: float = 0.0
    avg_mae_loser: float = 0.0
    avg_mfe_winner: float = 0.0
    avg_mfe_loser: float = 0.0
    avg_efficiency: float = 0.0
    avg_hold_days: float = 0.0
    profit_factor: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式。"""
        return {
            "strategy_name": self.strategy_name,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": f"{self.win_rate:.2%}",
            "avg_pnl": f"{self.avg_pnl:.2f}",
            "avg_winner": f"{self.avg_winner:.2f}",
            "avg_loser": f"{self.avg_loser:.2f}",
            "payoff_ratio": f"{self.payoff_ratio:.2f}",
            "avg_mae_pct": f"{self.avg_mae:.2%}",
            "avg_mfe_pct": f"{self.avg_mfe:.2%}",
            "avg_efficiency": f"{self.avg_efficiency:.2%}",
            "avg_hold_days": f"{self.avg_hold_days:.1f}",
            "profit_factor": f"{self.profit_factor:.2f}",
        }

    def summary(self) -> str:
        """生成汇总摘要字符串。"""
        lines = [
            f"\n{'='*60}",
            f"  交易质量分析: {self.strategy_name}",
            f"{'='*60}",
            f"  总交易次数:   {self.total_trades:>10}",
            f"  盈利交易:     {self.winning_trades:>10}",
            f"  亏损交易:     {self.losing_trades:>10}",
            f"  胜率:         {self.win_rate:>10.2%}",
            f"{'-'*60}",
            f"  平均盈亏:     {self.avg_pnl:>10.2f}",
            f"  平均盈利:     {self.avg_winner:>10.2f}",
            f"  平均亏损:     {self.avg_loser:>10.2f}",
            f"  盈亏比:       {self.payoff_ratio:>10.2f}",
            f"  盈利因子:     {self.profit_factor:>10.2f}",
            f"{'-'*60}",
            f"  平均 MAE:     {self.avg_mae:>10.2%}",
            f"  平均 MFE:     {self.avg_mfe:>10.2%}",
            f"  平均效率:     {self.avg_efficiency:>10.2%}",
            f"  平均持仓:     {self.avg_hold_days:>10.1f} 天",
            f"{'='*60}",
        ]
        return "\n".join(lines)


class TradeAnalyzer:
    """
    交易质量分析器。

    计算 MAE、MFE、交易效率等交易质量指标，支持论文所需的对比分析。

    使用方法:
        analyzer = TradeAnalyzer()

        # 从回测结果计算交易质量
        metrics_list = analyzer.analyze_trades(
            trade_log=trade_df,
            ohlcv_data=ohlcv_df,
        )

        # 汇总统计
        summary = analyzer.summarize(metrics_list, strategy_name="LightGBM")

        # 生成可视化
        analyzer.plot_mae_mfe_scatter(metrics_list, save_path="docs/figures/mae_mfe.png")
    """

    def __init__(self) -> None:
        """初始化分析器。"""
        self.metrics_list: List[TradeQualityMetrics] = []

    def analyze_trades(
        self,
        trade_log: pd.DataFrame,
        ohlcv_data: pd.DataFrame,
        price_col: str = "close",
        direction_col: str = "type",
        assume_long_only: bool = True,
    ) -> List[TradeQualityMetrics]:
        """
        分析交易质量。

        Args:
            trade_log: 交易日志 DataFrame，需包含 'date', 'price', 'size' 等列。
            ohlcv_data: OHLCV 数据 DataFrame，需包含 'high', 'low', 'close' 列。
            price_col: 用于计算的价格列名。
            direction_col: 方向列名（用于判断买卖方向）。
            assume_long_only: 是否假设只有多头交易（A股规则）。

        Returns:
            TradeQualityMetrics 列表。
        """
        if trade_log.empty or ohlcv_data.empty:
            logger.warning("交易日志或 OHLCV 数据为空")
            return []

        metrics_list: List[TradeQualityMetrics] = []

        # 确保日期索引
        if not isinstance(ohlcv_data.index, pd.DatetimeIndex):
            ohlcv_data = ohlcv_data.copy()
            ohlcv_data.index = pd.to_datetime(ohlcv_data.index)

        # 按日期排序交易记录
        trade_log = trade_log.copy()
        if "date" in trade_log.columns:
            trade_log["date"] = pd.to_datetime(trade_log["date"])
        trade_log = trade_log.sort_values("date").reset_index(drop=True)

        # 配对买入和卖出交易
        trades = self._pair_trades(trade_log, direction_col, assume_long_only)

        for i, trade in enumerate(trades):
            entry_date = trade["entry_date"]
            exit_date = trade["exit_date"]
            entry_price = trade["entry_price"]
            exit_price = trade["exit_price"]
            direction = trade["direction"]
            pnl = trade.get("pnl", 0)
            size = trade.get("size", 1)

            # 获取持仓期间的 OHLCV 数据
            mask = (ohlcv_data.index >= entry_date) & (ohlcv_data.index <= exit_date)
            holding_data = ohlcv_data.loc[mask]

            if len(holding_data) < 2:
                continue

            # 计算 MAE 和 MFE
            if direction > 0:  # 多头
                # MAE: 入场价 - 持仓期最低价
                low_price = holding_data["low"].min()
                mae = (entry_price - low_price) / entry_price
                mae_pct = max(0, mae)

                # MFE: 持仓期最高价 - 入场价
                high_price = holding_data["high"].max()
                mfe = (high_price - entry_price) / entry_price
                mfe_pct = max(0, mfe)
            else:  # 空头
                # MAE: 持仓期最高价 - 入场价
                high_price = holding_data["high"].max()
                mae = (high_price - entry_price) / entry_price
                mae_pct = max(0, mae)

                # MFE: 入场价 - 持仓期最低价
                low_price = holding_data["low"].min()
                mfe = (entry_price - low_price) / entry_price
                mfe_pct = max(0, mfe)

            # 计算盈亏百分比
            if direction > 0:
                pnl_pct = (exit_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - exit_price) / entry_price

            # 计算交易效率
            if mfe_pct > 0:
                efficiency = pnl_pct / mfe_pct
            else:
                efficiency = 0.0

            # 持仓天数
            hold_days = len(holding_data)

            # 是否盈利
            is_winner = pnl_pct > 0

            metrics = TradeQualityMetrics(
                trade_id=i,
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=entry_price,
                exit_price=exit_price,
                direction=direction,
                pnl=pnl,
                pnl_pct=pnl_pct,
                mae=mae * entry_price,
                mfe=mfe * entry_price,
                mae_pct=mae_pct,
                mfe_pct=mfe_pct,
                efficiency=efficiency,
                hold_days=hold_days,
                is_winner=is_winner,
            )
            metrics_list.append(metrics)

        self.metrics_list = metrics_list
        return metrics_list

    def _pair_trades(
        self,
        trade_log: pd.DataFrame,
        direction_col: str,
        assume_long_only: bool,
    ) -> List[Dict[str, Any]]:
        """
        配对买入和卖出交易。

        Args:
            trade_log: 交易日志 DataFrame。
            direction_col: 方向列名。
            assume_long_only: 是否假设只有多头交易。

        Returns:
            配对后的交易列表。
        """
        trades: List[Dict[str, Any]] = []

        # 尝试识别买卖方向
        buy_mask = pd.Series([True] * len(trade_log), index=trade_log.index)
        sell_mask = pd.Series([False] * len(trade_log), index=trade_log.index)

        if direction_col in trade_log.columns:
            # 根据方向列判断
            direction_values = trade_log[direction_col].astype(str).str.lower()
            buy_mask = direction_values.str.contains("买入|buy|long|b", na=False)
            sell_mask = direction_values.str.contains("卖出|sell|short|s", na=False)

        # 按 size 判断（正数为买入，负数为卖出）
        if "size" in trade_log.columns:
            size_positive = trade_log["size"] > 0
            size_negative = trade_log["size"] < 0
            # 只有在没有明确方向时才使用 size
            buy_mask = buy_mask | (~buy_mask & ~sell_mask & size_positive)
            sell_mask = sell_mask | (~buy_mask & ~sell_mask & size_negative)

        buy_trades = trade_log[buy_mask].reset_index(drop=True)
        sell_trades = trade_log[sell_mask].reset_index(drop=True)

        # 简单配对：按顺序配对买入和卖出
        # 更精确的方法需要根据持仓 ID 或更复杂的逻辑
        for i in range(min(len(buy_trades), len(sell_trades))):
            buy_trade = buy_trades.iloc[i]

            # 找到对应的卖出交易（在买入之后）
            buy_date = buy_trade["date"] if "date" in buy_trade else buy_trade.name
            matching_sells = sell_trades[sell_trades["date"] > buy_date] if "date" in sell_trades.columns else sell_trades

            if len(matching_sells) > 0:
                sell_trade = matching_sells.iloc[0]

                entry_price = buy_trade.get("price", buy_trade.get("entry_price", 0))
                exit_price = sell_trade.get("price", sell_trade.get("exit_price", 0))

                # 计算 PnL
                size = abs(buy_trade.get("size", 1))
                pnl = (exit_price - entry_price) * size

                # 佣金
                commission = buy_trade.get("commission", 0) + sell_trade.get("commission", 0)
                pnl -= commission

                trades.append({
                    "entry_date": buy_date,
                    "exit_date": sell_trade["date"] if "date" in sell_trade else sell_trade.name,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "direction": 1 if assume_long_only else (1 if buy_trade.get("size", 1) > 0 else -1),
                    "pnl": pnl,
                    "size": size,
                })

        return trades

    def summarize(
        self,
        metrics_list: Optional[List[TradeQualityMetrics]] = None,
        strategy_name: str = "Strategy",
    ) -> TradeQualitySummary:
        """
        生成交易质量汇总统计。

        Args:
            metrics_list: TradeQualityMetrics 列表，如果为 None 则使用实例的 metrics_list。
            strategy_name: 策略名称。

        Returns:
            TradeQualitySummary 对象。
        """
        metrics_list = metrics_list or self.metrics_list

        if not metrics_list:
            return TradeQualitySummary(strategy_name=strategy_name)

        # 转换为 DataFrame 便于统计
        df = pd.DataFrame([m.to_dict() for m in metrics_list])

        # 基础统计
        total_trades = len(metrics_list)
        winning_trades = sum(1 for m in metrics_list if m.is_winner)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0

        # 盈亏统计
        pnls = [m.pnl for m in metrics_list]
        winners = [m.pnl for m in metrics_list if m.is_winner]
        losers = [m.pnl for m in metrics_list if not m.is_winner]

        avg_pnl = np.mean(pnls) if pnls else 0.0
        avg_winner = np.mean(winners) if winners else 0.0
        avg_loser = np.mean(losers) if losers else 0.0

        # 盈亏比
        payoff_ratio = abs(avg_winner / avg_loser) if avg_loser != 0 else float('inf') if avg_winner > 0 else 0.0

        # MAE/MFE 统计
        maes = [m.mae_pct for m in metrics_list]
        mfes = [m.mfe_pct for m in metrics_list]
        efficiencies = [m.efficiency for m in metrics_list]
        hold_days = [m.hold_days for m in metrics_list]

        avg_mae = np.mean(maes) if maes else 0.0
        avg_mfe = np.mean(mfes) if mfes else 0.0
        avg_efficiency = np.mean(efficiencies) if efficiencies else 0.0
        avg_hold_days = np.mean(hold_days) if hold_days else 0.0

        # 按盈亏分组统计 MAE/MFE
        winner_maes = [m.mae_pct for m in metrics_list if m.is_winner]
        loser_maes = [m.mae_pct for m in metrics_list if not m.is_winner]
        winner_mfes = [m.mfe_pct for m in metrics_list if m.is_winner]
        loser_mfes = [m.mfe_pct for m in metrics_list if not m.is_winner]

        avg_mae_winner = np.mean(winner_maes) if winner_maes else 0.0
        avg_mae_loser = np.mean(loser_maes) if loser_maes else 0.0
        avg_mfe_winner = np.mean(winner_mfes) if winner_mfes else 0.0
        avg_mfe_loser = np.mean(loser_mfes) if loser_mfes else 0.0

        # 盈利因子
        total_profit = sum(winners) if winners else 0.0
        total_loss = abs(sum(losers)) if losers else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0.0

        return TradeQualitySummary(
            strategy_name=strategy_name,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_pnl=avg_pnl,
            avg_winner=avg_winner,
            avg_loser=avg_loser,
            payoff_ratio=payoff_ratio,
            avg_mae=avg_mae,
            avg_mfe=avg_mfe,
            avg_mae_winner=avg_mae_winner,
            avg_mae_loser=avg_mae_loser,
            avg_mfe_winner=avg_mfe_winner,
            avg_mfe_loser=avg_mfe_loser,
            avg_efficiency=avg_efficiency,
            avg_hold_days=avg_hold_days,
            profit_factor=profit_factor,
        )

    def plot_mae_mfe_scatter(
        self,
        metrics_list: Optional[List[TradeQualityMetrics]] = None,
        save_path: Optional[str] = None,
        title: str = "MAE/MFE 散点图",
        figsize: tuple = (12, 8),
    ) -> plt.Figure:
        """
        绘制 MAE/MFE 散点图。

        Args:
            metrics_list: TradeQualityMetrics 列表。
            save_path: 保存路径。
            title: 图表标题。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        metrics_list = metrics_list or self.metrics_list

        if not metrics_list:
            logger.warning("没有交易数据可供绘图")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "没有交易数据", ha="center", va="center")
            return fig

        df = pd.DataFrame([
            {
                "MAE (%)": m.mae_pct * 100,
                "MFE (%)": m.mfe_pct * 100,
                "PnL (%)": m.pnl_pct * 100,
                "is_winner": m.is_winner,
                "efficiency": m.efficiency,
            }
            for m in metrics_list
        ])

        fig, ax = plt.subplots(figsize=figsize)

        # 盈利交易
        winners = df[df["is_winner"]]
        losers = df[~df["is_winner"]]

        ax.scatter(
            winners["MAE (%)"],
            winners["MFE (%)"],
            c="green",
            alpha=0.6,
            s=100,
            label=f"盈利交易 ({len(winners)})",
            edgecolors="darkgreen",
        )

        ax.scatter(
            losers["MAE (%)"],
            losers["MFE (%)"],
            c="red",
            alpha=0.6,
            s=100,
            label=f"亏损交易 ({len(losers)})",
            edgecolors="darkred",
        )

        # 添加对角线（MAE = MFE）
        max_val = max(df["MAE (%)"].max(), df["MFE (%)"].max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="MAE = MFE")

        ax.set_xlabel("MAE (%) - 最大不利偏移", fontsize=12)
        ax.set_ylabel("MFE (%) - 最大有利偏移", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        # 添加注释
        ax.annotate(
            "左上区域: 低MAE高MFE = 理想交易\n（抗逆境能力强，机会捕获充分）",
            xy=(0.02, 0.98),
            xycoords="axes fraction",
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"MAE/MFE 散点图已保存: {save_path}")

        return fig

    def plot_efficiency_distribution(
        self,
        metrics_list: Optional[List[TradeQualityMetrics]] = None,
        save_path: Optional[str] = None,
        title: str = "交易效率分布",
        figsize: tuple = (10, 6),
    ) -> plt.Figure:
        """
        绘制交易效率分布直方图。

        Args:
            metrics_list: TradeQualityMetrics 列表。
            save_path: 保存路径。
            title: 图表标题。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        metrics_list = metrics_list or self.metrics_list

        if not metrics_list:
            logger.warning("没有交易数据可供绘图")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "没有交易数据", ha="center", va="center")
            return fig

        efficiencies = [m.efficiency for m in metrics_list]
        # 限制效率范围在 [-1, 2] 便于显示
        efficiencies = np.clip(efficiencies, -1, 2)

        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(efficiencies, bins=30, edgecolor="black", alpha=0.7, color="steelblue")

        # 添加参考线
        ax.axvline(x=0, color="red", linestyle="--", label="盈亏平衡 (0%)")
        ax.axvline(x=1, color="green", linestyle="--", label="完美效率 (100%)")

        # 统计信息
        avg_eff = np.mean(efficiencies)
        ax.axvline(x=avg_eff, color="orange", linestyle="-", linewidth=2, label=f"平均效率: {avg_eff:.1%}")

        ax.set_xlabel("交易效率 (实际盈亏 / MFE)", fontsize=12)
        ax.set_ylabel("交易次数", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")

        # 添加注释
        ax.text(
            0.02, 0.98,
            "效率 = 0: 盈亏平衡\n效率 = 1: 捕获全部有利波动\n效率 < 0: 亏损交易\n效率 > 1: 超额收益",
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.5),
        )

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"交易效率分布图已保存: {save_path}")

        return fig

    def plot_rolling_win_rate(
        self,
        metrics_list: Optional[List[TradeQualityMetrics]] = None,
        window: int = 20,
        save_path: Optional[str] = None,
        title: str = "滚动胜率曲线",
        figsize: tuple = (12, 6),
    ) -> plt.Figure:
        """
        绘制滚动胜率曲线。

        Args:
            metrics_list: TradeQualityMetrics 列表。
            window: 滚动窗口大小。
            save_path: 保存路径。
            title: 图表标题。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        metrics_list = metrics_list or self.metrics_list

        if not metrics_list:
            logger.warning("没有交易数据可供绘图")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "没有交易数据", ha="center", va="center")
            return fig

        # 创建时间序列
        df = pd.DataFrame([
            {
                "exit_date": m.exit_date,
                "is_winner": 1 if m.is_winner else 0,
            }
            for m in metrics_list
        ])
        df = df.sort_values("exit_date")
        df.set_index("exit_date", inplace=True)

        # 计算滚动胜率
        df["rolling_win_rate"] = df["is_winner"].rolling(window=window, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(df.index, df["rolling_win_rate"], linewidth=2, color="steelblue", label=f"{window}笔滚动胜率")
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% 参考线")

        # 平均胜率
        avg_win_rate = df["is_winner"].mean()
        ax.axhline(y=avg_win_rate, color="green", linestyle="-", linewidth=2, label=f"总体胜率: {avg_win_rate:.1%}")

        ax.set_xlabel("交易日期", fontsize=12)
        ax.set_ylabel("胜率", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"滚动胜率曲线已保存: {save_path}")

        return fig

    def compare_strategies(
        self,
        strategies_metrics: Dict[str, List[TradeQualityMetrics]],
        save_path: Optional[str] = None,
        figsize: tuple = (14, 10),
    ) -> plt.Figure:
        """
        对比多策略的交易质量。

        Args:
            strategies_metrics: 策略名称到 TradeQualityMetrics 列表的映射。
            save_path: 保存路径。
            figsize: 图表尺寸。

        Returns:
            matplotlib Figure 对象。
        """
        if not strategies_metrics:
            logger.warning("没有策略数据可供对比")
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, "没有策略数据", ha="center", va="center")
            return fig

        # 计算各策略汇总
        summaries = {}
        for name, metrics_list in strategies_metrics.items():
            summaries[name] = self.summarize(metrics_list, strategy_name=name)

        # 创建对比图
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. 胜率对比
        ax1 = axes[0, 0]
        names = list(summaries.keys())
        win_rates = [s.win_rate for s in summaries.values()]
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        bars = ax1.bar(names, win_rates, color=colors, edgecolor="black")
        ax1.axhline(y=0.5, color="red", linestyle="--", alpha=0.5)
        ax1.set_ylabel("胜率")
        ax1.set_title("胜率对比", fontweight="bold")
        ax1.set_xticklabels(names, rotation=45, ha="right")
        for bar, rate in zip(bars, win_rates):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f"{rate:.1%}", ha="center", va="bottom", fontsize=9)

        # 2. 盈亏比对比
        ax2 = axes[0, 1]
        payoff_ratios = [s.payoff_ratio if s.payoff_ratio != float('inf') else 10 for s in summaries.values()]
        ax2.bar(names, payoff_ratios, color=colors, edgecolor="black")
        ax2.axhline(y=1, color="red", linestyle="--", alpha=0.5)
        ax2.set_ylabel("盈亏比")
        ax2.set_title("盈亏比对比", fontweight="bold")
        ax2.set_xticklabels(names, rotation=45, ha="right")

        # 3. MAE/MFE 对比
        ax3 = axes[1, 0]
        maes = [s.avg_mae for s in summaries.values()]
        mfes = [s.avg_mfe for s in summaries.values()]
        x = np.arange(len(names))
        width = 0.35
        ax3.bar(x - width / 2, maes, width, label="平均 MAE", color="indianred", edgecolor="black")
        ax3.bar(x + width / 2, mfes, width, label="平均 MFE", color="seagreen", edgecolor="black")
        ax3.set_ylabel("百分比")
        ax3.set_title("MAE vs MFE 对比", fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, rotation=45, ha="right")
        ax3.legend()

        # 4. 交易效率对比
        ax4 = axes[1, 1]
        efficiencies = [s.avg_efficiency for s in summaries.values()]
        ax4.bar(names, efficiencies, color=colors, edgecolor="black")
        ax4.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax4.axhline(y=1, color="green", linestyle="--", alpha=0.5, label="完美效率")
        ax4.set_ylabel("交易效率")
        ax4.set_title("平均交易效率对比", fontweight="bold")
        ax4.set_xticklabels(names, rotation=45, ha="right")

        plt.suptitle("多策略交易质量对比", fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"策略对比图已保存: {save_path}")

        return fig


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("  交易质量分析器示例")
    print("=" * 60)

    # 创建模拟交易日志
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

    trade_log = pd.DataFrame({
        "date": dates[::5],
        "type": ["买入", "卖出"] * 10,
        "price": 100 + np.cumsum(np.random.randn(20) * 2),
        "size": [100] * 20,
    })

    # 创建模拟 OHLCV 数据
    ohlcv_data = pd.DataFrame({
        "open": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "high": 100 + np.cumsum(np.random.randn(100) * 0.5) + 2,
        "low": 100 + np.cumsum(np.random.randn(100) * 0.5) - 2,
        "close": 100 + np.cumsum(np.random.randn(100) * 0.5),
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)

    # 分析交易
    analyzer = TradeAnalyzer()
    metrics = analyzer.analyze_trades(trade_log, ohlcv_data)
    summary = analyzer.summarize(metrics, strategy_name="示例策略")

    print(summary.summary())