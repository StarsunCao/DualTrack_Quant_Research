"""
Backtrader 回测引擎模块。

将双轨融合信号接入 Backtrader 回测框架进行策略回测。

注意：Python 3.10+ 需要修复 collections 兼容性问题。
"""

# ============================================================================
# Python 3.10+ 兼容性修复 (Monkey Patch)
# ============================================================================
import collections
import collections.abc

# 修复 Backtrader 在 Python 3.10+ 中的兼容性问题
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
collections.Callable = collections.abc.Callable

# ============================================================================
# 正式导入
# ============================================================================
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Optional, Union
from pathlib import Path

# 导入策略类和配置
from .a_share_strategy import AShareStrategy
from .us_market_strategy import USMarketStrategy
from .a_share_sizer import AShareCommission
from .us_market_sizer import USMarketCommission
from ..config.market_config import MarketConfig, MarketType


@dataclass
class BacktestResult:
    """
    回测结果数据类。

    Attributes:
        initial_cash: 初始资金。
        final_value: 最终资产价值。
        total_return: 总收益率。
        annual_return: 年化收益率。
        sharpe_ratio: 夏普比率。
        max_drawdown: 最大回撤。
        max_drawdown_len: 最大回撤持续期。
        equity_curve: 资产净值曲线 (DataFrame)。
        trades: 交易记录 (DataFrame)。
        analyzers: 分析器结果字典。
        trade_details: 详细交易记录 (DataFrame)。
        position_details: 持仓记录 (DataFrame)。
        rebalance_details: 调仓记录 (DataFrame)。
        config: 实验配置字典。
        git_commit_hash: Git commit hash 用于复现。
        timestamp: 结果保存时间戳。
    """
    initial_cash: float
    final_value: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_len: int
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    analyzers: dict = field(default_factory=dict)
    trade_details: pd.DataFrame = field(default_factory=pd.DataFrame)
    position_details: pd.DataFrame = field(default_factory=pd.DataFrame)
    rebalance_details: pd.DataFrame = field(default_factory=pd.DataFrame)
    config: dict = field(default_factory=dict)
    git_commit_hash: str = field(default_factory=lambda: BacktestResult._get_git_commit())
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @staticmethod
    def _get_git_commit() -> str:
        """获取当前 Git commit hash。"""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "unknown"

    def to_dict(self) -> dict:
        """转换为字典格式。"""
        return {
            "initial_cash": self.initial_cash,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_len": self.max_drawdown_len,
            "config": self.config,
            "git_commit_hash": self.git_commit_hash,
            "timestamp": self.timestamp,
        }

    def summary(self) -> str:
        """生成回测结果摘要字符串。"""
        lines = [
            "=" * 60,
            "  回测结果摘要",
            "=" * 60,
            f"  初始资金: {self.initial_cash:,.2f}",
            f"  最终资产: {self.final_value:,.2f}",
            f"  总收益率: {self.total_return:.2%}",
            f"  年化收益率: {self.annual_return:.2%}",
            "-" * 60,
            f"  夏普比率: {self.sharpe_ratio:.4f}",
            f"  最大回撤: {self.max_drawdown:.2%}",
            f"  回撤持续期: {self.max_drawdown_len} 天",
            "=" * 60,
        ]
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> Path:
        """
        保存回测结果到文件。

        保存内容包括：
        - 回测指标（JSON格式）
        - 资产净值曲线（Parquet格式）
        - 交易记录（Parquet格式）
        - 持仓记录（Parquet格式）
        - 调仓记录（Parquet格式）
        - 完整结果（Pickle格式，用于完整恢复）

        Args:
            path: 保存路径前缀或目录

        Returns:
            保存的目录路径
        """
        save_path = Path(path)
        if save_path.suffix:  # 如果有后缀，视为文件路径，取其目录
            save_path = save_path.parent / save_path.stem
        save_path.mkdir(parents=True, exist_ok=True)

        # 1. 保存回测指标和元数据（JSON）
        with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        # 2. 保存资产净值曲线（Parquet格式，带索引）
        if not self.equity_curve.empty:
            self.equity_curve.to_parquet(save_path / "equity_curve.parquet")

        # 3. 保存交易记录
        if not self.trade_details.empty:
            self.trade_details.to_parquet(save_path / "trade_details.parquet")
        if not self.trades.empty:
            self.trades.to_parquet(save_path / "trades.parquet")

        # 4. 保存持仓记录
        if not self.position_details.empty:
            self.position_details.to_parquet(save_path / "position_details.parquet")

        # 5. 保存调仓记录
        if not self.rebalance_details.empty:
            self.rebalance_details.to_parquet(save_path / "rebalance_details.parquet")

        # 6. 保存分析器结果（JSON）
        # 需要将分析器结果转换为可序列化的格式
        serializable_analyzers = {}
        for key, value in self.analyzers.items():
            if isinstance(value, dict):
                serializable_analyzers[key] = value
            else:
                # 尝试转换为字典
                try:
                    serializable_analyzers[key] = dict(value)
                except (TypeError, ValueError):
                    serializable_analyzers[key] = str(value)

        with open(save_path / "analyzers.json", "w", encoding="utf-8") as f:
            json.dump(serializable_analyzers, f, indent=2, ensure_ascii=False, default=str)

        # 7. 保存完整结果（Pickle格式，用于完整恢复）
        import pickle
        with open(save_path / "result.pkl", "wb") as f:
            pickle.dump(self, f)

        print(f"回测结果已保存到: {save_path}")
        return save_path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BacktestResult":
        """
        从文件加载回测结果。

        优先尝试加载 Pickle 格式的完整结果，
        如果失败则从各个组件重新构建。

        Args:
            path: 保存路径前缀或目录

        Returns:
            BacktestResult 实例
        """
        load_path = Path(path)
        if load_path.suffix == ".pkl":
            # 直接加载 Pickle 文件
            import pickle
            with open(load_path, "rb") as f:
                return pickle.load(f)

        # 从目录加载
        if not load_path.exists():
            raise FileNotFoundError(f"回测结果目录不存在: {load_path}")

        # 1. 尝试加载 Pickle 完整结果
        pickle_path = load_path / "result.pkl"
        if pickle_path.exists():
            import pickle
            with open(pickle_path, "rb") as f:
                return pickle.load(f)

        # 2. 从各个组件重新构建
        # 加载元数据
        with open(load_path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # 创建结果对象
        result = cls(
            initial_cash=metadata["initial_cash"],
            final_value=metadata["final_value"],
            total_return=metadata["total_return"],
            annual_return=metadata["annual_return"],
            sharpe_ratio=metadata["sharpe_ratio"],
            max_drawdown=metadata["max_drawdown"],
            max_drawdown_len=metadata["max_drawdown_len"],
            config=metadata.get("config", {}),
            git_commit_hash=metadata.get("git_commit_hash", "unknown"),
            timestamp=metadata.get("timestamp", ""),
        )

        # 加载 DataFrame 组件
        equity_path = load_path / "equity_curve.parquet"
        if equity_path.exists():
            result.equity_curve = pd.read_parquet(equity_path)

        trades_path = load_path / "trades.parquet"
        if trades_path.exists():
            result.trades = pd.read_parquet(trades_path)

        trade_details_path = load_path / "trade_details.parquet"
        if trade_details_path.exists():
            result.trade_details = pd.read_parquet(trade_details_path)

        position_path = load_path / "position_details.parquet"
        if position_path.exists():
            result.position_details = pd.read_parquet(position_path)

        rebalance_path = load_path / "rebalance_details.parquet"
        if rebalance_path.exists():
            result.rebalance_details = pd.read_parquet(rebalance_path)

        # 加载分析器结果
        analyzers_path = load_path / "analyzers.json"
        if analyzers_path.exists():
            with open(analyzers_path, "r", encoding="utf-8") as f:
                result.analyzers = json.load(f)

        return result


class PandasDataFeed(bt.feeds.PandasData):
    """
    自定义 Pandas Data Feed。

    用于加载我们的 OHLCV 数据到 Backtrader。

    使用方法:
        data = PandasDataFeed(
            dataname=df,  # DataFrame with OHLCV columns
            datetime=None,  # 使用索引作为时间
        )
    """

    params = (
        ("datetime", None),  # 使用索引
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", None),  # 不使用持仓量
    )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str = "",
        **kwargs,
    ) -> "PandasDataFeed":
        """
        从 DataFrame 创建 Data Feed。

        Args:
            df: 包含 OHLCV 数据的 DataFrame，索引应为日期。
            name: 数据名称。
            **kwargs: 其他参数。

        Returns:
            PandasDataFeed 实例。
        """
        # 确保索引为 datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.copy()
            df.index = pd.to_datetime(df.index)

        return cls(dataname=df, name=name, **kwargs)


class DualTrackStrategy(bt.Strategy):
    """
    双轨策略类。

    接收外部计算好的目标仓位字典，在 Backtrader 中执行调仓。

    参数:
        target_positions: 目标仓位字典，格式为 {datetime: {symbol: weight}}
        rebalance_freq: 调仓频率（天数），默认为 1（每天调仓）
        printlog: 是否打印日志
        allow_short: 是否允许做空，默认 False（A股规则）
        short_confidence_threshold: 触发做空所需的最低置信度，默认 0.85
    """

    params = (
        ("target_positions", None),  # 目标仓位字典
        ("rebalance_freq", 1),  # 调仓频率
        ("printlog", True),  # 打印日志
        ("allow_short", False),  # 是否允许做空（美股为True）
        ("short_confidence_threshold", 0.85),  # 做空所需的最低置信度
    )

    def __init__(self) -> None:
        """初始化策略。"""
        self.dataclose = self.datas[0].close
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.trade_count = 0
        self.last_rebalance = None

        # 记录每日资产价值
        self.equity_curve: list[dict] = []

        # 新增：详细交易记录
        self.trade_records: list[dict] = []  # 交易记录
        self.position_records: list[dict] = []  # 每日持仓记录
        self.rebalance_records: list[dict] = []  # 调仓记录

    def log(self, txt: str, dt: Optional[datetime] = None) -> None:
        """打印日志。"""
        if self.params.printlog:
            dt = dt or self.datas[0].datetime.date(0)
            print(f"  [{dt.isoformat()}] {txt}")

    def notify_order(self, order: bt.Order) -> None:
        """订单状态通知 - 增强版，记录详细交易信息。"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0)
            record = {
                "date": dt.isoformat(),
                "type": "买入" if order.isbuy() else "卖出",
                "price": order.executed.price,
                "size": order.executed.size,
                "value": order.executed.value,
                "commission": order.executed.comm,
                "symbol": self.datas[0]._name or "CSI300",
            }
            self.trade_records.append(record)

            if order.isbuy():
                self.log(
                    f"买入执行: 价格={order.executed.price:.2f}, "
                    f"数量={order.executed.size:.0f}, "
                    f"成本={order.executed.value:.2f}, "
                    f"手续费={order.executed.comm:.2f}"
                )
            else:
                self.log(
                    f"卖出执行: 价格={order.executed.price:.2f}, "
                    f"数量={order.executed.size:.0f}, "
                    f"成本={order.executed.value:.2f}, "
                    f"手续费={order.executed.comm:.2f}"
                )
            self.trade_count += 1

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("订单取消/保证金不足/拒绝")

        self.order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """交易通知。"""
        if not trade.isclosed:
            return

        self.log(
            f"交易盈亏: 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}"
        )

    def next(self) -> None:
        """每个 bar 执行的逻辑 - 增强版，记录持仓和调仓。"""
        current_date = self.datas[0].datetime.date(0)
        current_value = self.broker.getvalue()

        # 记录每日资产价值
        self.equity_curve.append({
            "date": current_date,
            "value": current_value,
            "cash": self.broker.getcash(),
        })

        # 记录每日持仓详情
        pos = self.getposition(self.datas[0])
        position_record = {
            "date": current_date.isoformat(),
            "symbol": self.datas[0]._name or "CSI300",
            "position_size": pos.size if pos else 0,
            "position_value": pos.size * self.dataclose[0] if pos else 0,
            "cash": self.broker.getcash(),
            "total_value": current_value,
            "close_price": self.dataclose[0],
        }
        self.position_records.append(position_record)

        # 检查是否有未完成订单
        if self.order:
            return

        # 获取目标仓位
        target_positions = self.params.target_positions
        if not target_positions:
            return

        # 查找当前日期对应的目标仓位
        current_dt = self.datas[0].datetime.datetime(0)

        # 尝试匹配日期
        target = None
        for date_key, positions in target_positions.items():
            if isinstance(date_key, str):
                date_key = pd.to_datetime(date_key)

            # 比较日期部分
            if hasattr(date_key, 'date'):
                key_date = date_key.date()
            else:
                key_date = pd.to_datetime(date_key).date()

            if key_date == current_date:
                target = positions
                break

        if not target:
            return

        # 检查调仓频率
        if self.last_rebalance is not None:
            # 直接计算天数差，避免类型转换问题
            try:
                days_since_last = (current_date - self.last_rebalance).days
                if days_since_last < self.params.rebalance_freq:
                    return
            except (AttributeError, TypeError):
                # 如果日期类型不兼容，跳过频率检查
                pass

        self.last_rebalance = current_date

        # 记录调仓信息
        for data in self.datas:
            symbol = data._name if hasattr(data, '_name') else "default"
            if symbol in target:
                pos = self.getposition(data)
                current_position_value = pos.size * data.close[0] if pos else 0
                current_weight = current_position_value / current_value if current_value > 0 else 0

                rebalance_record = {
                    "date": current_date.isoformat(),
                    "symbol": symbol,
                    "target_weight": target[symbol],
                    "current_weight": current_weight,
                    "portfolio_value": current_value,
                    "close_price": data.close[0],
                }
                self.rebalance_records.append(rebalance_record)

        # 执行调仓（根据allow_short参数决定是否允许做空）
        self.log(f"执行调仓, 目标仓位: {target}, 允许做空: {self.params.allow_short}")

        # 对每个数据源进行调仓
        for data in self.datas:
            symbol = data._name if hasattr(data, '_name') else "default"

            if symbol in target:
                weight = target[symbol]

                # 处理负权重（sell信号）
                if weight < 0:
                    if self.params.allow_short:
                        # 美股规则：需要检查置信度是否足够高才做空
                        confidence = abs(weight)  # 负权重的绝对值即为置信度
                        if confidence >= self.params.short_confidence_threshold:
                            # 高置信度sell信号：触发做空
                            self.log(f"  {symbol}: 高置信度做空 {weight:.2%}（置信度>{self.params.short_confidence_threshold:.0%}）")
                            target_value = current_value * weight
                            target_size = int(target_value / data.close[0])
                            self.order_target_size(data=data, target=target_size)
                            continue
                        else:
                            # 低置信度sell信号：仅清仓/减仓，不做空
                            self.log(f"  {symbol}: 低置信度SELL {weight:.2%} → 清仓（不做空，置信度<{self.params.short_confidence_threshold:.0%}）")
                            weight = 0.0  # 清仓
                    else:
                        # A股规则：SELL信号（负权重）→ 分级减仓而非做空
                        # 负权重转换为"保留仓位"
                        # weight = -0.95 → 保留仓位 = 1 + (-0.95) = 0.05 (5%)
                        # weight = -0.70 → 保留仓位 = 1 + (-0.70) = 0.30 (30%)
                        # 置信度越高，保留仓位越少（减仓越多）
                        target_weight = 1 + weight  # 负权重转正
                        target_weight = max(0.0, target_weight)  # 确保非负
                        self.log(f"  {symbol}: SELL信号 {weight:.2%} → 保留仓位 {target_weight:.2%}（减仓{-weight:.2%}）")
                        weight = target_weight

                self.log(f"  {symbol}: 调整仓位至 {weight:.2%}")
                self.order_target_percent(data=data, target=weight)
            else:
                # 如果不在目标仓位中，清仓
                self.log(f"  {symbol}: 清仓")
                self.order_target_percent(data=data, target=0.0)

    def stop(self) -> None:
        """策略结束时调用。"""
        final_value = self.broker.getvalue()
        self.log(f"策略结束, 最终资产: {final_value:,.2f}", dt=self.datas[0].datetime.date(0))


class BacktestEngine:
    """
    回测引擎类。

    封装 Backtrader 的 Cerebro 引擎，简化回测配置和执行。

    使用方法:
        engine = BacktestEngine(initial_cash=100000)
        engine.add_data(datafeed)
        engine.add_strategy(DualTrackStrategy, target_positions=positions)
        result = engine.run()
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission: float = 0.0002,  # 万分之二
        slippage_perc: float = 0.0,
        stamp_duty: float = 0.0,
    ) -> None:
        """
        初始化回测引擎。

        Args:
            initial_cash: 初始资金，默认 100,000。
            commission: 佣金率，默认万分之二。
            slippage_perc: 滑点百分比，默认 0。
            stamp_duty: 印花税率（仅卖出），默认 0。
        """
        self.cerebro = bt.Cerebro()
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage_perc = slippage_perc
        self.stamp_duty = stamp_duty

        # 设置初始资金
        self.cerebro.broker.setcash(initial_cash)

        # 设置佣金
        self.cerebro.broker.setcommission(commission=commission)

        # 设置滑点
        if slippage_perc > 0:
            self.cerebro.broker.set_slippage_perc(perc=slippage_perc)

        # 数据存储
        self.datas: list[bt.feeds.PandasData] = []
        self.strategies: list = []
        self._result = None

    def add_data(
        self,
        data: Union[pd.DataFrame, bt.feeds.PandasData],
        name: str = "",
    ) -> None:
        """
        添加数据源。

        Args:
            data: DataFrame 或 PandasData 实例。
            name: 数据名称。
        """
        if isinstance(data, pd.DataFrame):
            data = PandasDataFeed.from_dataframe(data, name=name)
        elif isinstance(data, bt.feeds.PandasData):
            if name:
                data._name = name

        self.datas.append(data)
        self.cerebro.adddata(data)

    def add_strategy(self, strategy: bt.Strategy, **kwargs) -> None:
        """
        添加策略。

        Args:
            strategy: 策略类。
            **kwargs: 策略参数。
        """
        self.cerebro.addstrategy(strategy, **kwargs)
        self.strategies.append(strategy)

    def add_analyzer(
        self,
        analyzer: bt.Analyzer,
        **kwargs,
    ) -> None:
        """
        添加分析器。

        Args:
            analyzer: 分析器类。
            **kwargs: 分析器参数。
        """
        self.cerebro.addanalyzer(analyzer, **kwargs)

    def setup_default_analyzers(self) -> None:
        """设置默认分析器。"""
        # TimeReturn - 时间序列收益率
        self.cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="time_return")

        # SharpeRatio - 夏普比率
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe_ratio")

        # DrawDown - 回撤分析
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")

        # Returns - 收益率分析
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")

        # TradeAnalyzer - 交易分析
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

    def run(self) -> BacktestResult:
        """
        执行回测。

        Returns:
            BacktestResult 对象，包含回测结果。
        """
        print("=" * 60)
        print("  开始执行回测")
        print("=" * 60)

        # 设置默认分析器
        self.setup_default_analyzers()

        # 记录开始时间
        start_time = datetime.now()

        # 执行回测
        results = self.cerebro.run()
        self._result = results[0] if results else None

        # 计算耗时
        elapsed = (datetime.now() - start_time).total_seconds()

        # 提取结果
        final_value = self.cerebro.broker.getvalue()
        total_return = (final_value - self.initial_cash) / self.initial_cash

        # 提取分析器结果
        analyzers = {}
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        max_drawdown_len = 0

        if self._result is not None:
            # 夏普比率
            sharpe_analysis = self._result.analyzers.sharpe_ratio.get_analysis()
            sharpe_ratio = sharpe_analysis.get("sharperatio", 0.0) or 0.0

            # 回撤
            dd_analysis = self._result.analyzers.drawdown.get_analysis()
            max_drawdown = dd_analysis.get("max", {}).get("drawdown", 0.0) / 100
            max_drawdown_len = dd_analysis.get("max", {}).get("len", 0)

            # 时间序列收益率
            time_return = self._result.analyzers.time_return.get_analysis()
            analyzers["time_return"] = time_return

            # 收益率
            returns = self._result.analyzers.returns.get_analysis()
            analyzers["returns"] = returns

            # 交易分析
            trades = self._result.analyzers.trades.get_analysis()
            analyzers["trades"] = trades

        # 计算年化收益率
        # 假设回测周期为数据天数
        if self.datas:
            first_date = self.datas[0].fromdate
            last_date = self.datas[0].todate
            if first_date and last_date:
                try:
                    days = (last_date - first_date).days
                    years = max(days / 365, 1)
                    annual_return = (1 + total_return) ** (1 / years) - 1
                except (AttributeError, TypeError):
                    # 如果日期类型不兼容，使用简化计算
                    annual_return = total_return
            else:
                annual_return = total_return
        else:
            annual_return = total_return

        # 提取资产净值曲线
        equity_curve = pd.DataFrame()
        if self._result is not None and hasattr(self._result, "equity_curve"):
            equity_curve = pd.DataFrame(self._result.equity_curve)
            if not equity_curve.empty:
                equity_curve["date"] = pd.to_datetime(equity_curve["date"])
                equity_curve.set_index("date", inplace=True)

        # 提取详细记录
        trade_details = pd.DataFrame()
        position_details = pd.DataFrame()
        rebalance_details = pd.DataFrame()

        if self._result is not None:
            if hasattr(self._result, "trade_records"):
                trade_details = pd.DataFrame(self._result.trade_records)
            if hasattr(self._result, "position_records"):
                position_details = pd.DataFrame(self._result.position_records)
            if hasattr(self._result, "rebalance_records"):
                rebalance_details = pd.DataFrame(self._result.rebalance_records)

        # 构建结果对象
        result = BacktestResult(
            initial_cash=self.initial_cash,
            final_value=final_value,
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_len=max_drawdown_len,
            equity_curve=equity_curve,
            trades=trade_details,
            analyzers=analyzers,
        )

        # 附加详细记录
        result.trade_details = trade_details
        result.position_details = position_details
        result.rebalance_details = rebalance_details

        print(f"\n  回测耗时: {elapsed:.2f} 秒")
        print(result.summary())

        return result

    def plot(self, **kwargs) -> None:
        """
        绘制回测结果图表。

        Args:
            **kwargs: 绘图参数。
        """
        self.cerebro.plot(**kwargs)


def get_strategy_for_market(symbol: str):
    """
    根据股票代码获取策略类。

    Args:
        symbol: 股票代码

    Returns:
        对应的策略类（AShareStrategy 或 USMarketStrategy）
    """
    market_type = MarketConfig.get_market_type_for_symbol(symbol)

    if market_type == MarketType.US_MARKET:
        return USMarketStrategy
    else:
        return AShareStrategy


def get_commission_for_market(symbol: str):
    """
    根据股票代码获取佣金计算器类。

    Args:
        symbol: 股票代码

    Returns:
        对应的佣金计算器类（AShareCommission 或 USMarketCommission）
    """
    market_type = MarketConfig.get_market_type_for_symbol(symbol)

    if market_type == MarketType.US_MARKET:
        return USMarketCommission()
    else:
        return AShareCommission()


def run_backtest(
    ohlcv_data: pd.DataFrame,
    target_positions: dict,
    initial_cash: float = 100000.0,
    commission: float = 0.0002,
    strategy_params: Optional[dict] = None,
    symbol: str = "CSI300",
) -> BacktestResult:
    """
    快速回测函数。

    提供简化的接口执行单资产回测，自动根据symbol选择策略和费用计算器。

    Args:
        ohlcv_data: OHLCV 数据 DataFrame，索引为日期。
        target_positions: 目标仓位字典，格式 {datetime: {symbol: weight}}。
        initial_cash: 初始资金。
        commission: 佣金率（仅A股使用，美股忽略此参数）。
        strategy_params: 策略额外参数。
        symbol: 股票代码（用于自动选择策略）。

    Returns:
        BacktestResult 对象。
    """
    strategy_params = strategy_params or {}

    # 获取市场配置
    market_config = MarketConfig.get_config_for_symbol(symbol)

    # 创建引擎（根据市场类型设置费用）
    if market_config.market_type == MarketType.US_MARKET:
        # 美股：使用美股佣金计算器
        engine = BacktestEngine(
            initial_cash=initial_cash,
            commission=0.0,  # 美股零佣金，实际费用由 USMarketCommission 计算
        )
        # 添加美股佣金计算器（包含SEC费）
        engine.cerebro.broker.addcommissioninfo(USMarketCommission())
    else:
        # A股：使用A股佣金计算器
        engine = BacktestEngine(
            initial_cash=initial_cash,
            commission=commission,
        )
        # 添加A股佣金计算器（包含印花税）
        engine.cerebro.broker.addcommissioninfo(AShareCommission())

    # 添加数据
    engine.add_data(ohlcv_data, name=symbol)

    # 自动选择策略
    strategy_class = get_strategy_for_market(symbol)

    # 添加策略
    engine.add_strategy(
        strategy_class,
        target_positions=target_positions,
        **strategy_params,
    )

    # 执行回测
    return engine.run()


if __name__ == "__main__":
    # 示例用法
    print("=" * 60)
    print("  Backtrader 回测引擎示例")
    print("=" * 60)

    # 创建示例 OHLCV 数据
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")

    base_price = 100
    returns = np.random.randn(100) * 0.02
    prices = base_price * (1 + returns).cumprod()

    sample_data = pd.DataFrame({
        "open": prices * (1 + np.random.randn(100) * 0.003),
        "high": prices * (1 + np.abs(np.random.randn(100)) * 0.008),
        "low": prices * (1 - np.abs(np.random.randn(100)) * 0.008),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)

    # 确保价格合理
    sample_data["high"] = sample_data[["open", "high", "close"]].max(axis=1)
    sample_data["low"] = sample_data[["open", "low", "close"]].min(axis=1)

    # 创建目标仓位字典
    # 简单策略：每 10 天切换一次仓位
    target_positions = {}
    for i, date in enumerate(dates):
        if i % 20 < 10:
            target_positions[date] = {"ASSET": 0.8}  # 80% 仓位
        else:
            target_positions[date] = {"ASSET": 0.2}  # 20% 仓位

    # 执行回测
    result = run_backtest(
        ohlcv_data=sample_data,
        target_positions=target_positions,
        initial_cash=100000,
    )

    # 打印资产净值曲线
    print("\n资产净值曲线 (前10天):")
    if not result.equity_curve.empty:
        print(result.equity_curve.head(10).to_string())