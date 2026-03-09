"""
特征工程模块。

基于 OHLCV 数据计算基础技术因子，包括动量、均线交叉、RSI、MACD 等。
"""

from typing import Optional

import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    特征工程类。

    基于 OHLCV 数据计算各类技术分析因子，用于机器学习模型训练。

    Attributes:
        feature_names: 计算后的特征名称列表。
    """

    def __init__(self) -> None:
        """初始化特征工程器。"""
        self.feature_names: list[str] = []

    @staticmethod
    def _validate_ohlcv(df: pd.DataFrame) -> None:
        """
        验证输入数据是否包含必要的 OHLCV 列。

        Args:
            df: 输入的 DataFrame。

        Raises:
            ValueError: 当缺少必要列时抛出。
        """
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"输入数据缺少必要的列: {missing_cols}")

    @staticmethod
    def compute_returns(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
        """
        计算收益率因子。

        Args:
            df: 包含 'close' 列的 DataFrame。
            periods: 收益率计算周期列表，默认为 [1, 5, 10, 20]。

        Returns:
            添加了收益率列的 DataFrame。
        """
        periods = periods or [1, 5, 10, 20]
        result = df.copy()

        for period in periods:
            result[f"return_{period}d"] = result["close"].pct_change(period)

        return result

    @staticmethod
    def compute_momentum(
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        计算动量因子（价格变化率）。

        动量 = 价格 / N天前价格 - 1

        Args:
            df: 包含 'close' 列的 DataFrame。
            windows: 动量计算窗口列表，默认为 [5, 10, 20]。

        Returns:
            添加了动量列的 DataFrame。
        """
        windows = windows or [5, 10, 20]
        result = df.copy()

        for window in windows:
            result[f"momentum_{window}d"] = result["close"] / result["close"].shift(window) - 1

        return result

    @staticmethod
    def compute_ma_cross(
        df: pd.DataFrame,
        short_windows: list[int] | None = None,
        long_windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        计算均线交叉因子。

        计算短期均线与长期均线的比值，大于1表示短期均线在长期均线上方。

        Args:
            df: 包含 'close' 列的 DataFrame。
            short_windows: 短期均线窗口列表，默认为 [5, 10]。
            long_windows: 长期均线窗口列表，默认为 [20, 60]。

        Returns:
            添加了均线交叉列的 DataFrame。
        """
        short_windows = short_windows or [5, 10]
        long_windows = long_windows or [20, 60]
        result = df.copy()

        # 计算各周期均线
        all_windows = set(short_windows + long_windows)
        for window in all_windows:
            result[f"ma_{window}"] = result["close"].rolling(window=window).mean()

        # 计算交叉因子
        for short in short_windows:
            for long in long_windows:
                if short < long:
                    result[f"ma_cross_{short}_{long}"] = (
                        result[f"ma_{short}"] / result[f"ma_{long}"] - 1
                    )

        return result

    @staticmethod
    def compute_rsi(
        df: pd.DataFrame,
        periods: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        计算相对强弱指标 (RSI)。

        RSI = 100 - 100 / (1 + RS)
        RS = 平均上涨幅度 / 平均下跌幅度

        Args:
            df: 包含 'close' 列的 DataFrame。
            periods: RSI 计算周期列表，默认为 [6, 14, 24]。

        Returns:
            添加了 RSI 列的 DataFrame。
        """
        periods = periods or [6, 14, 24]
        result = df.copy()

        delta = result["close"].diff()

        for period in periods:
            gain = delta.where(delta > 0, 0.0)
            loss = (-delta).where(delta < 0, 0.0)

            # 使用指数移动平均
            avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
            avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

            rs = avg_gain / avg_loss.replace(0, np.inf)
            result[f"rsi_{period}"] = 100 - 100 / (1 + rs)

        return result

    @staticmethod
    def compute_macd(
        df: pd.DataFrame,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> pd.DataFrame:
        """
        计算 MACD (Moving Average Convergence Divergence)。

        MACD = EMA(fast) - EMA(slow)
        Signal = EMA(MACD, signal_period)
        Histogram = MACD - Signal

        Args:
            df: 包含 'close' 列的 DataFrame。
            fast_period: 快线周期，默认12。
            slow_period: 慢线周期，默认26。
            signal_period: 信号线周期，默认9。

        Returns:
            添加了 MACD 相关列的 DataFrame。
        """
        result = df.copy()

        # 计算 EMA
        ema_fast = result["close"].ewm(span=fast_period, adjust=False).mean()
        ema_slow = result["close"].ewm(span=slow_period, adjust=False).mean()

        # MACD 线
        result["macd"] = ema_fast - ema_slow

        # 信号线
        result["macd_signal"] = result["macd"].ewm(span=signal_period, adjust=False).mean()

        # MACD 柱状图
        result["macd_histogram"] = result["macd"] - result["macd_signal"]

        # MACD 斜率
        result["macd_slope"] = result["macd"].diff()

        return result

    @staticmethod
    def compute_bollinger_bands(
        df: pd.DataFrame,
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """
        计算布林带因子。

        中轨 = N日移动平均
        上轨 = 中轨 + N × 标准差
        下轨 = 中轨 - N × 标准差
        带宽 = (上轨 - 下轨) / 中轨
        位置 = (价格 - 下轨) / (上轨 - 下轨)

        Args:
            df: 包含 'close' 列的 DataFrame。
            window: 计算窗口，默认20。
            num_std: 标准差倍数，默认2.0。

        Returns:
            添加了布林带相关列的 DataFrame。
        """
        result = df.copy()

        # 中轨
        result["bb_middle"] = result["close"].rolling(window=window).mean()

        # 标准差
        rolling_std = result["close"].rolling(window=window).std()

        # 上下轨
        result["bb_upper"] = result["bb_middle"] + num_std * rolling_std
        result["bb_lower"] = result["bb_middle"] - num_std * rolling_std

        # 带宽
        result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / result["bb_middle"]

        # 价格位置 (0-1之间，超过1或低于0表示突破)
        result["bb_position"] = (
            (result["close"] - result["bb_lower"]) /
            (result["bb_upper"] - result["bb_lower"])
        )

        return result

    @staticmethod
    def compute_volatility(
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        计算波动率因子。

        包括历史波动率和 ATR (Average True Range)。

        Args:
            df: 包含 OHLC 数据的 DataFrame。
            windows: 波动率计算窗口列表，默认为 [5, 10, 20]。

        Returns:
            添加了波动率列的 DataFrame。
        """
        windows = windows or [5, 10, 20]
        result = df.copy()

        # 收益率波动率
        returns = result["close"].pct_change()
        for window in windows:
            result[f"volatility_{window}d"] = returns.rolling(window=window).std() * np.sqrt(252)

        # ATR (Average True Range)
        high = result["high"]
        low = result["low"]
        close = result["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        for window in windows:
            result[f"atr_{window}d"] = true_range.rolling(window=window).mean()

        return result

    @staticmethod
    def compute_volume_features(
        df: pd.DataFrame,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        计算成交量相关因子。

        Args:
            df: 包含 'close' 和 'volume' 列的 DataFrame。
            windows: 计算窗口列表，默认为 [5, 10, 20]。

        Returns:
            添加了成交量因子列的 DataFrame。
        """
        windows = windows or [5, 10, 20]
        result = df.copy()

        # 成交量变化率
        for window in windows:
            result[f"volume_ma_{window}d"] = result["volume"].rolling(window=window).mean()
            result[f"volume_ratio_{window}d"] = result["volume"] / result[f"volume_ma_{window}d"]

        # OBV (On Balance Volume)
        obv = (np.sign(result["close"].diff()) * result["volume"]).cumsum()
        result["obv"] = obv

        # VWAP (Volume Weighted Average Price) - 近似日级别
        result["vwap"] = (result["close"] * result["volume"]).cumsum() / result["volume"].cumsum()

        # 成交量波动率
        for window in windows:
            result[f"volume_volatility_{window}d"] = (
                result["volume"].rolling(window=window).std() /
                result["volume"].rolling(window=window).mean()
            )

        return result

    @staticmethod
    def compute_price_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算价格相关因子。

        Args:
            df: 包含 OHLC 数据的 DataFrame。

        Returns:
            添加了价格因子列的 DataFrame。
        """
        result = df.copy()

        # 日内幅度
        result["intraday_range"] = (result["high"] - result["low"]) / result["close"]

        # 开盘价与收盘价关系
        result["close_to_open"] = result["close"] / result["open"] - 1

        # 最高价位置 (当日最高价与收盘价的关系)
        result["high_position"] = (result["high"] - result["close"]) / (result["high"] - result["low"] + 1e-10)

        # 最低价位置
        result["low_position"] = (result["close"] - result["low"]) / (result["high"] - result["low"] + 1e-10)

        # 上影线比例
        result["upper_shadow"] = (result["high"] - result[["open", "close"]].max(axis=1)) / (
            result["high"] - result["low"] + 1e-10
        )

        # 下影线比例
        result["lower_shadow"] = (result[["open", "close"]].min(axis=1) - result["low"]) / (
            result["high"] - result["low"] + 1e-10
        )

        return result

    def compute_all_features(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """
        计算所有技术因子。

        Args:
            df: 包含 OHLCV 数据的 DataFrame。
            drop_na: 是否删除包含 NaN 的行，默认为 True。
            normalize: 是否对特征进行标准化，默认为 False。

        Returns:
            包含所有技术因子的 DataFrame。

        Raises:
            ValueError: 当输入数据缺少必要的列时抛出。
        """
        self._validate_ohlcv(df)

        result = df.copy()

        # 按顺序计算各类因子
        result = self.compute_returns(result)
        result = self.compute_momentum(result)
        result = self.compute_ma_cross(result)
        result = self.compute_rsi(result)
        result = self.compute_macd(result)
        result = self.compute_bollinger_bands(result)
        result = self.compute_volatility(result)
        result = self.compute_volume_features(result)
        result = self.compute_price_features(result)

        # 删除原始 OHLCV 列的特征名（保留原始数据）
        ohlcv_cols = ["open", "high", "low", "close", "volume"]

        # 排除不相关的列（来自yfinance但数据稀疏）
        exclude_cols = ["dividends", "stock splits", "capital gains", "symbol"]
        result = result.drop(columns=[c for c in exclude_cols if c in result.columns], errors='ignore')

        # 记录特征名称
        self.feature_names = [col for col in result.columns if col not in ohlcv_cols]

        # 删除 NaN 行
        if drop_na:
            result = result.dropna()

        # 标准化（可选）
        if normalize:
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            result[self.feature_names] = scaler.fit_transform(result[self.feature_names])

        return result

    def get_feature_names(self) -> list[str]:
        """
        获取计算后的特征名称列表。

        Returns:
            特征名称列表。
        """
        return self.feature_names.copy()

    def create_target(
        self,
        df: pd.DataFrame,
        forward_period: int = 1,
        threshold: float = 0.0,
    ) -> pd.DataFrame:
        """
        创建预测目标（涨跌标签）。

        Args:
            df: 包含 'close' 列的 DataFrame。
            forward_period: 预测未来 N 期的涨跌，默认为 1。
            threshold: 涨跌判定阈值，默认为 0.0（任何正收益为涨）。

        Returns:
            添加了目标列的 DataFrame。
            - target_return: 未来收益率
            - target_label: 涨跌标签 (1=涨, 0=跌)
        """
        result = df.copy()

        # 计算未来收益率
        result["target_return"] = result["close"].pct_change(forward_period).shift(-forward_period)

        # 创建标签
        result["target_label"] = (result["target_return"] > threshold).astype(int)

        return result


if __name__ == "__main__":
    # 示例用法
    import numpy as np

    # 创建示例数据
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=100, freq="B")
    base_price = 100
    returns = np.random.randn(100) * 0.02
    prices = base_price * (1 + returns).cumprod()

    sample_df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(100) * 0.005),
        "high": prices * (1 + np.abs(np.random.randn(100)) * 0.01),
        "low": prices * (1 - np.abs(np.random.randn(100)) * 0.01),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, 100),
    }, index=dates)

    # 计算特征
    engineer = FeatureEngineer()
    features_df = engineer.compute_all_features(sample_df)

    print(f"原始数据形状: {sample_df.shape}")
    print(f"特征数据形状: {features_df.shape}")
    print(f"特征数量: {len(engineer.feature_names)}")
    print(f"\n特征列表:\n{engineer.feature_names}")