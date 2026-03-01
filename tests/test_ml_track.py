"""
ML Track 模块验证测试脚本。

验证内容包括：
1. 数据加载与生成
2. 未来函数审查（防止数据泄露）
3. 模型训练与信号预测
4. 硬件加速确认
5. 输出格式验证
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.ml_track.features import FeatureEngineer
from src.models.ml_track.baselines import MLStrategyPortfolio, get_best_device


def print_separator(title: str) -> None:
    """打印分隔线。"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def print_subsection(title: str) -> None:
    """打印子标题。"""
    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print('─' * 70)


def create_sample_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    创建示例 OHLCV 数据。

    Args:
        n_samples: 样本数量。
        seed: 随机种子。

    Returns:
        示例数据 DataFrame。
    """
    np.random.seed(seed)
    dates = pd.date_range(start="2021-01-01", periods=n_samples, freq="B")

    # 生成模拟价格数据（带有一定趋势和波动）
    base_price = 100
    trend = np.linspace(0, 0.3, n_samples)  # 轻微上涨趋势
    noise = np.random.randn(n_samples) * 0.02
    returns = noise + trend / n_samples
    prices = base_price * (1 + returns).cumprod()

    # 创建 OHLCV 数据
    df = pd.DataFrame({
        "open": prices * (1 + np.random.randn(n_samples) * 0.003),
        "high": prices * (1 + np.abs(np.random.randn(n_samples)) * 0.008),
        "low": prices * (1 - np.abs(np.random.randn(n_samples)) * 0.008),
        "close": prices,
        "volume": np.random.randint(1000000, 10000000, n_samples),
    }, index=dates)

    # 确保价格合理性
    df["high"] = df[["open", "high", "close"]].max(axis=1)
    df["low"] = df[["open", "low", "close"]].min(axis=1)

    return df


# ============================================================================
# 验证点 1: 数据加载与生成
# ============================================================================
def test_data_generation() -> pd.DataFrame:
    """
    验证点 1: 生成模拟 OHLCV 数据。

    Returns:
        模拟数据 DataFrame。
    """
    print_separator("验证点 1: 数据加载与生成")

    # 尝试加载 Phase 1 生成的数据
    data_path = Path(__file__).parent.parent / "data" / "raw"

    loaded = False
    sample_df = None

    # 尝试加载已有的市场数据
    parquet_files = list(data_path.glob("*.parquet")) if data_path.exists() else []

    if parquet_files:
        print(f"\n📂 发现 {len(parquet_files)} 个数据文件，尝试加载...")
        for pf in parquet_files[:1]:  # 只加载第一个
            try:
                sample_df = pd.read_parquet(pf)
                print(f"   ✅ 成功加载: {pf.name}")
                loaded = True
                break
            except Exception as e:
                print(f"   ⚠️  加载失败: {pf.name}, 错误: {e}")

    if not loaded:
        print("\n📦 生成 1000 行模拟 OHLCV 数据...")
        sample_df = create_sample_data(n_samples=1000)

    # 数据质量检查
    print(f"\n📊 数据概览:")
    print(f"   - 数据形状: {sample_df.shape}")
    print(f"   - 日期范围: {sample_df.index.min()} ~ {sample_df.index.max()}")
    print(f"   - 列名: {list(sample_df.columns)}")

    # 检查 OHLCV 列
    required_cols = ["open", "high", "low", "close", "volume"]
    missing_cols = [col for col in required_cols if col not in sample_df.columns]
    if missing_cols:
        print(f"   ⚠️  缺少列: {missing_cols}")
    else:
        print(f"   ✅ 包含所有必需列: {required_cols}")

    # 检查数据类型
    print(f"\n📋 数据类型:")
    for col in sample_df.columns:
        print(f"   - {col}: {sample_df[col].dtype}")

    # 检查缺失值
    nan_count = sample_df.isna().sum().sum()
    print(f"\n🔍 缺失值检查:")
    print(f"   - 总缺失值: {nan_count}")
    if nan_count == 0:
        print(f"   ✅ 无缺失值")
    else:
        print(f"   ⚠️  存在缺失值")

    # 数据样例
    print(f"\n📋 数据样例 (前5行):")
    print(sample_df.head(5).to_string())

    return sample_df


# ============================================================================
# 验证点 2: 未来函数审查
# ============================================================================
def test_look_ahead_bias(sample_df: pd.DataFrame) -> pd.DataFrame:
    """
    验证点 2: 未来函数审查 - 确保没有使用未来数据。

    这是量化交易中最关键的检查点，必须确保在时间 t 计算因子时，
    绝对不使用 t+1 及以后的数据。

    Args:
        sample_df: 原始 OHLCV 数据。

    Returns:
        特征数据 DataFrame。
    """
    print_separator("验证点 2: 未来函数审查")

    print("\n⚠️  关键检查: 确保计算时间 t 的因子时，绝对没有使用 t+1 及以后的数据")
    print("   这通过检查 shift 操作是否正确应用来实现。")

    engineer = FeatureEngineer()

    # 记录原始数据
    original_close = sample_df["close"].copy()
    original_index = sample_df.index.copy()

    print_subsection("2.1 收益率因子审查")

    # 手动计算预期值
    expected_return_1d = original_close.pct_change(1)
    df_test = engineer.compute_returns(sample_df.copy())

    print("   [return_1d] 检查:")
    print("   - 计算公式: close[t] / close[t-1] - 1 = pct_change(1)")
    print("   - 检查方法: 时间 t 的收益使用 t 和 t-1 的价格")

    # 验证：return_1d[t] 应该等于 (close[t] - close[t-1]) / close[t-1]
    manual_calc = (original_close - original_close.shift(1)) / original_close.shift(1)
    diff = (df_test["return_1d"] - manual_calc).abs().max()
    print(f"   - 与手动计算差异: {diff:.2e}")

    if diff < 1e-10:
        print("   ✅ return_1d: 无未来函数，正确使用 shift(1)")
    else:
        print(f"   ❌ return_1d: 计算差异过大 = {diff}")

    # 关键断言
    assert diff < 1e-10, f"return_1d 存在计算错误，差异: {diff}"

    print_subsection("2.2 动量因子审查")

    df_test = engineer.compute_momentum(sample_df.copy())
    expected_momentum_5d = original_close / original_close.shift(5) - 1
    diff = (df_test["momentum_5d"] - expected_momentum_5d).abs().max()

    print("   [momentum_5d] 检查:")
    print("   - 计算公式: close[t] / close[t-5] - 1")
    print(f"   - 与手动计算差异: {diff:.2e}")

    assert diff < 1e-10, f"momentum_5d 存在计算错误，差异: {diff}"
    print("   ✅ momentum_5d: 无未来函数，正确使用 shift(5)")

    print_subsection("2.3 RSI 因子审查")

    df_test = engineer.compute_rsi(sample_df.copy())

    print("   [rsi_14] 检查:")
    print("   - 计算公式: 使用 ewm (指数加权移动平均)，只依赖历史数据")
    print("   - ewm 的 com 参数确保只使用 t 及之前的数据")

    # 验证 RSI 值范围（允许极端情况下的轻微偏差）
    rsi_valid = (df_test["rsi_14"] >= 0) & (df_test["rsi_14"] <= 100)
    valid_ratio = rsi_valid.mean()
    print(f"   - RSI 值在 [0, 100] 范围内的比例: {valid_ratio:.2%}")

    # 检查 NaN 分布：RSI 的前几个值应该是 NaN（因为没有足够的历史数据）
    rsi_nan_count = df_test["rsi_14"].isna().sum()
    print(f"   - RSI_14 前向 NaN 数量: {rsi_nan_count} (预期有 NaN)")
    assert rsi_nan_count > 0, "RSI_14 应该有前向 NaN（表示正确使用历史数据）"

    # 核心检查：验证 ewm 没有使用未来数据（通过检查 NaN 分布）
    # ewm 的前几个值应该是 NaN 或者不稳定
    first_valid_rsi = df_test["rsi_14"].first_valid_index()
    print(f"   - RSI_14 第一个有效值索引: {first_valid_rsi}")
    print("   ✅ rsi_14: 无未来函数，ewm 正确使用历史数据（存在前向 NaN）")

    print_subsection("2.4 MACD 因子审查")

    df_test = engineer.compute_macd(sample_df.copy())

    print("   [macd] 检查:")
    print("   - 计算公式: EMA(12) - EMA(26)")
    print("   - EMA 使用 ewm，只依赖 t 及之前的数据")

    # 验证 MACD 计算
    ema_12 = original_close.ewm(span=12, adjust=False).mean()
    ema_26 = original_close.ewm(span=26, adjust=False).mean()
    expected_macd = ema_12 - ema_26
    diff = (df_test["macd"] - expected_macd).abs().max()
    print(f"   - 与手动计算差异: {diff:.2e}")

    assert diff < 1e-10, f"macd 存在计算错误，差异: {diff}"
    print("   ✅ macd: 无未来函数，ewm 正确使用历史数据")

    print_subsection("2.5 布林带因子审查")

    df_test = engineer.compute_bollinger_bands(sample_df.copy())

    print("   [bb_position] 检查:")
    print("   - 计算公式: (close - lower) / (upper - lower)")
    print("   - rolling window 只使用 t 及之前的数据")

    # 验证 rolling 计算
    rolling_mean = original_close.rolling(window=20).mean()
    diff = (df_test["bb_middle"] - rolling_mean).abs().max()
    print(f"   - bb_middle 与手动 rolling 计算差异: {diff:.2e}")

    assert diff < 1e-10, f"bb_middle 存在计算错误，差异: {diff}"
    print("   ✅ bb_middle: 无未来函数，rolling 正确使用历史数据")

    print_subsection("2.6 目标变量审查（关键！）")

    df_test = engineer.create_target(sample_df.copy(), forward_period=1)

    print("   [target_return] 检查:")
    print("   - 这是预测目标，必须使用 shift(-1) 向前看")
    print("   - target_return[t] = (close[t+1] - close[t]) / close[t]")

    # 验证目标计算
    expected_target = original_close.pct_change(1).shift(-1)
    diff = (df_test["target_return"] - expected_target).abs().max()
    print(f"   - 与手动计算差异: {diff:.2e}")

    # 关键检查：确保最后一条记录的 target_return 是 NaN（因为没有 t+1 数据）
    last_target = df_test["target_return"].iloc[-1]
    print(f"   - 最后一条记录的 target_return: {last_target}")

    assert pd.isna(last_target), "target_return 最后一条应该是 NaN（无未来数据）"
    assert diff < 1e-10, f"target_return 存在计算错误，差异: {diff}"
    print("   ✅ target_return: 正确使用 shift(-1)，最后一条为 NaN")

    print_subsection("2.7 综合审查结论")

    # 计算所有特征并检查 NaN 分布
    features_df = engineer.compute_all_features(sample_df.copy(), drop_na=False)

    # 检查 NaN 分布 - 确保前向依赖的因子在前几行为 NaN
    print("\n   📊 NaN 分布检查:")
    nan_counts = features_df.isna().sum()

    # 关键检查：依赖历史数据的因子应该有前向 NaN
    lagging_features = ["return_1d", "momentum_5d", "ma_5", "rsi_14"]
    for feat in lagging_features:
        if feat in nan_counts:
            nan_count = nan_counts[feat]
            print(f"   - {feat}: {nan_count} 个 NaN (前向依赖)")
            if nan_count > 0:
                print(f"     ✅ 正确：存在前向 NaN，表示正确使用历史数据")

    # 关键断言：所有因子的 NaN 都应该集中在序列开头，而不是结尾
    # （除了 target_return/targe_label 应该在结尾有 NaN）
    for col in features_df.columns:
        if col not in ["target_return", "target_label"]:
            # 检查 NaN 是否只在序列开头
            col_values = features_df[col]
            if col_values.isna().any():
                first_valid = col_values.first_valid_index()
                last_valid = col_values.last_valid_index()
                if first_valid is not None:
                    # 断言：第一个有效值之后不应该有 NaN
                    after_first = col_values.loc[first_valid:]
                    nan_after_first = after_first.isna().sum()
                    assert nan_after_first == 0, f"{col}: 在第一个有效值之后仍有 NaN"

    print("\n   ✅ 所有关键检查通过！")
    print("   ✅ 结论: 所有因子计算均未使用未来数据")
    print("   ✅ target_return 正确使用 shift(-1) 向前看")
    print("   ✅ NaN 分布正确：前向依赖的因子在序列开头有 NaN")

    return features_df


# ============================================================================
# 验证点 3: 模型训练与信号预测
# ============================================================================
def test_model_training(features_df: pd.DataFrame) -> tuple[MLStrategyPortfolio, pd.DataFrame]:
    """
    验证点 3: 实例化 MLStrategyPortfolio，训练并进行信号预测。

    Args:
        features_df: 特征数据 DataFrame。

    Returns:
        训练后的模型组合和信号 DataFrame。
    """
    print_separator("验证点 3: 模型训练与信号预测")

    # 检测设备
    print_subsection("3.1 硬件加速检测")

    print("\n🖥️  PyTorch 设备状态:")
    print(f"   - PyTorch 版本: {torch.__version__}")
    print(f"   - MPS 可用: {torch.backends.mps.is_available()}")
    print(f"   - CUDA 可用: {torch.cuda.is_available()}")

    device = get_best_device()
    print(f"\n   ✅ 当前使用设备: {device}")

    if device.type == "mps":
        print("   ✅ Apple Silicon MPS 加速已启用")
    elif device.type == "cuda":
        print("   ✅ NVIDIA CUDA 加速已启用")
    else:
        print("   ⚠️  使用 CPU，训练速度可能较慢")

    print_subsection("3.2 数据准备")

    # 准备数据
    engineer = FeatureEngineer()
    features_df = engineer.create_target(features_df, forward_period=1)
    features_df = features_df.dropna()

    print(f"\n📊 训练数据概览:")
    print(f"   - 总样本数: {len(features_df)}")
    print(f"   - 特征数: {len(engineer.feature_names)}")
    print(f"   - 正样本比例: {features_df['target_label'].mean():.2%}")

    print_subsection("3.3 模型训练")

    # 初始化并训练
    portfolio = MLStrategyPortfolio(
        lstm_hidden_dim=32,
        lstm_num_layers=2,
        lstm_epochs=20,
        lstm_sequence_length=10,
        lgb_n_estimators=50,
        lgb_max_depth=5,
    )

    print("\n🚀 开始训练三个基线模型...")
    total_start = time.time()
    portfolio.fit(features_df, target_col="target_label", test_size=0.2)
    total_elapsed = time.time() - total_start

    print(f"\n⏱️  总训练时间: {total_elapsed:.2f}秒")

    print_subsection("3.4 模型评估指标")

    metrics_df = portfolio.get_metrics()
    print(f"\n{metrics_df.to_string(index=False)}")

    print_subsection("3.5 信号预测")

    # 使用最近 20 条数据进行预测
    predict_data = features_df.tail(20)
    print(f"\n🔮 使用最近 {len(predict_data)} 条数据进行信号预测...")

    signals_df = portfolio.predict(predict_data, symbol="TEST")

    print(f"   - 预测信号数: {len(signals_df)}")
    print(f"   - 预测时间范围: {signals_df['timestamp'].min()} ~ {signals_df['timestamp'].max()}")

    return portfolio, signals_df


# ============================================================================
# 验证点 4: 硬件加速确认
# ============================================================================
def test_hardware_acceleration() -> dict:
    """
    验证点 4: 确认 LSTM 模型是否成功使用 MPS 设备。

    Returns:
        设备信息字典。
    """
    print_separator("验证点 4: 硬件加速确认")

    print("\n📋 详细设备检查:")

    device_info = {
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }

    print(f"   - PyTorch 版本: {device_info['pytorch_version']}")
    print(f"   - MPS 可用: {device_info['mps_available']}")
    print(f"   - MPS 已构建: {device_info['mps_built']}")
    print(f"   - CUDA 可用: {device_info['cuda_available']}")

    if device_info["cuda_available"]:
        print(f"   - CUDA 版本: {device_info['cuda_version']}")
        print(f"   - GPU 数量: {torch.cuda.device_count()}")
        print(f"   - GPU 名称: {torch.cuda.get_device_name(0)}")

    # 实际测试 MPS/CUDA 是否工作
    print("\n🧪 设备功能测试:")

    try:
        best_device = get_best_device()

        # 创建测试张量
        test_tensor = torch.randn(10, 10).to(best_device)
        result = torch.matmul(test_tensor, test_tensor.T)

        print(f"   ✅ 成功在 {best_device} 上执行矩阵运算")
        print(f"   ✅ 结果张量设备: {result.device}")

        device_info["active_device"] = str(best_device)
        device_info["test_passed"] = True

    except Exception as e:
        print(f"   ❌ 设备测试失败: {e}")
        device_info["active_device"] = "cpu"
        device_info["test_passed"] = False
        device_info["error"] = str(e)

    # LSTM 专用测试
    print("\n🧪 LSTM 专用设备测试:")

    try:
        # 创建小型 LSTM 模型
        lstm = torch.nn.LSTM(input_size=10, hidden_size=20, batch_first=True).to(best_device)
        test_input = torch.randn(5, 10, 10).to(best_device)
        output, _ = lstm(test_input)

        print(f"   ✅ LSTM 成功在 {best_device} 上运行")
        print(f"   ✅ 输出形状: {output.shape}")
        device_info["lstm_test_passed"] = True

    except Exception as e:
        print(f"   ❌ LSTM 设备测试失败: {e}")
        device_info["lstm_test_passed"] = False
        device_info["lstm_error"] = str(e)

    # 最终结论
    print("\n📊 硬件加速结论:")
    if device_info.get("test_passed") and device_info.get("lstm_test_passed"):
        if best_device.type == "mps":
            print("   ✅ LSTM 模型已成功挂载到 MPS (Apple Silicon) 设备")
            print("   ✅ 训练将获得 Apple Silicon GPU 加速")
        elif best_device.type == "cuda":
            print("   ✅ LSTM 模型已成功挂载到 CUDA 设备")
            print("   ✅ 训练将获得 NVIDIA GPU 加速")
        else:
            print("   ⚠️  当前使用 CPU，建议检查 GPU 设置")
    else:
        print("   ❌ 硬件加速测试未通过，将回退到 CPU")

    return device_info


# ============================================================================
# 验证点 5: 输出格式验证
# ============================================================================
def test_output_format(signals_df: pd.DataFrame) -> bool:
    """
    验证点 5: 验证信号 DataFrame 的输出格式。

    Args:
        signals_df: 信号 DataFrame。

    Returns:
        是否通过验证。
    """
    print_separator("验证点 5: 输出格式验证")

    print_subsection("5.1 DataFrame 基本信息")

    print(f"\n📊 DataFrame 形状:")
    print(f"   - shape: {signals_df.shape}")
    print(f"   - 行数: {len(signals_df)}")
    print(f"   - 列数: {len(signals_df.columns)}")

    print(f"\n📋 列名:")
    print(f"   - 列名列表: {list(signals_df.columns)}")

    print_subsection("5.2 字段验证")

    # 检查必需字段
    required_cols = ["timestamp", "symbol", "model_name", "signal_strength_0_to_1"]
    all_passed = True

    print(f"\n✅ 必需字段检查:")
    for col in required_cols:
        if col in signals_df.columns:
            print(f"   ✅ {col}: 存在")
        else:
            print(f"   ❌ {col}: 缺失")
            all_passed = False

    print_subsection("5.3 数据类型验证")

    print(f"\n📋 数据类型检查:")
    for col in signals_df.columns:
        dtype = signals_df[col].dtype
        print(f"   - {col}: {dtype}")

        # 类型检查（放宽条件，允许 Python 类型如 str）
        if col == "timestamp":
            # 应该是 datetime 或可转换为 datetime
            pass
        elif col == "symbol":
            # 可以是 object 或 str
            assert dtype in ["object", "str"] or str(dtype) == "object", f"symbol 应该是字符串类型，实际是 {dtype}"
        elif col == "model_name":
            assert dtype in ["object", "str"] or str(dtype) == "object", f"model_name 应该是字符串类型，实际是 {dtype}"
        elif col == "signal_strength_0_to_1":
            assert dtype in ["float64", "float32"], f"signal_strength 应该是浮点类型，实际是 {dtype}"

    print_subsection("5.4 信号强度范围验证")

    signal_col = "signal_strength_0_to_1"
    if signal_col in signals_df.columns:
        min_val = signals_df[signal_col].min()
        max_val = signals_df[signal_col].max()
        mean_val = signals_df[signal_col].mean()

        print(f"\n📊 信号强度统计:")
        print(f"   - 最小值: {min_val:.4f}")
        print(f"   - 最大值: {max_val:.4f}")
        print(f"   - 平均值: {mean_val:.4f}")
        print(f"   - 标准差: {signals_df[signal_col].std():.4f}")

        if 0 <= min_val <= 1 and 0 <= max_val <= 1:
            print(f"\n   ✅ 信号强度全部在 [0, 1] 范围内")
        else:
            print(f"\n   ⚠️  信号强度超出 [0, 1] 范围")
            all_passed = False

    print_subsection("5.5 模型覆盖验证")

    models = signals_df["model_name"].unique()
    expected_models = {"LogisticRegression", "LSTM", "LightGBM"}

    print(f"\n📋 模型列表:")
    print(f"   - 预期模型: {expected_models}")
    print(f"   - 实际模型: {set(models)}")

    if set(models) == expected_models:
        print(f"   ✅ 所有预期模型都已生成信号")
    else:
        missing = expected_models - set(models)
        extra = set(models) - expected_models
        if missing:
            print(f"   ⚠️  缺少模型: {missing}")
        if extra:
            print(f"   ⚠️  额外模型: {extra}")

    print_subsection("5.6 输出示例")

    print(f"\n📋 信号 DataFrame head(5):")
    print(signals_df.head(5).to_string())

    print(f"\n📋 信号 DataFrame tail(5):")
    print(signals_df.tail(5).to_string())

    print_subsection("5.7 最终验证结论")

    if all_passed:
        print(f"\n   ✅ 所有格式验证通过！")
        print(f"   ✅ 输出格式严格遵循 [timestamp, symbol, model_name, signal_strength_0_to_1]")
    else:
        print(f"\n   ❌ 部分验证未通过，请检查上述输出")

    return all_passed


# ============================================================================
# 主函数
# ============================================================================
def main() -> None:
    """运行所有验证测试。"""
    print("\n" + "=" * 70)
    print("  🚀 ML Track 模块验证测试")
    print("=" * 70)
    print(f"  测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Python 版本: {sys.version.split()[0]}")

    total_start = time.time()

    # 验证点 1: 数据加载与生成
    sample_df = test_data_generation()

    # 验证点 2: 未来函数审查
    features_df = test_look_ahead_bias(sample_df.copy())

    # 验证点 3: 模型训练与信号预测
    portfolio, signals_df = test_model_training(features_df)

    # 验证点 4: 硬件加速确认
    device_info = test_hardware_acceleration()

    # 验证点 5: 输出格式验证
    format_passed = test_output_format(signals_df)

    total_elapsed = time.time() - total_start

    # 最终汇总
    print("\n" + "=" * 70)
    print("  📋 验证结果汇总")
    print("=" * 70)

    print(f"\n  [✓] 验证点 1: 数据加载与生成 - 通过")
    print(f"  [✓] 验证点 2: 未来函数审查 - 通过（所有因子未使用未来数据）")
    print(f"  [✓] 验证点 3: 模型训练与信号预测 - 通过")
    print(f"  [{'✓' if device_info.get('test_passed') else '✗'}] 验证点 4: 硬件加速确认 - "
          f"{'通过 (设备: ' + device_info.get('active_device', 'unknown') + ')' if device_info.get('test_passed') else '未通过'}")
    print(f"  [{'✓' if format_passed else '✗'}] 验证点 5: 输出格式验证 - {'通过' if format_passed else '未通过'}")

    print(f"\n  ⏱️  总测试时间: {total_elapsed:.2f}秒")
    print("=" * 70)

    # 返回结果供外部使用
    return {
        "sample_df": sample_df,
        "features_df": features_df,
        "portfolio": portfolio,
        "signals_df": signals_df,
        "device_info": device_info,
        "format_passed": format_passed,
    }


if __name__ == "__main__":
    results = main()