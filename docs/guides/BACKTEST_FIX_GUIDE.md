# DualTrack 回测框架修复指导意见

**版本**: 2.0
**日期**: 2026-03-01
**优先级**: P0（阻塞性问题）

---

## 🎯 DualTrack 核心架构思想

> **"双轨制不是融合策略，而是完成对比实验的工程基础设施（Testbed）。"**

### 为什么要用"双轨制"做对比实验？

传统论文的对比实验通常是：
- 拿一段历史数据跑一遍传统模型（算一个夏普比率）
- 再拿这段数据跑一遍新模型（算一个夏普比率）
- 最后把两张表拼在一起

**这种做法在真实交易系统中是没有意义的**，因为你无法衡量它们在同一时刻发生分歧时的代价。

### "双轨制"架构的三个核心价值

#### 1. 创造绝对公平的"竞技场"（控制变量）

要对比两种截然不同的技术，必须保证它们吃的数据、看到的时间戳是绝对同步的。

**DualTrack Orchestrator** 集成公共数据源（Data Feed），同时向多个独立的执行引擎派发数据：
- LR Track
- LSTM Track
- LightGBM Track
- LLM(Cloud) Track
- LLM(Local) Track

这就建立了一个统一的基准测试标准，能在**完全相同的市场条件**下，量化它们在财务表现（ROI）和工程成本（Latency/Cost）之间的权衡。

#### 2. "拟合"与"推理"的实时对抗（对比假设的验证）

双轨制不是简单地把模型混在一起，而是让它们在**同一个时间线上暴露各自的优缺点**：

| 轨道 | 代表 | 优势 | 劣势 |
|------|------|------|------|
| **ML Tracks** | 拟合（Fitting） | 执行速度快（<10ms）、在技术面驱动的市场中极其稳定 | 无法理解语义，黑天鹅事件中失效 |
| **LLM Tracks** | 推理（Reasoning） | 能理解语义，在突发新闻和高波动率阶段提供下行风险保护 | 延迟高（>1s）、成本高 |

**如何体现对比？** 通过 Signal Converter（信号转换器）记录分歧点：
- 当 ML 基于历史价格继续看多
- 但 LLM 读到负面新闻强制看空
- 系统记录下这个分歧点

这正是论文要证明的核心假设：**LLM 能避开 ML 踩进去的坑**。

#### 3. 多维度的工程可行性对比（Engineering Feasibility）

如果只比盈亏，就脱离了量化开发的实际。DualTrack 同时监测各轨道的"造价"：

| 维度 | ML Tracks | LLM Tracks |
|------|-----------|------------|
| 延迟 | < 10ms | 1-5s |
| 成本 | ≈ 0（本地计算） | $0.001-0.01/次 |
| 硬件 | CPU/MPS | GPU/云端 |

最终结论不仅是"谁更赚钱"，而是**"购买 LLM 的智能是否值回它的计算成本"**。

### 五轨道对比设计

```
┌─────────────────────────────────────────────────────────────────┐
│                     DualTrack 测试平台                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  LR Track   │  │ LSTM Track  │  │ LightGBM    │   ML阵营     │
│  │  (拟合)     │  │  (序列)     │  │  (集成)     │   速度优先   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│              ┌─────────────────────┐                           │
│              │  Signal Converter   │  ← 独立转换，不融合！      │
│              │  (信号→目标仓位)     │                           │
│              └─────────────────────┘                           │
│                          │                                      │
│  ┌───────────────────────┼───────────────────────┐             │
│  │                       ▼                       │             │
│  │  ┌─────────────┐  ┌─────────────┐           │             │
│  │  │ LLM(Cloud)  │  │ LLM(Local)  │   LLM阵营  │             │
│  │  │ DeepSeek    │  │ Ollama      │   智能优先 │             │
│  │  └──────┬──────┘  └──────┬──────┘           │             │
│  │         │                │                  │             │
│  │         └────────────────┘                  │             │
│  │                      │                       │             │
│  │                      ▼                       │             │
│  │         ┌─────────────────────┐             │             │
│  │         │  Signal Converter   │             │             │
│  │         └─────────────────────┘             │             │
│  │                      │                       │             │
│  └──────────────────────┼───────────────────────┘             │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              Backtrader 回测引擎                     │      │
│  │  • 分别回测5个轨道                                   │      │
│  │  • 记录每笔交易、每日收益                            │      │
│  │  • 计算金融指标（Sharpe/MaxDD等）                    │      │
│  └─────────────────────────────────────────────────────┘      │
│                         │                                      │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              Comparator 对比分析器                   │      │
│  │  • LR vs LSTM vs LightGBM vs LLM(Cloud) vs LLM(Local)│     │
│  │  • 回答：谁更赚钱？谁更稳健？谁更划算？              │      │
│  └─────────────────────────────────────────────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**对比实验设计**：

| 实验组 | 轨道 | 对比维度 |
|--------|------|----------|
| **Exp-1** | LR vs LSTM vs LightGBM | 哪种ML模型最适合量化？ |
| **Exp-2** | LLM(Cloud) vs LLM(Local) | 云端vs本地：效果vs成本 |
| **Exp-3** | Best ML vs Best LLM | 最终对决：拟合vs推理 |
| **Exp-4** | All 5 Tracks | 全景对比 |

---

## 🚨 问题诊断总结

经过详细检查，发现三个核心问题：

| 问题 | 严重程度 | 影响 |
|------|----------|------|
| 1. ML使用Mock数据 | ⛔ 高 | 无法评估真实ML效果 |
| 2. LLM无实际交易 | ⛔ 高 | 回测结果为零收益 |
| 3. 时间戳格式混乱 | ⚠️ 中 | 信号与数据无法对齐 |

---

## 问题1: ML使用Mock数据而非真实数据

### 问题分析

当前 `main.py` 中的 `_generate_mock_ml_signals()` 函数生成模拟信号：

```python
# ❌ 当前代码 (main.py ~line 654)
def _generate_mock_ml_signals(symbol: str, n: int) -> pd.DataFrame:
    """生成模拟 ML 信号。"""
    np.random.seed(42)
    return pd.DataFrame({
        "symbol": [symbol] * n,
        "model_name": np.random.choice(["LightGBM", "LogisticRegression", "LSTM"], n),
        "signal_strength_0_to_1": np.random.uniform(0.3, 0.8, n),
        "latency_ms": np.random.uniform(1, 15, n),
    })  # ❌ 缺少 timestamp 列！
```

**问题**:
1. 使用随机生成的信号，非真实模型预测
2. 缺少 `timestamp` 列，导致信号无法与OHLCV数据对齐
3. 无法评估ML模型的真实性能

### 修复方案

#### Plan A: 自动获取并训练（推荐用于新数据）

修改 `main.py` 的 Phase 2 部分：

```python
# ================================================================
# Phase 2: ML Track 信号生成 (使用真实模型)
# ================================================================
@click.echo("\n[Phase 2/6] ML Track 信号生成 (真实模型)...")

ml_signals = None

# 尝试加载已有模型，或训练新模型
model_path = Path(f"models/{symbol}_ml_portfolio.pkl")

if model_path.exists():
    click.echo(f"  加载已有模型: {model_path}")
    portfolio = MLStrategyPortfolio()
    # 这里需要根据实际保存方式加载模型
else:
    click.echo("  🚀 训练新模型...")

    # 1. 特征工程
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.compute_all_features(aligned_data["ohlcv"])

    # 2. 创建目标标签（下一天涨跌）
    features_df = feature_engineer.create_target(features_df, forward_period=1)
    features_df = features_df.dropna()

    if len(features_df) < 50:
        click.echo("  ⚠️ 数据不足，使用Mock信号")
        ml_signals = _generate_mock_ml_signals(symbol, len(ohlcv_data), ohlcv_data.index)
    else:
        # 3. 训练模型
        portfolio = MLStrategyPortfolio(
            lstm_hidden_dim=64,
            lstm_num_layers=2,
            lstm_epochs=20,  # 快速训练
            lstm_sequence_length=20,
            lgb_n_estimators=100,
        )

        try:
            portfolio.fit(features_df, target_col="target_label", test_size=0.2)

            # 4. 保存模型
            model_path.parent.mkdir(parents=True, exist_ok=True)
            portfolio.save_models(str(model_path))
            click.echo(f"  ✅ 模型训练完成，已保存")
        except Exception as e:
            click.echo(f"  ⚠️ 模型训练失败: {e}，使用Mock信号")
            ml_signals = _generate_mock_ml_signals(symbol, len(ohlcv_data), ohlcv_data.index)

# 5. 生成预测信号
if ml_signals is None:
    try:
        # 重新计算特征（不包含target）
        features_df = feature_engineer.compute_all_features(aligned_data["ohlcv"])
        ml_signals = portfolio.predict(features_df, symbol=symbol)
        click.echo(f"  ✅ ML 信号生成: {len(ml_signals)} 条")
    except Exception as e:
        click.echo(f"  ⚠️ 信号生成失败: {e}，使用Mock信号")
        ml_signals = _generate_mock_ml_signals(symbol, len(ohlcv_data), ohlcv_data.index)
```

#### Plan B: 手动下载数据并清洗（推荐用于已有数据）

如果已有CSV数据文件，创建数据加载脚本：

```python
# scripts/load_real_data.py
import pandas as pd
from pathlib import Path

def load_and_clean_csi300(csv_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载并清洗手动下载的沪深300数据。

    支持格式:
    - 同花顺、东方财富等导出的CSV
    - 需要包含: date, open, high, low, close, volume
    """
    df = pd.read_csv(csv_path)

    # 标准化列名
    column_mapping = {
        '日期': 'date',
        'Date': 'date',
        '开盘': 'open',
        'Open': 'open',
        '最高': 'high',
        'High': 'high',
        '最低': 'low',
        'Low': 'low',
        '收盘': 'close',
        'Close': 'close',
        '成交量': 'volume',
        'Volume': 'volume',
    }
    df = df.rename(columns=column_mapping)

    # 确保必要列存在
    required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"缺少必要列: {col}")

    # 清洗数据
    df['date'] = pd.to_datetime(df['date'])
    df = df[required_cols].copy()

    # 日期过滤
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    df = df[(df['date'] >= start_dt) & (df['date'] <= end_dt)]

    # 设置索引
    df.set_index('date', inplace=True)
    df['symbol'] = 'CSI300'

    # 排序
    df.sort_index(inplace=True)

    return df

if __name__ == "__main__":
    # 使用示例
    df = load_and_clean_csi300(
        'data/raw/manual_download.csv',
        '2025-01-01',
        '2026-02-28'
    )
    df.to_csv('data/raw/real_csi300_cleaned.csv')
    print(f"清洗完成: {len(df)} 条记录")
```

#### 修改 Mock 信号生成器（临时方案）

```python
def _generate_mock_ml_signals(symbol: str, n: int, dates: pd.DatetimeIndex = None) -> pd.DataFrame:
    """
    生成模拟 ML 信号（带时间戳）。

    Args:
        symbol: 股票代码
        n: 信号数量
        dates: 日期索引（必须与OHLCV对齐）
    """
    np.random.seed(42)

    if dates is None:
        dates = pd.date_range(end=pd.Timestamp.now(), periods=n, freq='B')

    return pd.DataFrame({
        "timestamp": dates,  # ✅ 添加时间戳
        "symbol": [symbol] * n,
        "model_name": np.random.choice(["LightGBM", "LogisticRegression", "LSTM"], n),
        "signal_strength_0_to_1": np.random.uniform(0.3, 0.8, n),
        "latency_ms": np.random.uniform(1, 15, n),
    })
```

---

## 问题2: LLM无实际交易

### 问题分析

**根本原因**: LLM缓存中的信号日期与OHLCV交易日**未对齐**

检查数据：
```
LLM缓存日期: 2026-01-09, 2026-01-09, 2026-01-09...（新闻发布时间）
OHLCV日期:   2025-02-28, 2025-03-03, 2025-03-04...（交易日）
```

**问题**:
1. LLM信号按新闻时间戳存储，但回测需要交易日信号
2. 同一日期有多条新闻信号，需要聚合
3. 非交易日的新闻信号需要映射到最近交易日

### 修复方案

修改 `SignalConverter.llm_signals_to_positions()`：

```python
@staticmethod
def llm_signals_to_positions(
    llm_signals: pd.DataFrame,
    ohlcv_dates: pd.DatetimeIndex = None
) -> dict:
    """
    将 LLM 信号转换为目标仓位（支持交易日对齐）。

    Args:
        llm_signals: LLM Track 信号 DataFrame
        ohlcv_dates: OHLCV数据的日期索引，用于对齐

    Returns:
        目标仓位字典 {datetime: {symbol: weight}}
    """
    from datetime import datetime
    positions = {}
    signal_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}

    if llm_signals.empty:
        return positions

    # 1. 确保timestamp列存在且为datetime类型
    if 'timestamp' in llm_signals.columns:
        llm_signals = llm_signals.copy()
        llm_signals['timestamp'] = pd.to_datetime(llm_signals['timestamp'])
    else:
        click.echo("  ⚠️ LLM信号缺少timestamp列")
        return positions

    # 2. 按日期聚合多条新闻信号（取平均）
    llm_signals['date'] = llm_signals['timestamp'].dt.date

    daily_signals = []
    for date, group in llm_signals.groupby('date'):
        # 计算当日平均信号
        avg_confidence = group['confidence'].mean()

        # 信号投票（buy/sell/hold取多数）
        signal_counts = group['llm_signal'].value_counts()
        dominant_signal = signal_counts.index[0]

        # 转换为权重
        signal_weight = signal_map.get(dominant_signal, 0.0)
        final_weight = signal_weight * avg_confidence

        daily_signals.append({
            'date': pd.Timestamp(date),
            'symbol': group['symbol'].iloc[0] if 'symbol' in group.columns else 'CSI300',
            'weight': final_weight,
            'signal': dominant_signal,
            'confidence': avg_confidence,
            'news_count': len(group)
        })

    daily_df = pd.DataFrame(daily_signals)

    # 3. 如果提供了OHLCV日期，对齐到交易日
    if ohlcv_dates is not None:
        ohlcv_dates = pd.to_datetime(ohlcv_dates)
        aligned_positions = {}

        for trade_date in ohlcv_dates:
            # 找到最近的有信号日期（不晚于交易日）
            past_signals = daily_df[daily_df['date'] <= trade_date]

            if not past_signals.empty:
                # 使用最近日期的信号
                latest = past_signals.iloc[-1]
                aligned_positions[trade_date] = {
                    latest['symbol']: latest['weight']
                }
            else:
                # 无信号时保持空仓
                aligned_positions[trade_date] = {'CSI300': 0.0}

        return aligned_positions
    else:
        # 不对齐，直接使用信号日期
        for _, row in daily_df.iterrows():
            positions[row['date']] = {row['symbol']: row['weight']}

        return positions
```

在 `main.py` 中调用时传入OHLCV日期：

```python
# Phase 4: 信号转换
if experiment in ["llm", "both"]:
    try:
        # ✅ 传入OHLCV日期进行对齐
        llm_positions = SignalConverter.llm_signals_to_positions(
            llm_signals,
            ohlcv_dates=ohlcv_data.index
        )
        click.echo(f"  ✅ LLM 仓位: {len(llm_positions)} 个时间点")
    except Exception as e:
        click.echo(f"  ⚠️ LLM 信号转换失败: {e}")
```

---

## 问题3: 时间戳格式统一

### 问题分析

当前系统中的日期格式混乱：
- OHLCV数据: `datetime64[ns]` 索引
- LLM缓存: `datetime64[us]` 字符串
- Mock信号: 无时间戳

### 修复方案

创建统一的时间处理工具：

```python
# src/utils/time_utils.py
import pandas as pd
from datetime import datetime
from typing import Union

def normalize_timestamp(
    ts: Union[str, datetime, pd.Timestamp],
    fmt: str = None
) -> pd.Timestamp:
    """
    统一时间戳格式。

    Args:
        ts: 输入时间戳（字符串、datetime或Timestamp）
        fmt: 字符串格式（可选）

    Returns:
        标准化的 pandas Timestamp
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

    Args:
        signals: 信号DataFrame
        trading_days: 交易日索引
        date_col: 日期列名

    Returns:
        对齐后的信号DataFrame
    """
    signals = signals.copy()
    signals[date_col] = pd.to_datetime(signals[date_col]).dt.normalize()

    # 创建交易日映射
    aligned_data = []
    trading_days = pd.to_datetime(trading_days).normalize()

    for trade_day in trading_days:
        # 找到该交易日或之前的最新信号
        past_signals = signals[signals[date_col] <= trade_day]

        if not past_signals.empty:
            latest = past_signals.iloc[-1:].copy()
            latest[date_col] = trade_day
            aligned_data.append(latest)

    if aligned_data:
        return pd.concat(aligned_data, ignore_index=True)
    else:
        return pd.DataFrame()
```

---

## 🛠️ 实施步骤

### Step 1: 修复时间戳问题（30分钟）

1. 创建 `src/utils/time_utils.py`
2. 修改 `_generate_mock_ml_signals()` 添加时间戳参数
3. 修改 `SignalConverter` 使用统一时间处理

### Step 2: 修复LLM交易问题（1小时）

1. 修改 `SignalConverter.llm_signals_to_positions()` 支持交易日对齐
2. 在 `main.py` 中传入OHLCV日期
3. 添加信号聚合逻辑（多日新闻合并）

### Step 3: 集成真实ML模型（2小时）

1. 修改 `main.py` Phase 2 逻辑
2. 添加模型训练/加载流程
3. 确保特征工程正确运行
4. 测试ML信号生成

### Step 4: 扩展为五轨道CLI设计（1小时）

修改 `main.py` 支持五个独立轨道：

```python
# main.py CLI 参数修改
@cli.command("run")
@click.option("--track", "-t",
              type=click.Choice(["lr", "lstm", "lgb", "llm-cloud", "llm-local", "all"]),
              default="all",
              help="选择回测轨道: lr=LR, lstm=LSTM, lgb=LightGBM, llm-cloud=云端LLM, llm-local=本地LLM, all=全部")
@click.option("--compare", "-c", is_flag=True,
              help="生成对比分析报告")
```

五轨道回测逻辑：

```python
def run_backtest(track: str, compare: bool, ...):
    tracks_to_run = []

    if track == "all":
        tracks_to_run = ["lr", "lstm", "lgb", "llm-cloud", "llm-local"]
    else:
        tracks_to_run = [track]

    results = {}

    # 共用数据准备
    ohlcv_data, news_data = fetch_data(...)

    # 轨道1: LR
    if "lr" in tracks_to_run:
        click.echo("\n" + "="*70)
        click.echo("  【轨道1/5】Logistic Regression")
        click.echo("="*70)
        lr_signals = generate_lr_signals(ohlcv_data)
        lr_positions = SignalConverter.ml_signals_to_positions(lr_signals)
        lr_result = run_backtest_engine(lr_positions, label="LR_Track")
        results["LR"] = lr_result

    # 轨道2: LSTM
    if "lstm" in tracks_to_run:
        click.echo("\n" + "="*70)
        click.echo("  【轨道2/5】LSTM")
        click.echo("="*70)
        lstm_signals = generate_lstm_signals(ohlcv_data)
        lstm_positions = SignalConverter.ml_signals_to_positions(lstm_signals)
        lstm_result = run_backtest_engine(lstm_positions, label="LSTM_Track")
        results["LSTM"] = lstm_result

    # 轨道3: LightGBM
    if "lgb" in tracks_to_run:
        click.echo("\n" + "="*70)
        click.echo("  【轨道3/5】LightGBM")
        click.echo("="*70)
        lgb_signals = generate_lgb_signals(ohlcv_data)
        lgb_positions = SignalConverter.ml_signals_to_positions(lgb_signals)
        lgb_result = run_backtest_engine(lgb_positions, label="LightGBM_Track")
        results["LightGBM"] = lgb_result

    # 轨道4: LLM(Cloud)
    if "llm-cloud" in tracks_to_run:
        click.echo("\n" + "="*70)
        click.echo("  【轨道4/5】LLM (Cloud - DeepSeek)")
        click.echo("="*70)
        llm_cloud_signals = generate_llm_signals(news_data, executor="deepseek")
        llm_cloud_positions = SignalConverter.llm_signals_to_positions(llm_cloud_signals, ohlcv_dates=ohlcv_data.index)
        llm_cloud_result = run_backtest_engine(llm_cloud_positions, label="LLM_Cloud_Track")
        results["LLM_Cloud"] = llm_cloud_result

    # 轨道5: LLM(Local)
    if "llm-local" in tracks_to_run:
        click.echo("\n" + "="*70)
        click.echo("  【轨道5/5】LLM (Local - Ollama)")
        click.echo("="*70)
        llm_local_signals = generate_llm_signals(news_data, executor="ollama")
        llm_local_positions = SignalConverter.llm_signals_to_positions(llm_local_signals, ohlcv_dates=ohlcv_data.index)
        llm_local_result = run_backtest_engine(llm_local_positions, label="LLM_Local_Track")
        results["LLM_Local"] = llm_local_result

    # 对比分析
    if compare and len(results) > 1:
        click.echo("\n" + "="*70)
        click.echo("  【五轨道对比分析】")
        click.echo("="*70)
        comparison = compare_five_tracks(results)
        print_five_track_comparison(comparison)
```

### Step 5: 测试验证（30分钟）

```bash
# 单轨道测试
python main.py run --track lr --symbol CSI300
python main.py run --track lstm --symbol CSI300
python main.py run --track lgb --symbol CSI300
python main.py run --track llm-cloud --symbol CSI300
python main.py run --track llm-local --symbol CSI300

# 全轨道对比测试
python main.py run --track all --compare --symbol CSI300
```

**预期输出**：
```
======================================================================
  五轨道对比分析
======================================================================

【财务指标对比】
轨道            夏普比率    最大回撤    总收益率    胜率
------------------------------------------------------------
LR              1.05       18.5%      12.3%      54%
LSTM            1.28       15.2%      15.7%      56%  ⭐ 最佳夏普
LightGBM        1.18       16.8%      14.1%      55%
LLM(Cloud)      0.92       12.1%       9.8%      52%  ⭐ 最佳风控
LLM(Local)      0.88       13.5%       8.9%      51%

【工程指标对比】
轨道            平均延迟    总成本      信号数量
------------------------------------------------------------
LR              2.1ms      $0.00      242
LSTM            15.3ms     $0.00      242
LightGBM        3.2ms      $0.00      242
LLM(Cloud)      1250ms     $2.40      51
LLM(Local)      3200ms     $0.00      51

【核心结论】
📊 收益能力：LSTM > LightGBM > LR > LLM(Cloud) > LLM(Local)
🛡️ 风险控制：LLM(Cloud) > LLM(Local) > LSTM > LightGBM > LR
⚡ 执行效率：LR > LightGBM > LSTM > LLM(Cloud) > LLM(Local)
💰 成本效益：LR = LSTM = LightGBM > LLM(Local) > LLM(Cloud)

【论文核心发现】
✅ 假设1验证：ML Tracks（LSTM）在收益能力上优于LLM Tracks
✅ 假设2验证：LLM Tracks 在黑天鹅事件中展现出更好的风险控制
✅ 假设3验证：ML Tracks 在成本效益上显著优于LLM Tracks
```

---

## ✅ 验证清单

修复完成后，必须验证：

### 五轨道独立验证
- [ ] **LR Track**: 信号生成、回测执行、非零收益
- [ ] **LSTM Track**: 信号生成、回测执行、非零收益
- [ ] **LightGBM Track**: 信号生成、回测执行、非零收益
- [ ] **LLM(Cloud) Track**: 信号生成、回测执行、非零收益
- [ ] **LLM(Local) Track**: 信号生成、回测执行、非零收益

### 数据质量验证
- [ ] 所有轨道信号包含正确的时间戳
- [ ] 信号日期与OHLCV交易日对齐
- [ ] 无未来函数（信号时间 ≤ 交易日）
- [ ] ML信号使用真实模型预测（非随机）
- [ ] LLM信号正确聚合（多新闻合并为日频）

### 对比分析验证
- [ ] 五轨道可同时独立运行
- [ ] 对比分析报告生成（表格 + 图表）
- [ ] 回答核心假设：
  - [ ] H1: ML Tracks 收益能力 vs LLM Tracks
  - [ ] H2: LLM Tracks 风险控制（黑天鹅事件）
  - [ ] H3: 成本效益比（Cost-per-Alpha）
- [ ] 图表显示五条资金曲线
- [ ] 图表标注最佳性能轨道

---

## 📝 代码审查要点

修复后的代码必须满足：

1. **时间戳一致性**
   ```python
   # 所有信号必须使用 pd.Timestamp 类型
   assert isinstance(signal['timestamp'], pd.Timestamp)
   ```

2. **日期对齐**
   ```python
   # 信号日期必须是交易日
   assert signal_date in ohlcv_dates
   ```

3. **数据完整性**
   ```python
   # 信号数量应与OHLCV行数匹配
   assert len(ml_signals) == len(ohlcv_data)
   ```

4. **无未来函数**
   ```python
   # 信号时间不能晚于交易日
   assert signal['timestamp'] <= trade_date
   ```

---

**修复完成标志**:

1. **五轨道独立运行成功**
   ```bash
   python main.py run --track lr        # ✅ 有交易，非零收益
   python main.py run --track lstm      # ✅ 有交易，非零收益
   python main.py run --track lgb       # ✅ 有交易，非零收益
   python main.py run --track llm-cloud # ✅ 有交易，非零收益
   python main.py run --track llm-local # ✅ 有交易，非零收益
   ```

2. **五轨道对比分析成功**
   ```bash
   python main.py run --track all --compare
   # ✅ 生成五轨道对比报告
   # ✅ 回答三个核心假设（H1/H2/H3）
   # ✅ 图表显示五条资金曲线
   ```

3. **数据质量验证通过**
   - 所有信号日期与交易日对齐
   - 无未来函数
   - ML使用真实模型（非随机）

**测试数据建议**: 使用 `2025-03-01` 到 `2026-02-28` 的数据（与缓存文件日期重叠最多）。

---

## 📚 附录：五轨道设计原理

### 为什么是这五个轨道？

| 轨道 | 代表 | 核心问题 |
|------|------|----------|
| **LR** | 线性基线 | 最简单的拟合能赚钱吗？ |
| **LSTM** | 序列建模 | 时序特性有用吗？ |
| **LightGBM** | 集成学习 | 树模型适合量化吗？ |
| **LLM(Cloud)** | 云端智能 | 高质量推理的价值？ |
| **LLM(Local)** | 本地智能 | 私有化部署可行吗？ |

### 对比维度矩阵

```
                    收益能力    风险控制    执行速度    运营成本
                    ─────────────────────────────────────────────
LR                    ★★★       ★★         ★★★★★      ★★★★★
LSTM                  ★★★★      ★★★        ★★★        ★★★★★
LightGBM              ★★★★      ★★★        ★★★★       ★★★★★
LLM(Cloud)            ★★        ★★★★       ★          ★★
LLM(Local)            ★★        ★★★        ★          ★★★★
                    ─────────────────────────────────────────────
```

### 论文核心贡献

通过五轨道对比，论文将回答：

1. **拟合 vs 推理**：传统ML和LLM哪种范式更适合量化交易？
2. **速度 vs 智能**：低延迟的拟合 vs 高延迟的推理，哪个更优？
3. **云端 vs 本地**：LLM的智能是否值得额外的API成本？
4. **黑天鹅事件**：LLM的语义理解能否在极端行情中提供保护？

> "DualTrack 不是交易策略，而是量化交易技术对比的**工程基础设施（Testbed）**。
> 它建立了一个公平的竞技场，让'拟合'与'推理'在同一个时间线上展开对决。"
