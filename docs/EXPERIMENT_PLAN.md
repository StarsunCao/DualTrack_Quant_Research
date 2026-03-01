# DualTrack Quant Research - 实验执行计划

**版本**: 1.0
**日期**: 2026-03-01

---

## 1. 实验目标

验证以下核心假设：

1. **H1**: DualTrack 融合策略的夏普比率优于单一 ML Track
2. **H2**: 在黑天鹅事件中，LLM 的风险识别能力能降低最大回撤
3. **H3**: LLM Track 的 Cost-per-Alpha 显著高于 ML Track，但风险调整后收益更优

---

## 2. 实验配置

### 2.1 数据配置

```yaml
# config/data_config.yaml
symbols:
  - name: "CSI300"
    source: "akshare"
    start_date: "2020-01-01"
    end_date: "2024-01-01"
    description: "沪深300指数，代表A股大盘"

  - name: "QQQ"
    source: "yfinance"
    start_date: "2020-01-01"
    end_date: "2024-01-01"
    description: "NASDAQ-100 ETF，代表美股科技股"

news_config:
  source: "mock"  # 或 "real" 如果有真实数据
  frequency: "daily"
  sentiment_distribution:
    positive: 0.3
    negative: 0.3
    neutral: 0.4
```

### 2.2 ML Track 配置

```yaml
# config/ml_config.yaml
feature_engineering:
  windows: [5, 10, 20, 60]
  normalize: true
  drop_na: true

models:
  logistic_regression:
    max_iter: 1000
    C: 1.0

  lstm:
    hidden_dim: 64
    num_layers: 2
    dropout: 0.2
    learning_rate: 0.001
    batch_size: 32
    epochs: 50
    sequence_length: 20
    device: "mps"  # Apple Silicon加速

  lightgbm:
    n_estimators: 100
    max_depth: 6
    learning_rate: 0.1

training:
  test_size: 0.2
  random_state: 42
  cv_folds: 5
```

### 2.3 LLM Track 配置

```yaml
# config/llm_config.yaml
executors:
  ollama:
    model: "qwen2.5:7b"
    base_url: "http://localhost:11434"
    temperature: 0.7
    timeout: 120

  deepseek:
    model: "deepseek-chat"
    temperature: 0.7
    timeout: 60

cache:
  enabled: true
  directory: "docs/cache/llm_responses"
  format: "jsonl"

prompt:
  use_cot: true  # Chain-of-Thought
  language: "zh"
```

### 2.4 融合引擎配置

```yaml
# config/fusion_config.yaml
market_regimes:
  normal:
    volatility_threshold: 0.02
    ml_weight: 0.7
    llm_weight: 0.3

  high_volatility:
    volatility_threshold: 0.05
    ml_weight: 0.5
    llm_weight: 0.5

  black_swan:
    volatility_threshold: 0.10
    ml_weight: 0.0
    llm_weight: 1.0
    llm_veto_enabled: true

optimization:
  rebalance_threshold: 0.05  # 5%调仓死区
  llm_signal_decay_hours: 72  # 3天信号衰减
  decay_curve: "linear"
```

### 2.5 回测配置

```yaml
# config/backtest_config.yaml
initial_cash: 100000.0
commission: 0.0002  # 万分之二
slippage: 0.0
stamp_duty: 0.0  # 美股无印花税，A股卖出0.1%

rebalance:
  frequency: 1  # 每日调仓
  execution_time: "market_close"  # 收盘调仓

risk_management:
  max_position_size: 1.0  # 满仓
  stop_loss: null  # 暂时不设止损
```

---

## 3. 实验流程

### 3.1 Phase 8: 数据准备与模型训练

#### Step 1: 数据获取与验证

```bash
# 创建数据目录
mkdir -p data/raw data/processed

# 运行数据模块测试
uv run python tests/test_data_module.py

# 获取真实数据（如果需要）
uv run python -c "
from src.data.market_data import MarketDataFetcher
fetcher = MarketDataFetcher()

# 获取沪深300数据
csi300 = fetcher.fetch_csi300(
    start_date='2020-01-01',
    end_date='2024-01-01',
    save_to_file=True
)
print(f'CSI300: {len(csi300)} rows')

# 获取QQQ数据
qqq = fetcher.fetch_qqq(
    start_date='2020-01-01',
    end_date='2024-01-01',
    save_to_file=True
)
print(f'QQQ: {len(qqq)} rows')
"
```

#### Step 2: ML Track 训练

```python
# scripts/train_ml_models.py
import pandas as pd
from src.models.ml_track.features import FeatureEngineer
from src.models.ml_track.baselines import MLStrategyPortfolio

# 加载数据
df = pd.read_parquet("data/raw/csi300_daily_*.parquet")

# 特征工程
engineer = FeatureEngineer()
features_df = engineer.compute_all_features(df, drop_na=True)
features_df = engineer.create_target(features_df, forward_period=1)
features_df = features_df.dropna()

# 保存特征数据
features_df.to_parquet("data/processed/csi300_features.parquet")

# 训练模型
portfolio = MLStrategyPortfolio(
    lstm_hidden_dim=64,
    lstm_num_layers=2,
    lstm_epochs=50,
    lstm_sequence_length=20,
    lgb_n_estimators=100,
    lgb_max_depth=6,
)

portfolio.fit(features_df, target_col="target_label", test_size=0.2)

# 保存模型
portfolio.save_models("models/csi300_ml")

# 输出评估指标
metrics_df = portfolio.get_metrics()
print(metrics_df)
metrics_df.to_csv("docs/output/ml_metrics.csv", index=False)
```

#### Step 3: LLM Track 缓存构建

```bash
# 生成或加载新闻数据
uv run python -c "
from src.data.news_data import MockNewsGenerator
import pandas as pd

generator = MockNewsGenerator()
news_df = generator.generate_mock_news(
    start_date='2020-01-01',
    end_date='2024-01-01',
    symbols=['CSI300'],
    num_per_day=2
)
news_df.to_csv('data/raw/csi300_news_4y.csv', index=False)
print(f'Generated {len(news_df)} news items')
"

# 构建LLM缓存（使用Ollama本地模型）
uv run python main.py cache-build \
    --symbol CSI300 \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --news-file data/raw/csi300_news_4y.csv \
    --executor ollama \
    --output-dir docs/cache/llm_responses

# 验证缓存
ls -lh docs/cache/llm_responses/
```

### 3.2 Phase 9: 对比实验

#### 实验A: ML Track 基线

```python
# experiments/exp_a_ml_baseline.py
"""
实验A: ML Track 基线
只使用机器学习模型信号，不使用LLM
"""

import pandas as pd
from src.models.ml_track.baselines import MLStrategyPortfolio
from src.execution.bt_engine import BacktestEngine, DualTrackStrategy
from src.evaluation.metrics_calculator import MetricsCalculator

# 加载模型
portfolio = MLStrategyPortfolio()
portfolio.load_models("models/csi300_ml", feature_dim=50)

# 加载特征数据
features_df = pd.read_parquet("data/processed/csi300_features.parquet")

# 生成ML信号
ml_signals = portfolio.predict(features_df, symbol="CSI300")

# 转换为目标仓位格式
target_positions = {}
for timestamp, group in ml_signals.groupby("timestamp"):
    # 平均各模型信号
    avg_signal = group["signal_strength_0_to_1"].mean()
    weight = (avg_signal - 0.5) * 2  # 转换到[-1, 1]
    target_positions[timestamp] = {"CSI300": weight}

# 运行回测
ohlcv_data = pd.read_parquet("data/raw/csi300_daily_*.parquet")
engine = BacktestEngine(initial_cash=100000, commission=0.0002)
engine.add_data(ohlcv_data, name="CSI300")
engine.add_strategy(DualTrackStrategy, target_positions=target_positions)
result = engine.run()

# 评估
calculator = MetricsCalculator()
evaluation = calculator.evaluate(
    strategy_name="ML_Baseline",
    equity_curve=result.equity_curve,
    num_signals=len(ml_signals),
)

# 保存结果
evaluation.to_dict()  # 保存到文件
```

#### 实验B: LLM Track 基线

```python
# experiments/exp_b_llm_baseline.py
"""
实验B: LLM Track 基线
只使用LLM信号，不使用ML
"""

import pandas as pd
from src.models.llm_track.agent import LLMTradingAgent
from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

# 加载缓存的LLM信号
llm_signals = pd.read_json("docs/cache/llm_responses/llm_cache_CSI300.jsonl", lines=True)

# 转换为目标仓位
target_positions = {}
for _, row in llm_signals.iterrows():
    timestamp = pd.to_datetime(row["timestamp"])
    signal_map = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
    weight = signal_map.get(row["signal"], 0.0) * row["confidence"]
    target_positions[timestamp] = {"CSI300": weight}

# 运行回测（同上）
# ...
```

#### 实验C-E: 融合策略

```python
# experiments/exp_cde_fusion.py
"""
实验C-E: 不同融合策略对比
"""

from src.orchestrator.fusion_engine import SignalFusionEngine, MarketRegime

configs = {
    "Fixed_7030": {"ml_weight_normal": 0.7, "dynamic": False},
    "Dynamic": {"ml_weight_normal": 0.7, "dynamic": True},
    "LLM_Veto": {"ml_weight_normal": 0.7, "llm_veto": True},
}

results = {}

for name, config in configs.items():
    fusion_engine = SignalFusionEngine(
        ml_weight_normal=config["ml_weight_normal"],
        # ... 其他参数
    )

    # 融合信号
    target_positions = fusion_engine.generate_target_positions(
        ml_signals=ml_signals,
        llm_signals=llm_signals,
        volatility=0.02,
    )

    # 运行回测
    # ...

    results[name] = evaluation

# 对比结果
compare_results(results)
```

### 3.3 Phase 10: 结果分析

#### 自动化报告生成

```python
# scripts/generate_report.py
from src.evaluation.visualizer import (
    plot_equity_curves,
    plot_drawdown_heatmap,
    plot_latency_boxplot,
    plot_signal_correlation,
)
from src.evaluation.metrics_calculator import MultiStrategyComparator

# 加载所有实验结果
experiments = {
    "ML_Baseline": load_result("exp_a"),
    "LLM_Baseline": load_result("exp_b"),
    "Fixed_7030": load_result("exp_c"),
    "Dynamic": load_result("exp_d"),
    "LLM_Veto": load_result("exp_e"),
}

# 生成对比表格
comparator = MultiStrategyComparator()
for name, result in experiments.items():
    comparator.add_result(name, result)

financial_df = comparator.compare_financial_metrics()
engineering_df = comparator.compare_engineering_metrics()

# 保存表格
financial_df.to_csv("docs/output/financial_comparison.csv")
engineering_df.to_csv("docs/output/engineering_comparison.csv")

# 生成LaTeX表格
latex_table = comparator.generate_latex_table(metric_type="financial")
with open("docs/output/tables.tex", "w") as f:
    f.write(latex_table)

# 生成图表
equity_curves = {name: r.equity_curve for name, r in experiments.items()}
plot_equity_curves(
    equity_curves,
    title="Strategy Comparison: Equity Curves",
    save_path="docs/figures/equity_curves_comparison.png",
)

# 更多图表...
```

---

## 4. main.py 修改计划

### 4.1 当前问题

当前 `main.py` 使用模拟信号 (`_generate_mock_ml_signals`, `_generate_mock_llm_signals`)，需要修改为使用真实模型。

### 4.2 修改方案

```python
# main.py 修改要点

# Phase 2: ML Track 信号生成 - 修改为使用真实模型
@click.echo("\n[Phase 2/6] ML Track 信号生成...")

try:
    from src.models.ml_track.features import FeatureEngineer
    from src.models.ml_track.baselines import MLStrategyPortfolio

    # 特征工程
    feature_engineer = FeatureEngineer()
    features = feature_engineer.compute_all_features(aligned_data["ohlcv"])

    # 尝试加载已有模型
    model_path = f"models/{symbol}_ml"
    portfolio = MLStrategyPortfolio()

    if Path(model_path).exists():
        # 加载已有模型
        portfolio.load_models(model_path, feature_dim=len(feature_engineer.feature_names))
        click.echo(f"  ✅ 加载已有模型: {model_path}")
    else:
        # 训练新模型
        click.echo("  🚀 训练新模型...")
        features_with_target = feature_engineer.create_target(features, forward_period=1)
        features_with_target = features_with_target.dropna()
        portfolio.fit(features_with_target, target_col="target_label")
        portfolio.save_models(model_path)

    # 生成信号
    ml_signals = portfolio.predict(features, symbol=symbol)
    click.echo(f"  ✅ ML 信号: {len(ml_signals)} 条")

except Exception as e:
    click.echo(f"  ⚠️ ML Track 失败: {e}")
    ml_signals = _generate_mock_ml_signals(symbol, len(ohlcv_data))


# Phase 3: LLM Track 信号生成 - 修改为优先使用缓存
@click.echo("\n[Phase 3/6] LLM Track 信号生成...")

try:
    from src.models.llm_track.agent import LLMTradingAgent

    cache_file = Path(f"docs/cache/llm_responses/llm_cache_{symbol}.jsonl")

    if cache_file.exists():
        # 使用缓存
        click.echo(f"  使用缓存: {cache_file}")
        llm_signals = pd.read_json(cache_file, lines=True)
    else:
        # 实时推理
        click.echo("  实时推理（无缓存）...")
        llm_agent = LLMTradingAgent(executor_type="ollama")

        news_list = aligned_data.get("news", pd.DataFrame()).to_dict("records")
        llm_signals = llm_agent.batch_analyze(
            news_list=news_list,
            symbol=symbol,
            cache_path=cache_file,
        )

    click.echo(f"  ✅ LLM 信号: {len(llm_signals)} 条")

except Exception as e:
    click.echo(f"  ⚠️ LLM Track 失败: {e}")
    llm_signals = _generate_mock_llm_signals(symbol, len(ohlcv_data))
```

---

## 5. 实验检查清单

### 实验前检查

- [ ] Python 3.12+ 环境配置正确
- [ ] `uv sync` 安装所有依赖成功
- [ ] MPS 设备检测通过 (`torch.backends.mps.is_available()`)
- [ ] Ollama 服务运行中 (`ollama serve`)
- [ ] 磁盘空间充足 (>10GB)
- [ ] 网络连接正常（用于获取数据）

### 实验中检查

- [ ] 数据获取无缺失值
- [ ] 特征工程无未来函数
- [ ] ML模型训练损失收敛
- [ ] LLM缓存生成无错误
- [ ] 回测订单正常成交

### 实验后检查

- [ ] 资金曲线合理（无异常跳跃）
- [ ] 夏普比率在合理范围（-2到5）
- [ ] 最大回撤 < 50%
- [ ] 交易次数 > 20次（统计显著）
- [ ] 图表生成成功且清晰

---

## 6. 输出文件结构

```
docs/output/
├── experiments/              # 实验结果
│   ├── exp_a_ml_baseline/    # ML基线
│   ├── exp_b_llm_baseline/   # LLM基线
│   ├── exp_c_fixed_7030/     # 固定权重
│   ├── exp_d_dynamic/        # 动态权重
│   └── exp_e_llm_veto/       # LLM否决
├── figures/                  # 论文图表
│   ├── equity_curves_comparison.png
│   ├── drawdown_heatmap.png
│   ├── latency_boxplot.png
│   ├── signal_correlation.png
│   └── regime_switching.png
├── tables/                   # 数据表格
│   ├── financial_metrics.csv
│   ├── engineering_metrics.csv
│   └── latex_tables.tex
└── report.md                 # 实验报告
```

---

## 7. 时间表与里程碑

| 日期 | 任务 | 交付物 |
|------|------|--------|
| Day 1 | 数据获取与清洗 | `data/raw/*.parquet` |
| Day 2 | 特征工程 | `data/processed/*_features.parquet` |
| Day 3 | ML模型训练 | `models/*_ml*.pkl` |
| Day 4 | LLM缓存构建 | `docs/cache/llm_responses/*.jsonl` |
| Day 5 | 端到端调试 | 成功运行的回测 |
| Day 6 | 对比实验A-C | 实验结果CSV |
| Day 7 | 对比实验D-E | 实验结果CSV |
| Day 8 | 可视化图表 | `docs/figures/*.png` |
| Day 9 | 报告撰写 | `docs/output/report.md` |
| Day 10 | 复盘与优化 | 优化后的代码 |

---

**备注**: 本计划应根据实际执行情况动态调整，建议每日更新进度。
