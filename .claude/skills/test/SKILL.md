# Test Skill

运行测试套件并生成覆盖率报告。

## 触发词
`/test`

## 用途
执行项目测试，验证代码正确性和功能完整性。

## 使用方法

```bash
# 运行所有测试
/test

# 运行指定模块测试
/test --module data

# 运行特定测试文件
/test tests/test_ml_track.py

# 带覆盖率报告
/test --coverage

# 快速测试（跳过慢速测试）
/test --fast

# 详细输出
/test --verbose
```

## 测试模块

### Phase 1: 数据层测试 (`test_data_module.py`)
- ✅ OHLCV 数据获取
- ✅ 新闻数据生成
- ✅ 数据对齐与缺失值处理

### Phase 2: ML Track 测试 (`test_ml_track.py`)
- ✅ 未来函数审计
- ✅ 特征计算正确性
- ✅ MPS 设备检测
- ✅ 模型训练与预测

### Phase 3: LLM Track 测试 (`test_llm_track.py`)
- ✅ JSON 解析容错
- ✅ 执行器连通性
- ✅ 缓存机制验证

### Phase 4: 编排器测试 (`test_orchestrator.py`)
- ✅ 信号融合逻辑
- ✅ 一票否决机制
- ✅ 调仓死区验证

### Phase 5: 回测引擎测试 (`test_bt_engine.py`)
- ✅ 订单成交验证
- ✅ 分析器提取
- ✅ 策略执行逻辑

### Phase 6: 评估测试 (`test_evaluation.py`)
- ✅ 指标计算正确性
- ✅ 图表落盘检查

## 输出示例

```
========================================
  运行测试套件
========================================

tests/test_data_module.py ............                          [ 12%]
tests/test_ml_track.py .................                         [ 29%]
tests/test_llm_track.py ..............                           [ 44%]
tests/test_orchestrator.py .............                         [ 59%]
tests/test_bt_engine.py ............                              [ 71%]
tests/test_evaluation.py ..........                               [ 82%]

========================================
  测试摘要
========================================
✅ 通过: 58
❌ 失败: 2
⚠️ 跳过: 3
⏱️ 耗时: 12.34s

❌ 失败测试:
1. test_orchestrator.py::test_veto_mechanism
   原因: 期望仓位 -1.0，实际 0.8

2. test_llm_track.py::test_concurrent_cache_write
   原因: 并发写入导致数据损坏
```

## 覆盖率报告

使用 `--coverage` 生成覆盖率报告：

```
----------- Coverage Report -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
src/data/__init__.py                  5      0   100%
src/data/market_data.py             120     12    90%
src/data/news_data.py                45      3    93%
src/data/data_aligner.py             38      2    95%
src/models/ml_track/features.py     215     18    92%
src/models/ml_track/baselines.py    180     15    92%
src/models/llm_track/agent.py       156     20    87%
src/orchestrator/fusion_engine.py   145     10    93%
src/execution/bt_engine.py          230     18    92%
src/evaluation/metrics_calculator.py 95      8    92%
-----------------------------------------------------
TOTAL                              1229    106    91%

✅ HTML 报告已生成: htmlcov/index.html
```

## 关键测试场景

### 未来函数审计
```python
def test_no_future_function():
    """验证特征计算未使用未来数据。"""
    # 生成测试数据
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    ohlcv = pd.DataFrame(...)

    # 计算特征
    features = compute_features(ohlcv)

    # 验证：t 时刻的特征不应依赖 t+1 的数据
    for i in range(len(features) - 1):
        current_features = features.iloc[i]
        future_prices = ohlcv.iloc[i+1:]

        # 如果特征包含未来信息，相关性会异常高
        correlation = compute_correlation(current_features, future_prices)
        assert abs(correlation) < 0.1, "检测到未来函数！"
```

### 一票否决机制
```python
def test_veto_mechanism():
    """测试 LLM 一票否决权。"""
    # ML 信号: +0.8 (强烈买入)
    ml_signal = 0.8

    # LLM 信号: -1.0 (黑天鹅事件，强烈卖出)
    llm_signal = -1.0
    volatility = 0.08  # 8% 波动率（黑天鹅）

    # 融合后的信号应该是 -1.0（LLM 否决）
    final_signal = fusion_engine.fuse(
        ml_signal, llm_signal, volatility
    )

    assert final_signal == -1.0, "一票否决机制失效！"
```

## 性能测试

```bash
# 运行性能基准测试
/test --benchmark

输出:
┌─────────────────┬──────────┬──────────┐
│ 测试项           │ 平均耗时  │ 内存峰值  │
├─────────────────┼──────────┼──────────┤
│ 特征计算(1000天) │ 1.23s    │ 45MB     │
│ LSTM 训练(100轮) │ 8.45s    │ 512MB    │
│ LLM 推理(100次)  │ 12.34s   │ 8MB      │
│ 回测执行(1年)    │ 3.21s    │ 128MB    │
└─────────────────┴──────────┴──────────┘
```

## 持续集成

建议配置 GitHub Actions 自动运行测试：

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Run tests
        run: uv run pytest tests/ --cov=src
```

## 相关命令
- `/backtest` - 运行回测
- `/review` - 代码审查
- `/lint` - 代码风格检查