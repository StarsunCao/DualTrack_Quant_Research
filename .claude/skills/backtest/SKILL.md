# Backtest Skill

运行回测并生成评估报告。

## 触发词
`/backtest`

## 用途
快速执行回测流程，支持单资产和多资产回测。

## 使用方法

```bash
# 运行默认回测（CSI300）
/backtest

# 指定交易标的
/backtest --symbol NASDAQ100

# 多资产回测
/backtest --symbol CSI300,NASDAQ100

# 指定回测时间段
/backtest --start 2023-01-01 --end 2024-01-01

# 使用离线缓存
/backtest --use-cache

# 完整回测流程（包括评估图表生成）
/backtest --full
```

## 回测流程

### Phase 1: 数据获取
- 下载 OHLCV 价格数据
- 生成/加载新闻数据
- 数据对齐与清洗

### Phase 2: ML Track
- 计算技术指标特征
- 训练 ML 模型（LSTM/LightGBM/LR）
- 生成 ML 信号

### Phase 3: LLM Track
- 生成 CoT 提示词
- 调用 LLM API（或使用缓存）
- 生成 LLM 信号

### Phase 4: 信号融合
- 根据波动率分配权重
- 应用一票否决机制
- 生成目标仓位

### Phase 5: 回测执行
- Backtrader 引擎执行
- 记录交易和资产曲线
- 计算绩效指标

### Phase 6: 评估输出
- 生成资金曲线图
- 计算风险指标
- 输出评估报告

## 输出内容

```
============================================================
  回测结果摘要
============================================================
  初始资金: 100,000.00
  最终资产: 125,342.56
  总收益率: 25.34%
  年化收益率: 12.67%
------------------------------------------------------------
  夏普比率: 1.2345
  最大回撤: 8.45%
  回撤持续期: 15 天
============================================================
```

## 配置选项

回测参数可通过 `.claude/backtest_config.json` 配置：

```json
{
  "initial_cash": 100000,
  "commission": 0.0002,
  "slippage": 0.001,
  "stamp_duty": 0.001,
  "rebalance_freq": 5,
  "dead_zone_threshold": 0.05,
  "llm_weight_normal": 0.3,
  "llm_weight_high_vol": 0.5,
  "llm_weight_black_swan": 1.0
}
```

## 结果存储

- 回测日志: `docs/output/backtest_YYYYMMDD_HHMMSS.log`
- 资产曲线: `docs/figures/equity_curves.png`
- 回撤热力图: `docs/figures/drawdown_heatmap.png`
- LLM 缓存: `docs/cache/llm_responses/*.jsonl`

## 性能优化

### Apple Silicon Mac
自动使用 MPS 加速 LSTM 训练：
```bash
✅ 使用 Apple Silicon MPS 加速
```

### 离线缓存加速
使用 `--use-cache` 跳过 LLM API 调用：
```bash
🚀 从缓存加载 LLM 响应: 1,234 条记录
加速比: 125x
```

## 错误处理

### 数据下载失败
```
⚠️ 沪深300数据获取失败，尝试使用本地缓存...
✅ 成功加载本地数据: data/raw/csi300_daily_20240228.parquet
```

### LLM API 错误
```
⚠️ DeepSeek API 调用失败 (超时)
回退至 Mock 执行器
```

## 相关命令
- `/test` - 运行测试验证
- `/review` - 代码审查
- `/evaluate` - 生成评估图表