# DualTrack Quant Research - 下一步执行计划

**版本**: 1.0
**日期**: 2026-03-01
**当前状态**: Phase 1-7 已完成，基础设施就绪

---

## 决策点：选择下一步方向

项目已完成基础架构搭建，现在面临关键决策：

```
┌─────────────────────────────────────────────────────────────────┐
│                        下一步选择                                │
├─────────────────────────┬───────────────────────────────────────┤
│  A. 接入真实数据实验      │  B. 完善技术架构                      │
│  (科研导向)              │  (工程导向)                           │
├─────────────────────────┼───────────────────────────────────────┤
│ • 获取2020-2024真实数据  │ • 接入真实ML/LLM模型到main.py         │
│ • 训练ML模型            │ • 完善错误处理和日志系统               │
│ • 构建LLM缓存           │ • 添加实时监控进度条                  │
│ • 运行对比实验          │ • 多资产组合支持                      │
│ • 生成论文图表          │ • 性能优化                           │
├─────────────────────────┼───────────────────────────────────────┤
│ 适合: 论文 deadline 紧迫 │ 适合: 追求代码质量，长期维护           │
│ 风险: main.py可能不稳定  │ 风险: 延迟实验进度                    │
└─────────────────────────┴───────────────────────────────────────┘
```

---

## 推荐方案：A+B 并行 (短期+中期)

### 第一阶段：接入真实模型 (2-3天)

**目标**: 让 main.py 能够使用真实模型运行回测

#### Task 1: 修改 main.py 接入真实ML模型
**优先级**: P0
**预估时间**: 1天

当前问题:
```python
# main.py 目前使用模拟信号
def _generate_mock_ml_signals(symbol: str, n: int) -> pd.DataFrame:
    ...
```

修改方案:
1. 在 Phase 2 添加真实ML模型调用:
   - 检查 `models/{symbol}_ml/` 是否存在已训练模型
   - 若存在: 加载模型并生成信号
   - 若不存在: 训练新模型并保存

2. 新增命令行参数:
   ```bash
   python main.py run --track lr --symbol CSI300 --train  # 强制重新训练
   python main.py run --track lr --symbol CSI300 --use-cache  # 使用已有模型
   ```

**验收标准**:
- [ ] main.py 可以加载/训练真实ML模型
- [ ] 生成的信号不再是随机数
- [ ] 模型文件保存到 `models/` 目录

---

#### Task 2: 修改 main.py 接入真实LLM缓存
**优先级**: P0
**预估时间**: 0.5天

当前问题:
```python
def _generate_mock_llm_signals(symbol: str, n: int) -> pd.DataFrame:
    ...
```

修改方案:
1. 优先检查缓存文件 `docs/cache/llm_responses/llm_cache_{symbol}.jsonl`
2. 缓存存在: 直接加载
3. 缓存不存在: 调用 LLMTradingAgent 实时推理

**验收标准**:
- [ ] main.py 优先使用缓存
- [ ] 缓存不存在时能自动调用LLM
- [ ] 实时推理结果自动保存到缓存

---

#### Task 3: 数据获取与预处理
**优先级**: P0
**预估时间**: 0.5天

当前状态:
- 已有 `data/raw/real_csi300_1y.csv` (1年数据)

需要完成:
1. 获取2020-2024年完整数据:
   ```python
   # 使用已有模块
   from src.data.market_data import MarketDataFetcher
   fetcher = MarketDataFetcher()
   df = fetcher.fetch_csi300("2020-01-01", "2024-01-01", save_to_file=True)
   ```

2. 验证数据质量:
   - 无缺失值
   - 时间连续
   - 价格合理性检查

**验收标准**:
- [ ] `data/raw/csi300_2020_2024.csv` 存在且完整
- [ ] 数据质量检查通过

---

### 第二阶段：工程优化 (2-3天)

#### Task 4: 添加日志与进度监控
**优先级**: P1
**预估时间**: 1天

当前问题:
- 回测过程没有进度显示
- 错误信息不够清晰
- 难以调试

改进方案:
1. 添加进度条 (tqdm):
   ```python
   from tqdm import tqdm
   for i in tqdm(range(n), desc="Processing"):
       ...
   ```

2. 结构化日志 (loguru):
   ```python
   from loguru import logger
   logger.info("Training LSTM model...")
   logger.debug(f"Feature shape: {X.shape}")
   ```

3. 不同级别日志输出到不同文件:
   - `logs/info.log`: 正常流程日志
   - `logs/error.log`: 错误日志
   - `logs/debug.log`: 调试日志

**验收标准**:
- [ ] 训练过程有进度条
- [ ] 关键节点有日志输出
- [ ] 错误信息清晰可定位

---

#### Task 5: 模型持久化与管理
**优先级**: P1
**预估时间**: 1天

当前状态:
- baselines.py 有模型训练代码
- 但缺少统一的保存/加载接口

需要完成:
1. 统一模型管理类:
   ```python
   class ModelManager:
       def save(self, model, path: str)
       def load(self, path: str) -> model
       def list_models(self) -> List[str]
       def get_model_info(self, path: str) -> dict
   ```

2. 模型版本控制:
   - 保存训练时间、参数、指标
   - 支持加载特定版本

**验收标准**:
- [ ] 模型可以保存和加载
- [ ] 支持版本管理
- [ ] 加载时能复现训练时的指标

---

#### Task 6: 回测结果持久化
**优先级**: P1
**预估时间**: 0.5天

当前状态:
- 回测结果只在内存中
- 程序退出后丢失

改进方案:
1. 结果序列化:
   ```python
   # 保存回测结果
   result.save("docs/output/experiments/exp_001_lr_csi300.pkl")

   # 加载回测结果
   result = BacktestResult.load("docs/output/experiments/exp_001_lr_csi300.pkl")
   ```

2. 元数据记录:
   - 实验配置
   - 运行时间
   - Git commit hash
   - 随机种子

**验收标准**:
- [ ] 回测结果可以保存/加载
- [ ] 实验可复现

---

### 第三阶段：对比实验 (3-5天)

#### Task 7: 运行5组对比实验
**优先级**: P0
**预估时间**: 2天

实验设计:

| 实验 | ML | LLM | 融合策略 | 目的 |
|------|-----|-----|---------|------|
| A | LR | - | - | ML基线 |
| B | LSTM | - | - | 序列模型效果 |
| C | LightGBM | - | - | 集成学习效果 |
| D | - | Ollama | - | 本地LLM效果 |
| E | - | DeepSeek | - | 云端LLM效果 |

执行命令:
```bash
# 分别运行5个轨道
python main.py run --track lr --symbol CSI300 --start 2020-01-01 --end 2024-01-01
python main.py run --track lstm --symbol CSI300 --start 2020-01-01 --end 2024-01-01
python main.py run --track lgb --symbol CSI300 --start 2020-01-01 --end 2024-01-01
python main.py run --track llm-local --symbol CSI300 --start 2020-01-01 --end 2024-01-01
python main.py run --track llm-cloud --symbol CSI300 --start 2020-01-01 --end 2024-01-01

# 生成对比报告
python main.py evaluate --compare
```

**验收标准**:
- [ ] 5组实验全部成功运行
- [ ] 每组实验保存完整交易记录
- [ ] 生成对比图表

---

#### Task 8: 黑天鹅事件专项测试
**优先级**: P1
**预估时间**: 1天

测试场景:
- 2020年2-3月: 新冠疫情暴跌
- 2022年2-3月: 俄乌冲突

验证假设:
- H2: LLM 在黑天鹅事件中能降低最大回撤

执行:
```bash
# 测试特定时间段
python main.py run --track llm-local --symbol CSI300 \
    --start 2020-02-01 --end 2020-04-01 \
    --output-dir docs/output/experiments/black_swan_covid
```

**验收标准**:
- [ ] 黑天鹅期间回测完成
- [ ] LLM vs ML 回撤对比数据

---

### 第四阶段：论文准备 (3-5天)

#### Task 9: 完善可视化图表
**优先级**: P1
**预估时间**: 2天

当前已有:
- ✅ 资金曲线对比图
- ⚠️ 回撤热力图 (需改进)
- ⚠️ 延迟箱线图 (需改进)

需要新增:
1. 信号相关性矩阵图
2. 权重变化时序图
3. 市场状态切换可视化
4. ROC曲线对比
5. 年化收益-风险散点图

**验收标准**:
- [ ] 至少8种图表类型
- [ ] 图表质量达到论文要求
- [ ] 图表自动保存到 docs/figures/

---

#### Task 10: 自动化报告生成
**优先级**: P2
**预估时间**: 1天

功能:
```bash
python main.py report --experiment-dir docs/output/experiments/
```

输出:
- `docs/output/report.md`: Markdown报告
- `docs/output/tables.csv`: 数据表格
- `docs/output/tables.tex`: LaTeX表格

**验收标准**:
- [ ] 一键生成完整报告
- [ ] 包含所有关键指标
- [ ] LaTeX表格可直接用于论文

---

## 执行时间表

### Week 1: 接入真实模型 (核心任务)

| 日期 | 任务 | 交付物 | 验收标准 |
|------|------|--------|----------|
| Day 1 | Task 1: 接入ML模型 | `main.py` 修改版 | ML使用真实模型 |
| Day 2 | Task 2: 接入LLM缓存 | `main.py` 完整版 | LLM优先使用缓存 |
| Day 3 | Task 3: 数据获取 | `csi300_2020_2024.csv` | 数据完整无缺失 |
| Day 4 | Task 4: 日志系统 | `logs/` 目录 | 有进度条和日志 |
| Day 5 | Task 5-6: 持久化 | `models/`, `docs/output/experiments/` | 可保存加载 |
| Day 6-7 | 集成测试 | 完整回测一次 | 全流程跑通 |

### Week 2: 对比实验与优化

| 日期 | 任务 | 交付物 |
|------|------|--------|
| Day 1-2 | Task 7: 5组实验 | 5组回测结果 |
| Day 3 | Task 8: 黑天鹅测试 | 黑天鹅专项数据 |
| Day 4-5 | Task 9: 完善图表 | 8+种图表 |
| Day 6-7 | Task 10: 报告生成 | 自动化报告 |

---

## 立即开始的3个任务

如果你想现在开始，建议按以下顺序：

### 🚀 立即开始 (今天)

**Task 1: 验证当前系统能跑通**
```bash
# 1. 运行测试
uv run python -m pytest tests/ -v

# 2. 快速回测验证
python main.py run --track lr --symbol CSI300 --start 2025-03-01 --end 2025-06-01

# 3. 检查输出
ls docs/output/track_results/lr/
```

**Task 2: 获取完整数据**
```bash
# 创建脚本获取2020-2024数据
python -c "
from src.data.market_data import MarketDataFetcher
fetcher = MarketDataFetcher()
df = fetcher.fetch_csi300('2020-01-01', '2024-01-01', save_to_file=True)
print(f'Got {len(df)} rows')
"
```

**Task 3: 阅读关键代码**
- `src/models/ml_track/baselines.py` - 了解ML模型接口
- `src/models/llm_track/agent.py` - 了解LLM代理接口
- 为修改 main.py 做准备

---

## 风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| 数据获取失败 | 低 | 高 | 使用已有1年数据先跑通流程 |
| LLM API 成本过高 | 中 | 中 | 优先使用 Ollama 本地模型 |
| 模型训练效果差 | 中 | 高 | 调整特征工程，增加数据预处理 |
| MPS 兼容性问题 | 低 | 中 | 提供 CPU fallback |
| 回测过拟合 | 高 | 高 | 严格时间序列交叉验证 |

---

## 决策检查点

在开始 Week 2 之前，请确认：

- [ ] Week 1 所有任务已完成
- [ ] 至少成功运行一次完整回测
- [ ] 回测结果合理 (夏普比率在 -2 到 5 之间)
- [ ] 已保存至少一个训练好的模型

如果 Week 1 遇到严重阻塞，可考虑：
- **方案B**: 使用 mock 数据先跑完对比实验框架
- **方案C**: 减少实验组数 (如只做 3 组: ML/LLM/Best)

---

**下一步动作**: 选择 Task 1/2/3 中的任意一个开始执行

**建议**: 先做 Task 1 (验证系统) → 然后 Task 2 (获取数据) → 最后 Task 3 (修改main.py)
