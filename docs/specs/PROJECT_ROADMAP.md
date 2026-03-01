# DualTrack Quant Research - 项目详细规划书

**版本**: 1.0
**日期**: 2026-03-01
**状态**: Phase 1-7 已完成，进入实验阶段

---

## 1. 项目现状评估

### 1.1 已完成的模块 (Phase 1-7)

| Phase | 模块 | 文件路径 | 状态 | 测试覆盖 |
|-------|------|----------|------|----------|
| 1 | 数据获取层 | `src/data/market_data.py` | ✅ 完成 | `tests/test_data_module.py` |
| 1 | 新闻生成 | `src/data/news_data.py` | ✅ 完成 | `tests/test_data_module.py` |
| 1 | 数据对齐 | `src/data/data_aligner.py` | ✅ 完成 | `tests/test_data_module.py` |
| 2 | 特征工程 | `src/models/ml_track/features.py` | ✅ 完成 | `tests/test_ml_track.py` |
| 2 | ML模型 | `src/models/ml_track/baselines.py` | ✅ 完成 | `tests/test_ml_track.py` |
| 3 | Prompt模板 | `src/models/llm_track/prompts.py` | ✅ 完成 | `tests/test_llm_track.py` |
| 3 | LLM代理 | `src/models/llm_track/agent.py` | ✅ 完成 | `tests/test_llm_track.py` |
| 4 | 信号融合 | `src/orchestrator/fusion_engine.py` | ✅ 完成 | `tests/test_orchestrator.py` |
| 5 | 回测引擎 | `src/execution/bt_engine.py` | ✅ 完成 | `tests/test_bt_engine.py` |
| 6 | 指标计算 | `src/evaluation/metrics_calculator.py` | ✅ 完成 | `tests/test_evaluation.py` |
| 6 | 可视化 | `src/evaluation/visualizer.py` | ✅ 完成 | `tests/test_evaluation.py` |
| 7 | CLI入口 | `main.py` | ✅ 完成 | 集成测试 |
| 7 | Docker部署 | `Dockerfile`, `docker-compose.yml` | ✅ 完成 | - |

### 1.2 代码质量评估

**优势**:
- ✅ 完整的技术指标计算（50+ 因子）
- ✅ 严格防止未来函数（shift操作审计通过）
- ✅ Apple Silicon MPS 优化
- ✅ JSON解析容错机制完善
- ✅ 离线缓存架构（加速比 >100x）
- ✅ 双轨融合逻辑完整（Normal/HighVol/BlackSwan三种状态）
- ✅ 调仓死区与信号衰减优化
- ✅ 完整的 CLI 接口

**待优化项**:
- ⚠️ main.py 目前使用模拟信号，需要接入真实模型
- ⚠️ 缺乏多资产组合回测支持
- ⚠️ 可视化模块图表类型有限
- ⚠️ 缺乏实时监控/日志系统

---

## 2. 下一阶段目标 (Phase 8-10)

### Phase 8: 真实数据实验与模型训练 (优先级: P0)

**目标**: 使用真实市场数据和新闻数据训练模型，完成端到端回测

#### 8.1 数据准备
- [ ] 获取2020-2024年沪深300历史数据
- [ ] 获取2020-2024年QQQ历史数据
- [ ] 收集/生成对应时间段的新闻语料
- [ ] 数据质量检查与清洗

#### 8.2 ML Track 训练
- [ ] 特征工程管道优化
- [ ] 训练 LogisticRegression 基线
- [ ] 训练 LSTM 序列模型（MPS加速）
- [ ] 训练 LightGBM 模型
- [ ] 交叉验证与超参数调优
- [ ] 模型持久化（保存/加载）

#### 8.3 LLM Track 缓存构建
- [ ] 使用 `cache-build` 命令预处理所有新闻
- [ ] 对比 Ollama 本地模型 vs DeepSeek API 效果
- [ ] 缓存质量验证

#### 8.4 端到端回测
- [ ] 修改 main.py 使用真实模型信号
- [ ] 运行完整回测流水线
- [ ] 结果验证与调试

### Phase 9: 多维度对比实验 (优先级: P0)

**目标**: 生成论文所需的对比分析数据

#### 9.1 实验设计

| 实验组 | ML Track | LLM Track | 融合策略 | 目的 |
|--------|----------|-----------|----------|------|
| A | ✅ | ❌ | - | ML 基线 |
| B | ❌ | ✅ | - | LLM 基线 |
| C | ✅ | ✅ | 固定权重 70/30 | 简单融合 |
| D | ✅ | ✅ | 动态权重 | 智能融合 |
| E | ✅ | ✅ | LLM否决权 | 风险控制 |

#### 9.2 需要记录的指标

**金融指标**:
- Sharpe Ratio
- Maximum Drawdown
- Win Rate
- Sortino Ratio
- Calmar Ratio
- Alpha/Beta

**工程指标**:
- Average Latency (ms)
- P95/P99 Latency
- Throughput (signals/sec)
- Total Tokens
- API Cost ($)
- Cost-per-Alpha
- Cache Hit Rate

#### 9.3 场景测试
- [ ] 正常市场条件 (2020-2021)
- [ ] 高波动市场 (2022俄乌冲突)
- [ ] 极端行情 (2020疫情暴跌/反弹)
- [ ] 单边上涨/下跌市场

### Phase 10: 论文图表与报告生成 (优先级: P1)

**目标**: 自动生成论文所需的所有图表和数据表格

#### 10.1 新增可视化
- [ ] 多策略资金曲线对比图（已完成基础版）
- [ ] 回撤热力图（季度维度）
- [ ] 延迟分布箱线图（已完成基础版）
- [ ] 信号相关性矩阵图
- [ ] 权重变化时序图
- [ ] 市场状态切换可视化
- [ ] ROC曲线对比

#### 10.2 自动化报告
- [ ] LaTeX表格自动生成
- [ ] Markdown报告模板
- [ ] 实验配置版本控制

---

## 3. 技术债务与优化

### 3.1 高优先级优化

| 问题 | 影响 | 解决方案 | 预计工作量 |
|------|------|----------|------------|
| main.py 使用模拟信号 | 无法评估真实效果 | 接入真实模型 | 2天 |
| 缺乏多资产支持 | 只能单资产回测 | 扩展数据结构和策略 | 3天 |
| 可视化图表有限 | 论文图表不足 | 新增图表类型 | 2天 |

### 3.2 中优先级优化

| 问题 | 影响 | 解决方案 | 预计工作量 |
|------|------|----------|------------|
| 缺乏实时监控 | 难以观察运行状态 | 添加日志和进度条 | 1天 |
| 缓存清理机制 | 缓存文件可能过大 | 添加LRU清理策略 | 1天 |
| 回测结果存储 | 结果易丢失 | 添加结果序列化 | 1天 |

### 3.3 低优先级优化

- 分布式训练支持
- Web UI 可视化界面
- 实时交易接口（非回测）

---

## 4. 实验时间表

### Week 1: 数据准备与模型训练
- Day 1-2: 获取真实历史数据
- Day 3-4: ML Track 模型训练
- Day 5-7: LLM Track 缓存构建

### Week 2: 端到端集成与调试
- Day 1-2: 修改 main.py 接入真实模型
- Day 3-4: 端到端回测调试
- Day 5-7: 多组对比实验运行

### Week 3: 结果分析与可视化
- Day 1-2: 新增可视化图表
- Day 3-4: 自动化报告生成
- Day 5-7: 论文图表优化与定稿

### Week 4: 论文撰写与代码整理
- Day 1-3: 论文初稿撰写
- Day 4-5: 代码文档完善
- Day 6-7: 开源准备（README、LICENSE等）

---

## 5. 关键里程碑

| 里程碑 | 完成标准 | 截止日期 |
|--------|----------|----------|
| M1 | 真实数据获取完成，数据质量检查通过 | Week 1 Day 2 |
| M2 | ML Track 模型训练完成，指标达标 | Week 1 Day 5 |
| M3 | LLM Track 缓存构建完成 | Week 1 Day 7 |
| M4 | 端到端回测成功运行 | Week 2 Day 4 |
| M5 | 5组对比实验数据收集完成 | Week 2 Day 7 |
| M6 | 论文图表全部生成 | Week 3 Day 4 |
| M7 | 论文初稿完成 | Week 4 Day 3 |
| M8 | 代码开源准备完成 | Week 4 Day 7 |

---

## 6. 风险评估与应对

| 风险 | 可能性 | 影响 | 应对策略 |
|------|--------|------|----------|
| 数据获取失败 | 低 | 高 | 准备备用数据源，使用模拟数据降级 |
| LLM API 成本过高 | 中 | 中 | 优先使用 Ollama 本地模型，限制 API 调用次数 |
| 模型效果不佳 | 中 | 高 | 调整特征工程，尝试更多超参数组合 |
| MPS 设备兼容性问题 | 低 | 中 | 提供 CPU 降级方案 |
| 回测过拟合 | 高 | 高 | 严格的时间序列交叉验证，使用 OOS 测试 |

---

## 7. 资源需求

### 7.1 计算资源
- Apple Silicon Mac (M1/M2/M3) - MPS 加速
- 内存: 16GB+
- 存储: 10GB+ (数据 + 缓存)

### 7.2 API 资源
- DeepSeek API Key (可选，优先使用 Ollama)
- Ollama 本地模型 (qwen2.5:7b 或更大)

### 7.3 数据资源
- 沪深300历史数据 (akshare免费)
- QQQ历史数据 (yfinance免费)
- 新闻数据 (需爬取或使用已有数据集)

---

## 8. 成功标准

### 8.1 功能完成标准
- [ ] 真实数据回测成功运行
- [ ] 5组对比实验数据完整
- [ ] 论文图表全部生成
- [ ] 代码通过所有单元测试

### 8.2 效果验证标准
- [ ] DualTrack 融合策略 Sharpe > ML 基线
- [ ] 黑天鹅事件中 LLM 否决机制有效降低回撤
- [ ] 工程指标（延迟、成本）可量化对比

### 8.3 论文产出标准
- [ ] 完整的实验结果表格
- [ ] 高质量的资金曲线对比图
- [ ] 可复现的实验代码

---

## 9. 附录

### 9.1 快速启动命令

```bash
# 1. 安装依赖
uv sync

# 2. 运行单元测试
uv run python -m pytest tests/

# 3. 构建 LLM 缓存
uv run python main.py cache-build --symbol CSI300 --start 2020-01-01 --end 2024-01-01

# 4. 执行完整回测
uv run python main.py run --symbol CSI300 --start 2020-01-01 --end 2024-01-01

# 5. 生成评估图表
uv run python main.py evaluate
```

### 9.2 关键文件索引

| 目的 | 文件路径 |
|------|----------|
| 主入口 | `main.py` |
| ML模型训练 | `src/models/ml_track/baselines.py` |
| LLM代理 | `src/models/llm_track/agent.py` |
| 信号融合 | `src/orchestrator/fusion_engine.py` |
| 回测引擎 | `src/execution/bt_engine.py` |
| 指标计算 | `src/evaluation/metrics_calculator.py` |
| 数据获取 | `src/data/market_data.py` |

### 9.3 参考文档
- Backtrader 文档: https://www.backtrader.com/docu/
- PyTorch MPS: https://pytorch.org/docs/stable/notes/mps.html
- Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
- DeepSeek API: https://platform.deepseek.com/api-docs

---

**规划书维护**: 每周更新进度，标记已完成项
**负责人**: [待填写]
**审核人**: [待填写]
