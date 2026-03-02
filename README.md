# DualTrack Quant Research

**五轨道量化交易对比实验平台** —— 严格对比传统机器学习（拟合/Fitting）与大语言模型（推理/Reasoning）在量化交易中的 ROI、鲁棒性与工程可行性。

> **核心定位**: DualTrack 不是融合策略，而是完成对比实验的**工程基础设施（Testbed）**，建立公平竞技场让"拟合"与"推理"在同一时间线上展开对决。

---

## ⚠️ 重要警告：Apple Silicon 用户必读

**Docker for Mac 无法穿透调用 Apple Silicon MPS (Metal Performance Shaders)。**

对于 M 系列芯片 Mac 用户：
- ❌ **不要**在 Docker 中运行本项目的 LSTM 训练
- ✅ **建议**直接使用本地 `uv` 虚拟环境运行回测
- ✅ Docker 镜像仅用于 Linux/CUDA 云端部署

```bash
# Apple Silicon Mac 推荐使用方式
uv sync
uv run python main.py run --track all --compare --symbol CSI300
```

---

## 五轨道架构设计

DualTrack 建立统一的基准测试标准，在**完全相同的市场条件**下，量化五种截然不同的技术在财务表现与工程成本之间的权衡。

```
┌─────────────────────────────────────────────────────────────────┐
│                     DualTrack 测试平台                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  LR Track   │  │ LSTM Track  │  │ LightGBM    │   ML阵营     │
│  │  (拟合)     │  │  (序列)     │  │  (集成)     │   速度优先   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│              ┌─────────────────────┐                           │
│              │  Signal Converter   │  ← 独立转换，不融合！      │
│              └─────────────────────┘                           │
│                          │                                      │
│  ┌───────────────────────┼───────────────────────┐             │
│  │           ┌─────────────┐  ┌─────────────┐   │             │
│  │           │ LLM(Cloud)  │  │ LLM(Local)  │   │   LLM阵营   │
│  │           │ DeepSeek    │  │ Ollama      │   │   智能优先  │
│  │           └──────┬──────┘  └──────┬──────┘   │             │
│  │                  │                │           │             │
│  │                  └────────────────┘           │             │
│  │                           │                   │             │
│  │                           ▼                   │             │
│  │              ┌─────────────────────┐         │             │
│  │              │  Signal Converter   │         │             │
│  │              └─────────────────────┘         │             │
│  └───────────────────────────┼───────────────────┘             │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              Backtrader 回测引擎                     │      │
│  │  • 分别回测5个轨道                                   │      │
│  │  • 记录每笔交易、每日收益                            │      │
│  │  • 计算金融指标（Sharpe/MaxDD等）                    │      │
│  └─────────────────────────────────────────────────────┘      │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────┐      │
│  │              Comparator 对比分析器                   │      │
│  │  • LR vs LSTM vs LightGBM vs LLM(Cloud) vs LLM(Local)│     │
│  │  • 回答：谁更赚钱？谁更稳健？谁更划算？              │      │
│  └─────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### 五轨道对比矩阵

| 轨道 | 代表 | 核心问题 | 优势 | 劣势 |
|------|------|----------|------|------|
| **LR** | 线性基线 | 最简单的拟合能赚钱吗？ | 速度极快 (<2ms) | 线性假设限制 |
| **LSTM** | 序列建模 | 时序特性有用吗？ | 捕捉长期依赖 | 训练慢，易过拟合 |
| **LightGBM** | 集成学习 | 树模型适合量化吗？ | 特征重要性可解释 | 对时序不敏感 |
| **LLM(Cloud)** | 云端智能 | 高质量推理的价值？ | 语义理解、风险识别 | API成本高 |
| **LLM(Local)** | 本地智能 | 私有化部署可行吗？ | 零API成本 | 推理速度慢 |

### 核心假设验证

| 假设 | 对比维度 | 预期结论 |
|------|----------|----------|
| **H1** | ML vs LLM 收益能力 | LSTM (Sharpe最高) vs LLM-Cloud |
| **H2** | 黑天鹅事件风险控制 | LLM Tracks 最大回撤 < ML Tracks |
| **H3** | 成本效益比 | ML成本 ≈ $0，LLM有显著API成本 |

---

## 快速开始

### 安装依赖

```bash
# 使用 uv 安装依赖（强烈推荐）
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 或使用 pip
pip install -r requirements.txt
```

### 五轨道 CLI

```bash
# 查看帮助
python main.py --help

# 单轨道测试
python main.py run --track lr --symbol CSI300
python main.py run --track lstm --symbol CSI300
python main.py run --track lgb --symbol CSI300
python main.py run --track llm-cloud --symbol CSI300
python main.py run --track llm-local --symbol CSI300

# 全轨道对比测试（生成对比报告）
python main.py run --track all --compare --symbol CSI300

# 重新生成评估图表
python main.py evaluate

# 构建 LLM 离线缓存（支持断点续传）
python main.py cache-build --symbol CSI300
```

### 五轨道对比报告示例

```bash
$ python main.py run --track all --compare --symbol CSI300

======================================================================
  【五轨道对比分析】
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
📊 收益能力: LSTM > LightGBM > LR > LLM-Cloud > LLM-Local
🛡️ 风险控制: LLM-Cloud > LLM-Local > LSTM > LightGBM > LR
⚡ 执行效率: LR > LightGBM > LSTM > LLM-Cloud > LLM-Local
💰 成本效益: LR = LSTM = LightGBM > LLM-Local > LLM-Cloud
```

---

## 项目结构

```
src/
├── data/              # Phase 1: 数据获取与对齐
│   ├── market_data.py     # OHLCV 数据获取 (akshare, yfinance)
│   ├── news_data.py       # 新闻/情绪数据生成
│   └── data_aligner.py    # 时间对齐与缺失值处理
├── models/
│   ├── ml_track/      # Phase 2: 机器学习轨道
│   │   ├── features.py    # 特征工程 (50+ 技术指标)
│   │   └── baselines.py   # 基准模型 (LR, LSTM, LightGBM)
│   └── llm_track/     # Phase 3: 大语言模型轨道
│       ├── prompts.py     # Chain-of-Thought 提示模板
│       └── agent.py       # LLM 智能体 (Ollama, DeepSeek)
├── orchestrator/      # Phase 4: 双轨编排器
│   └── fusion_engine.py   # 信号转换器（独立轨道，不融合）
├── execution/         # Phase 5: 回测执行引擎
│   └── bt_engine.py       # Backtrader 封装
├── evaluation/        # Phase 6: 多维度评估
│   ├── metrics_calculator.py  # 金融与工程指标计算
│   └── visualizer.py     # 论文图表生成
└── utils/             # 新增：时间处理工具
    └── time_utils.py      # 交易日对齐、防未来函数
docs/
├── figures/           # 论文图表 (PNG, 300 DPI)
├── cache/llm_responses/ # LLM 离线缓存 (.jsonl)
└── implementation/    # 实施记录
    └── BACKTEST_FIX_IMPLEMENTATION.md  # 五轨道架构实施记录
```

---

## 关键设计原则

### 1. 无未来函数（绝对禁止）

```python
# 防止未来函数的关键约束
past_signals = signals[signals[date_col] <= trade_day]  # 信号时间 <= 交易日
```

- 特征工程使用 `shift()` 确保只用历史数据
- 信号对齐使用 `ffill` 前向填充
- LLM 信号按日期聚合后对齐到交易日

### 2. 独立轨道设计（非融合）

传统"融合"设计：
```
ML信号 ──┐
         ├──融合──> 统一仓位 ──> 回测
LLM信号 ──┘
```

DualTrack 独立轨道设计：
```
ML信号 ──> Signal Converter ──> ML仓位 ──> 独立回测 ──┐
                                                      ├──对比分析
LLM信号 ──> Signal Converter ──> LLM仓位 ──> 独立回测 ─┘
```

### 3. 对比维度矩阵

| 维度 | LR | LSTM | LightGBM | LLM-Cloud | LLM-Local |
|------|-----|------|----------|-----------|-----------|
| 收益能力 | ★★★ | ★★★★ | ★★★★ | ★★ | ★★ |
| 风险控制 | ★★ | ★★★ | ★★★ | ★★★★ | ★★★ |
| 执行速度 | ★★★★★ | ★★★ | ★★★★ | ★ | ★ |
| 运营成本 | ★★★★★ | ★★★★★ | ★★★★★ | ★★ | ★★★★ |

---

## Docker 部署（仅 Linux/CUDA）

Docker is recommended **only for Linux/CUDA environments**:

```bash
# 构建镜像
docker build -t dualtrack-quant .

# 运行回测
docker run --rm dualtrack-quant python main.py run --track all --compare --symbol CSI300

# 使用 docker-compose
docker-compose up -d
```

---

## 环境变量

| 变量名 | 用途 |
|-------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `OLLAMA_HOST` | Ollama 服务地址 (默认: http://localhost:11434) |

---

## 论文核心贡献

通过五轨道对比，本项目回答以下核心问题：

1. **拟合 vs 推理**: 传统ML和LLM哪种范式更适合量化交易？
2. **速度 vs 智能**: 低延迟的拟合 vs 高延迟的推理，哪个更优？
3. **云端 vs 本地**: LLM的智能是否值得额外的API成本？
4. **黑天鹅事件**: LLM的语义理解能否在极端行情中提供保护？

> "DualTrack 不是交易策略，而是量化交易技术对比的**工程基础设施（Testbed）**。它建立了一个公平的竞技场，让'拟合'与'推理'在同一个时间线上展开对决。"

---

## 项目状态

| 阶段 | 状态 | 描述 |
|------|------|------|
| Phase 1: 数据层 | ✅ 完成 | OHLCV获取、新闻数据、时间对齐 |
| Phase 2: ML轨道 | ✅ 完成 | LR、LSTM、LightGBM 基准模型 |
| Phase 3: LLM轨道 | ✅ 完成 | Ollama、DeepSeek 智能体 |
| Phase 4: 编排器 | ✅ 完成 | 五轨道独立信号转换 |
| Phase 5: 执行引擎 | ✅ 完成 | Backtrader 回测封装 |
| Phase 6: 评估 | ✅ 完成 | 多维度指标、可视化 |
| Phase 7: CLI & Docker | ✅ 完成 | 五轨道CLI、容器化 |

**当前版本**: v2.0 五轨道对比平台
**最后更新**: 2026-03-01
**维护者**: DualTrack Research Team

---

## License

MIT License
