[中文](README_zh.md) | [English](README.md) | [Русский](README_ru.md)

---

# DualTrack Quant Research

**多轨道量化交易对比实验平台** —— 严格对比传统机器学习（Fitting）与大语言模型（Semantic Reasoning）在量化交易中的 ROI、鲁棒性与工程可行性，特别是在黑天鹅事件下的表现。

> **核心定位**: 本项目不是融合策略，而是对比实验的**工程基础设施（Testbed）**，建立公平竞技场让"拟合"与"推理"在同一市场条件下展开对决。

---

## 项目概述

DualTrack 是一个基于 Python 3.12+ 的双轨制量化回测框架，支持：

- **3 种机器学习模型**: Logistic Regression, LSTM, LightGBM
- **6 种大语言模型**: DeepSeek-V3.2, V4-Flash, R1-8B, Qwen 3.5-397B, GLM-5.1, Gemma-4-31B，支持云端与本地部署
- **SmartPromptAgent 状态增强智能体**: 技术指标、价格走势、闭环决策记忆，单次 LLM 调用实现 Agent 范式
- **2 个市场**: A股 (CSI300) 和美股 (QQQ/NASDAQ-100)
- **独立轨道回测**: 每个模型独立生成信号、独立回测，不做信号融合
- **多维度评估**: 金融指标 (Sharpe, MaxDD, WinRate) + 工程指标 (Latency, Cost)

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    DualTrack 测试平台                        │
├─────────────────────────────────────────────────────────────┤
│  ML 轨道                              LLM 轨道               │
│  ┌──────┐ ┌──────┐ ┌──────┐    ┌─────────┐ ┌─────────┐     │
│  │ LR   │ │LSTM  │ │ LGB  │    │DeepSeek │ │  Qwen   │     │
│  │(线性) │ │(序列)│ │(集成) │    │  R1/GLM │ │  系列    │     │
│  └──┬───┘ └──┬───┘ └──┬───┘    └────┬────┘ └────┬────┘     │
│     └────────┼────────┘             └─────┬──────┘         │
│              ▼                            ▼                │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │  Signal Converter    │  │  Signal Converter        │    │
│  │  (ML → 目标仓位)      │  │  (LLM → 目标仓位)          │    │
│  └──────────┬───────────┘  └────────────┬─────────────┘    │
│             └─────────────┬─────────────┘                  │
│                           ▼                                │
│              ┌─────────────────────────┐                   │
│              │   Backtrader 回测引擎    │                   │
│              │  独立回测 + 绩效计算       │                   │
│              └─────────────────────────┘                   │
└────────────────────────────────────────────────────────────┘
```

---

## 支持的模型

### ML 轨道 (机器学习)

| 模型 | 类型 | 特点 |
|------|------|------|
| `lr` | Logistic Regression | 线性基线，速度极快 |
| `lstm` | LSTM (PyTorch, MPS加速) | 时序建模，Apple Silicon 优化 |
| `lgb` | LightGBM | 集成学习，特征重要性可解释 |

### LLM 轨道 (大语言模型)

| 模型 | 部署方式 | 实验市场 | 说明 |
|------|---------|---------|------|
| `deepseek-v3.2` | 云端 (SiliconFlow) | A股 + 美股 | DeepSeek V3.2 标准版 |
| `deepseek-v4-flash` | 云端 (SiliconFlow) | A股 + 美股 | DeepSeek V4 Flash 轻量版 |
| `deepseek-r1-8b` | 本地 (Ollama) | A股 + 美股 | DeepSeek R1 8B 蒸馏推理模型 |
| `qwen3.5-397b` | 云端 (SiliconFlow) | A股 + 美股 | Qwen 3.5 397B 满血版 |
| `glm-5.1` | 云端 (SiliconFlow / DashScope) | A股 + 美股 | GLM-5.1 |
| `gemma-4-31b` | 云端 (SiliconFlow) | A股 + 美股 | Gemma 4 31B 开源模型 |

### SmartPromptAgent（状态增强型 LLM 智能体）

SmartPromptAgent 将 LLM 轨道从"Prompt 脚本"升级为现代 Agent 范式，同时保持每个交易日仅 1 次 LLM 调用以控制成本：

- **技术指标**：RSI、MACD、布林带、均线排列、量价关系 —— Python 预计算后以自然语言摘要注入
- **价格走势**：近 5 日价格走势表格，附带成交量标注
- **闭环记忆**：过去的决策及真实市场收益自动回填，让 LLM 能看到自己过去的判断是对是错
- **零未来函数**：所有注入数据严格限制在 T-1 及之前，由 `MarketEnricher` 的 assert 守卫强制保证
- **成本优化**：每天仅多约 300 token，仍为 1 次 LLM 调用/交易日（对比多调用 Agent 方案的 2-4 倍）
- **Temperature**：0.1（低随机性，确保回测可复现）

```bash
# 运行 SmartPromptAgent 回测
uv run python main.py run_llm_agent_backtest \
  --symbol CSI300 \
  --start 2020-01-02 \
  --end 2024-12-31 \
  --executor siliconflow \
  --model deepseek-ai/DeepSeek-V3.2
```

> **本地部署设备**: MacBook Air M2 (24GB 内存 + 512GB 存储)
>
> **Qwen 模型说明**: 因可能触发政治敏感词，Qwen 系列未用于 A股 市场实验，仅在美股市场进行测试。

---

## 论文

本项目基于实验数据撰写了完整的硕士学位论文，包含 4 个章节：

| 章节 | 内容 | 文件 |
|------|------|------|
| 第一章 | 文献综述：量化金融中的 ML 与 LLM | `thesis/sections/1.*` |
| 第二章 | 方法论：双轨架构设计 | `thesis/sections/2.*` |
| 第三章 | 实验与结果：跨市场实证对比 | `thesis/sections/3.*` |
| 第四章 | 结论与展望 | `thesis/sections/4.*` |

**架构图**（5 张，位于 `thesis/figures/`）：

| 图片 | 内容 |
|------|------|
| 图 1 | 双轨系统架构 |
| 图 2 | 数据处理与 Walk-Forward 管线 |
| 图 3 | LSTM 细胞门控机制 |
| 图 5 | LLM 提示结构与思维链推理 |
| 图 6 | SmartPromptAgent 架构 |

---

## 快速开始

### 安装

**本项目使用 `uv` 作为包管理器（禁止使用 pip）**

```bash
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库并安装依赖
git clone https://github.com/StarsunCao/DualTrack_Quant_Research.git
cd DualTrack_Quant_Research
uv sync
```

### 运行回测

```bash
# 单模型回测
uv run python main.py run --track lr --symbol CSI300
uv run python main.py run --track lstm --symbol CSI300
uv run python main.py run --track lgb --symbol QQQ

# LLM 模型回测（需要有效缓存或 API）
uv run python main.py run --track deepseek-v3.2 --symbol CSI300
uv run python main.py run --track glm-5.1 --symbol QQQ

# 全模型对比（需要所有模型的 LLM 缓存已就绪）
uv run python main.py run --track all --compare --symbol CSI300
```

### 构建 LLM 缓存

LLM API 调用成本高，项目支持离线缓存机制（断点续传）：

```bash
uv run python main.py cache-build \
  --symbol CSI300 \
  --start 2020-01-02 \
  --end 2024-12-31 \
  --news-file data/raw/csi300_news_combined_2020_2024.csv \
  --executor siliconflow \
  --model deepseek-ai/DeepSeek-V3.2 \
  --temperature 0.1
```

### 训练 ML 模型

```bash
uv run python scripts/train_ml_models.py --symbol CSI300
uv run python scripts/train_ml_models.py --symbol QQQ
uv run python scripts/train_ml_models.py --all  # 训练所有市场
```

### 评估与可视化

```bash
# 从已有回测结果生成评估图表
uv run python main.py evaluate

# 高级学术评估（MAE/MFE、市场状态切割、SHAP 归因等）
uv run python main.py evaluate-advanced \
  --output-dir docs/figures \
  --vix-file data/raw/vix_2015_2024.csv
```

### 环境变量

```bash
# SiliconFlow API (DeepSeek / Qwen / GLM-5.1 / Gemma 云端)
export SILICONFLOW_API_KEY="your-key-here"

# 阿里云 DashScope（GLM-5.1 备选端点）
export DASHSCOPE_API_KEY="your-key-here"

# 本地 Ollama 服务
export OLLAMA_HOST="http://localhost:11434"
```

---

## 项目结构

```
├── main.py                    # CLI 入口
├── pyproject.toml             # 依赖管理 (uv)
├── thesis/                    # 硕士学位论文（4 章）
│   ├── sections/              # 章节 markdown 文件
│   ├── figures/               # 架构图（5 张）
│   ├── itmo-vkr-template/     # LaTeX 模板
│   └── thesis_outline.md      # 论文大纲
├── src/
│   ├── data/                  # 数据获取与对齐
│   │   ├── market_data.py     # OHLCV (akshare, yfinance)
│   │   ├── news_data.py       # 新闻/情绪数据
│   │   └── data_aligner.py    # 时间对齐
│   ├── models/
│   │   ├── ml_track/          # ML 轨道
│   │   │   ├── features.py    # 55→17 核心特征（方差 + 相关性筛选）
│   │   │   └── baselines.py   # LR, LSTM, LightGBM
│   │   ├── llm_track/         # LLM 轨道
│   │   │   ├── prompts.py     # A股 CoT 提示词
│   │   │   ├── us_prompts.py  # 美股提示词
│   │   │   ├── agent.py       # 多模型执行器 + SmartPromptAgent
│   │   │   ├── memory.py      # 闭环反馈决策记忆
│   │   │   └── enricher.py    # 市场数据增强器（指标、价格、记忆）
│   │   └── model_manager.py
│   ├── orchestrator/          # 信号编排
│   │   ├── signal_converter.py
│   │   └── comparator.py
│   ├── execution/             # 回测执行
│   │   ├── bt_engine.py       # Backtrader 封装
│   │   ├── base_strategy.py
│   │   ├── a_share_strategy.py
│   │   └── us_market_strategy.py
│   └── evaluation/            # 多维度评估
│       ├── metrics_calculator.py
│       ├── visualizer.py
│       ├── trade_analyzer.py
│       ├── market_state_analyzer.py
│       ├── ml_explainer.py
│       ├── llm_explainer.py
│       └── attribution_comparator.py
├── config/                    # YAML 配置
│   ├── llm_config.yaml
│   ├── ml_config.yaml
│   ├── data_config.yaml
│   └── backtest_config.yaml
├── scripts/                   # 数据处理、训练与分析脚本
│   ├── train_ml_models.py
│   ├── fetch_financial_news.py
│   ├── prepare_us_news.py
│   └── chapter3_analysis.py   # 第三章实证分析脚本
├── models/                    # 预训练模型（Walk-Forward 版本）
├── tests/                     # 单元测试
└── data/                      # 数据文件 (.gitignore)
```

---

## 关键设计

### 1. 无未来函数（No Look-ahead Bias）

特征工程使用 `shift()` 确保只使用历史数据，信号对齐使用 `ffill` 前向填充。

### 2. 独立轨道（No Fusion）

每个模型独立生成信号、独立转换仓位、独立回测。不做信号融合，确保对比公平。

### 3. LLM 离线缓存

LLM API 调用昂贵且不稳定，项目支持 `.jsonl` 格式的离线缓存，支持断点续传。

### 4. Apple Silicon 优化

PyTorch LSTM 模型自动检测并使用 MPS (Metal Performance Shaders) 加速。

### 5. Walk-Forward 训练

ML 模型采用 912 天训练窗口、182 天重训频率的 Walk-Forward 验证机制。特征筛选通过方差阈值 + 相关性去冗余将 55 个原始特征降至 17 个核心特征。

---

## 评估维度

| 维度 | 说明 |
|------|------|
| **收益能力** | Sharpe Ratio, Sortino Ratio, Calmar Ratio, 总收益率 |
| **风险控制** | Maximum Drawdown, 回撤持续期, 黑天鹅期间表现 |
| **交易质量** | MAE/MFE 分析, 入场/持仓/出场效率 |
| **工程指标** | 推理延迟, 吞吐量, 每个 Alpha 信号成本 |
| **市场状态** | Normal / High-Vol / Black-Swan 状态下的分场景表现 |
| **可解释性** | ML: SHAP 特征归因; LLM: Reasoning 主题分析 |
| **统计鲁棒性** | Jobson-Korkie、Bootstrap、KS 检验用于绩效显著性验证 |

---

## 核心实证发现

| 发现 | 详情 |
|------|------|
| **ML 在 A 股占优** | LR: +9.34% vs 买入持有 -5.23%，夏普 0.172，换手率 0.0019 |
| **LLM 在美股有竞争力** | Qwen 3.5-397B: +22.80%，QQQ 市场 Beta Drift (+67.16%) |
| **LLM 置信度 ≠ 准确率** | 所有 6 个模型相关性接近零（-0.055 至 0.064） |
| **保守主义偏差** | LLM 很少表达高置信度（≥0.80）；V4-Flash 错过 20.54% 上涨日 |
| **过度解读** | 所有 6 个 LLM 在 2020-03-13（V 型反弹日，+8.47%）集体做空 |
| **无统计显著差异** | ML-Ensemble vs LLM-Qwen: Jobson-Korkie p=0.8412, Bootstrap CI [-1.87, 2.06] |

---

## 核心研究问题

1. **拟合 vs 推理**: 传统 ML 和 LLM 哪种范式更适合量化交易？
2. **速度 vs 智能**: 低延迟的拟合 vs 高延迟的推理，哪个更优？
3. **云端 vs 本地**: LLM 的语义理解是否值得额外的 API 成本？
4. **模型规模**: 不同参数量的 LLM 在量化任务中的表现差异？
5. **黑天鹅事件**: LLM 的语义理解能否在极端行情中提供尾部保护？
6. **跨市场泛化**: 同一套系统能否同时适用于 A股 和美股？

---

## 测试

```bash
uv run pytest tests/ -v
```

包含数据获取、ML 轨道、LLM 轨道、回测引擎、评估模块等完整测试覆盖。

---

## Docker 部署

Docker 镜像仅用于 Linux/CUDA 云端部署，Apple Silicon Mac 用户请直接使用 `uv`：

```bash
# Apple Silicon Mac 推荐方式
uv sync
uv run python main.py run --track lr --symbol CSI300

# Docker (Linux/CUDA)
docker build -t dualtrack-quant .
docker run --rm dualtrack-quant python main.py run --track all --compare --symbol CSI300
```

---

## 项目状态

| 阶段 | 状态 | 描述 |
|------|------|------|
| Phase 1: 数据层 | ✅ 完成 | OHLCV 获取、新闻数据、时间对齐 |
| Phase 2: ML 轨道 | ✅ 完成 | LR, LSTM, LightGBM |
| Phase 3: LLM 轨道 | ✅ 完成 | DeepSeek, Qwen, GLM-5, Ollama |
| Phase 4: 编排器 | ✅ 完成 | 独立信号转换 |
| Phase 5: 执行引擎 | ✅ 完成 | Backtrader 回测封装 |
| Phase 6: 评估 | ✅ 完成 | 多维度指标、可视化 |
| Phase 7: CLI & Docker | ✅ 完成 | 多轨道 CLI、容器化 |
| Phase 8: 高级评估 | ✅ 完成 | MAE/MFE、市场状态切割、SHAP 归因 |
| Phase 9: Agent 架构 | ✅ 完成 | SmartPromptAgent: 指标、价格、闭环记忆、零未来函数 |
| Phase 10: 统计检验 | ✅ 完成 | Jobson-Korkie、Bootstrap、KS、卡方检验 |
| Phase 11: 论文撰写 | ✅ 完成 | 4 章正文、5 张架构图、完整实证分析 |

---

## License

MIT License
