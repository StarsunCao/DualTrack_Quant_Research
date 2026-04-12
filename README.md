# DualTrack Quant Research

**多轨道量化交易对比实验平台** —— 严格对比传统机器学习（Fitting）与大语言模型（Semantic Reasoning）在量化交易中的 ROI、鲁棒性与工程可行性，特别是在黑天鹅事件下的表现。

> **核心定位**: 本项目不是融合策略，而是对比实验的**工程基础设施（Testbed）**，建立公平竞技场让"拟合"与"推理"在同一市场条件下展开对决。

---

## 项目概述

DualTrack 是一个基于 Python 3.12+ 的双轨制量化回测框架，支持：

- **3 种机器学习模型**: Logistic Regression, LSTM, LightGBM
- **7+ 种大语言模型**: DeepSeek V3.2, DeepSeek R1 系列, Qwen3.5, GLM-5 等
- **2 个市场**: A股 (CSI300) 和美股 (QQQ/NASDAQ-100)
- **独立轨道回测**: 每个模型独立生成信号、独立回测，不做信号融合
- **多维度评估**: 金融指标 (Sharpe, MaxDD, WinRate) + 工程指标 (Latency, Cost)

---

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    DualTrack 测试平台                        │
├─────────────────────────────────────────────────────────────┤
│  ML 轨道                              LLM 轨道              │
│  ┌──────┐ ┌──────┐ ┌──────┐    ┌─────────┐ ┌─────────┐    │
│  │ LR   │ │LSTM  │ │ LGB  │    │DeepSeek │ │  Qwen   │    │
│  │(线性)│ │(序列)│ │(集成)│    │  R1/GLM │ │  系列   │    │
│  └──┬───┘ └──┬───┘ └──┬───┘    └────┬────┘ └────┬────┘    │
│     └────────┼────────┘             └─────┬──────┘          │
│              ▼                            ▼                 │
│  ┌──────────────────────┐  ┌──────────────────────────┐    │
│  │  Signal Converter    │  │  Signal Converter        │    │
│  │  (ML → 目标仓位)     │  │  (LLM → 目标仓位)        │    │
│  └──────────┬───────────┘  └────────────┬─────────────┘    │
│             └─────────────┬─────────────┘                  │
│                           ▼                                │
│              ┌─────────────────────────┐                   │
│              │   Backtrader 回测引擎    │                   │
│              │  独立回测 + 绩效计算     │                   │
│              └─────────────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
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

| 模型 | 部署方式 | 说明 |
|------|---------|------|
| `deepseek-v3.2` | 云端 (SiliconFlow) | DeepSeek V3.2 标准版 |
| `deepseek-v3.2-reasoning` | 云端 (SiliconFlow) | DeepSeek V3.2 推理模式 |
| `deepseek-r1-14b` | 云端 (SiliconFlow) | DeepSeek R1 14B 蒸馏版 |
| `deepseek-r1-8b` | 云端 (SiliconFlow) | DeepSeek R1 8B 轻量版 |
| `qwen3.5` | 云端 (SiliconFlow) | Qwen3.5 397B 满血版 |
| `qwen3.5-9b` | 云端 (SiliconFlow) | Qwen3.5 9B 可部署版 |
| `glm-5` | 云端 (SiliconFlow / DashScope) | GLM-5，支持阿里云 DashScope |
| `ollama` | 本地 | 本地部署，默认 qwen2.5:7b |
| `mock` | 模拟 | 离线测试，无需 API |

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
uv run python main.py run --track glm-5 --symbol QQQ

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
  --model deepseek-ai/DeepSeek-V3.2
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
# SiliconFlow API（DeepSeek / Qwen / GLM-5 云端调用）
export SILICONFLOW_API_KEY="your-key-here"

# 阿里云 DashScope（GLM-5 备选端点）
export DASHSCOPE_API_KEY="your-key-here"

# 本地 Ollama 服务
export OLLAMA_HOST="http://localhost:11434"
```

---

## 项目结构

```
├── main.py                    # CLI 入口
├── pyproject.toml             # 依赖管理 (uv)
├── src/
│   ├── data/                  # 数据获取与对齐
│   │   ├── market_data.py     # OHLCV (akshare, yfinance)
│   │   ├── news_data.py       # 新闻/情绪数据
│   │   └── data_aligner.py    # 时间对齐
│   ├── models/
│   │   ├── ml_track/          # ML 轨道
│   │   │   ├── features.py    # 50+ 技术指标
│   │   │   └── baselines.py   # LR, LSTM, LightGBM
│   │   ├── llm_track/         # LLM 轨道
│   │   │   ├── prompts.py     # A股 CoT 提示词
│   │   │   ├── us_prompts.py  # 美股提示词
│   │   │   └── agent.py       # 多模型执行器
│   │   └── model_manager.py
│   ├── orchestrator/          # 信号编排
│   │   ├── fusion_engine.py   # 信号融合引擎
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
│       └── cross_market_analyzer.py
├── config/                    # YAML 配置
│   ├── llm_config.yaml
│   ├── ml_config.yaml
│   ├── data_config.yaml
│   └── backtest_config.yaml
├── scripts/                   # 数据处理与训练脚本
│   ├── train_ml_models.py
│   ├── fetch_financial_news.py
│   └── prepare_us_news.py
├── models/                    # 预训练模型
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

包含数据获取、ML 轨道、LLM 轨道、信号融合、回测引擎、评估模块等完整测试覆盖。

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

---

## License

MIT License
