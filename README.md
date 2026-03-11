# DualTrack Quant Research

**多轨道量化交易对比实验平台** —— 严格对比传统机器学习（拟合/Fitting）与大语言模型（推理/Reasoning）在量化交易中的 ROI、鲁棒性与工程可行性。

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

## 多轨道架构设计

DualTrack 建立统一的基准测试标准，在**完全相同的市场条件**下，量化九种截然不同的技术在财务表现与工程成本之间的权衡。

```
┌───────────────────────────────────────────────────────────────────────┐
│                     DualTrack 测试平台                                 │
├───────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                   │
│  │  LR Track   │  │ LSTM Track  │  │ LightGBM    │    ML阵营         │
│  │  (拟合)     │  │  (序列)     │  │  (集成)     │    速度优先       │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                   │
│         │                │                │                           │
│         └────────────────┼────────────────┘                           │
│                          ▼                                            │
│              ┌─────────────────────┐                                 │
│              │  Signal Converter   │  ← 独立转换，不融合！            │
│              └─────────────────────┘                                 │
│                          │                                            │
│  ┌───────────────────────┼───────────────────────────────────────────┐│
│  │           ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      ││
│  │           │DeepSeek-V3.2│  │DeepSeek R1  │  │   Qwen3.5   │      ││
│  │           │  (云端)     │  │  (本地)     │  │  (云端)     │      ││
│  │           └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      ││
│  │                  │                │                │              ││
│  │                  └────────────────┼────────────────┘              ││
│  │                                   │                               ││
│  │                                   ▼                               ││
│  │                     ┌─────────────────────┐                      ││
│  │                     │  Signal Converter   │                      ││
│  │                     └─────────────────────┘                      ││
│  └───────────────────────────────────┬───────────────────────────────┘│
│                                      ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Backtrader 回测引擎                               │  │
│  │  • 分别回测9个轨道                                             │  │
│  │  • 记录每笔交易、每日收益                                      │  │
│  │  • 计算金融指标（Sharpe/MaxDD等）                              │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                                      │                                │
│                                      ▼                                │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │              Comparator 对比分析器                             │  │
│  │  • 9轨道对比: LR vs LSTM vs LGB vs DeepSeek vs Qwen vs Ollama  │  │
│  │  • 回答：谁更赚钱？谁更稳健？谁更划算？                        │  │
│  └───────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────┘
```

### 轨道对比矩阵

| 轨道 | 代表 | 核心问题 | 优势 | 劣势 |
|------|------|----------|------|------|
| **LR** | 线性基线 | 最简单的拟合能赚钱吗？ | 速度极快 (<2ms) | 线性假设限制 |
| **LSTM** | 序列建模 | 时序特性有用吗？ | 捕捉长期依赖 | 训练慢，易过拟合 |
| **LightGBM** | 集成学习 | 树模型适合量化吗？ | 特征重要性可解释 | 对时序不敏感 |
| **DeepSeek-V3.2** | 云端推理 | 高质量推理的价值？ | 语义理解、风险识别 | API成本高 |
| **DeepSeek-V3.2-Reasoning** | 深度推理 | 推理模式效果如何？ | 深度思考链 | 延迟较高 |
| **DeepSeek-R1-14B** | 本地推理 | 本地大模型可行吗？ | 零API成本 | 推理速度慢 |
| **DeepSeek-R1-8B** | 轻量推理 | 小模型够用吗？ | 低资源需求 | 能力受限 |
| **Qwen3.5** | 通义千问 | 国产模型表现？ | 中文优化 | 依赖云端 |
| **Qwen3.5-9B** | 通义轻量 | 轻量通义够用吗？ | 平衡性能成本 | 能力有限 |

### 核心假设验证

| 假设 | 对比维度 | 预期结论 |
|------|----------|----------|
| **H1** | ML vs LLM 收益能力 | LSTM (Sharpe最高) vs LLM-Cloud |
| **H2** | 黑天鹅事件风险控制 | LLM Tracks 最大回撤 < ML Tracks |
| **H3** | 成本效益比 | ML成本 ≈ $0，LLM有显著API成本 |

---

## 快速开始

### 安装依赖

**本项目使用 `uv` 作为包管理器（禁止使用 pip）**

```bash
# 安装 uv（如果尚未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 安装项目依赖
uv sync

# 运行回测
uv run python main.py run --track all --compare --symbol CSI300
```

<details>
<summary>📋 uv 常用命令速查表</summary>

| 操作 | 命令 |
|------|------|
| 安装所有依赖 | `uv sync` |
| 添加新依赖 | `uv add <package>` |
| 添加开发依赖 | `uv add --dev <package>` |
| 运行脚本 | `uv run python <script.py>` |
| 更新所有依赖 | `uv lock --upgrade && uv sync` |
| 导出 requirements.txt | `uv pip freeze > requirements.txt` |

</details>

### 多轨道 CLI

```bash
# 查看帮助
uv run python main.py --help

# ML 轨道测试
uv run python main.py run --track lr --symbol CSI300
uv run python main.py run --track lstm --symbol CSI300
uv run python main.py run --track lgb --symbol CSI300

# LLM 轨道测试 (云端)
uv run python main.py run --track deepseek-v3.2 --symbol CSI300
uv run python main.py run --track deepseek-v3.2-reasoning --symbol CSI300
uv run python main.py run --track qwen3.5 --symbol QQQ

# LLM 轨道测试 (本地)
uv run python main.py run --track deepseek-r1-14b --symbol CSI300
uv run python main.py run --track deepseek-r1-8b --symbol CSI300

# 全轨道对比测试
uv run python main.py run --track all --compare --symbol CSI300

# 美股市场测试
uv run python main.py run --track all --compare --symbol QQQ

# 重新生成评估图表
uv run python main.py evaluate

# 生成高级学术评估报告
uv run python main.py evaluate-advanced \
  --output-dir docs/figures \
  --vix-file data/raw/vix_2015_2024.csv \
  --llm-cache-dir docs/cache/llm_responses

# 构建 LLM 离线缓存
uv run python main.py cache-build --symbol CSI300

# 训练 ML 模型（持久化到 models/）
uv run python scripts/train_ml_models.py --symbol CSI300
uv run python scripts/train_ml_models.py --symbol QQQ
uv run python scripts/train_ml_models.py --all  # 训练所有市场
```

### 多轨道对比报告示例

```bash
$ uv run python main.py run --track all --compare --symbol CSI300

======================================================================
  【多轨道对比分析】
======================================================================

【财务指标对比】
轨道                    夏普比率    最大回撤    总收益率    胜率
--------------------------------------------------------------------
LR                      1.05       18.5%      12.3%      54%
LSTM                    1.28       15.2%      15.7%      56%  ⭐ 最佳夏普
LightGBM                1.18       16.8%      14.1%      55%
DeepSeek-V3.2           0.95       11.5%      10.2%      53%  ⭐ 最佳风控
DeepSeek-V3.2-Reasoning 0.92       12.1%       9.8%      52%
DeepSeek-R1-14B         0.88       13.2%       9.1%      51%
DeepSeek-R1-8B          0.85       14.1%       8.5%      50%
Qwen3.5                 0.90       12.8%       9.5%      52%
Qwen3.5-9B              0.87       13.5%       8.9%      51%

【工程指标对比】
轨道                    平均延迟    总成本      信号数量
--------------------------------------------------------------------
LR                      2.1ms      $0.00      242
LSTM                    15.3ms     $0.00      242
LightGBM                3.2ms      $0.00      242
DeepSeek-V3.2           1250ms     $1.20      51
DeepSeek-V3.2-Reasoning 3500ms     $2.40      51
DeepSeek-R1-14B         4200ms     $0.00      51
DeepSeek-R1-8B          2800ms     $0.00      51
Qwen3.5                 1100ms     $0.80      51
Qwen3.5-9B              950ms      $0.60      51

【核心结论】
📊 收益能力: LSTM > LightGBM > LR > DeepSeek-V3.2 > Qwen3.5
🛡️ 风险控制: DeepSeek-V3.2 > Qwen3.5 > LSTM > LightGBM > LR
⚡ 执行效率: LR > LightGBM > LSTM > Qwen3.5 > DeepSeek-V3.2
💰 成本效益: ML轨道 > 本地LLM > 云端LLM
```

---

## 高级学术评估框架

Phase 8 新增四大评估维度，用于论文级别的深度分析：

### 评估模块

| 模块 | 功能 | 输出 |
|------|------|------|
| `trade_analyzer.py` | MAE/MFE 交易质量分析 | 入场效率、持仓效率、出场效率 |
| `market_state_analyzer.py` | 市场状态切割分析 | Normal/High-Vol/Black-Swan 状态绩效 |
| `ml_explainer.py` | ML 特征归因 | SHAP 重要性图 |
| `llm_explainer.py` | LLM Reasoning 分析 | 词云、主题分布、质量评分 |
| `cross_market_analyzer.py` | 跨市场对比 | A股 vs 美股雷达图 |
| `attribution_comparator.py` | 归因对比 | ML vs LLM 归因差异 |
| `advanced_visualizer.py` | 论文级可视化 | 300 DPI PNG 图表 |

### 高级评估命令

```bash
# 生成完整高级评估报告
uv run python main.py evaluate-advanced \
  --output-dir docs/figures \
  --vix-file data/raw/vix_2015_2024.csv \
  --llm-cache-dir docs/cache/llm_responses

# 预期输出图表
docs/figures/
├── trade_quality_comparison.png   # 交易质量对比
├── market_state_heatmap.png       # 市场状态热力图
├── ml_shap_importance.png         # ML SHAP 归因
├── llm_reasoning_wordcloud.png    # LLM Reasoning 词云
└── cross_market_radar.png         # 跨市场雷达图
```

### 学术话术提炼

基于四大评估维度，提炼论文核心结论：

1. **交易质量**: ML Track 入场效率高，LLM Track 持仓效率优
2. **市场状态**: LLM 在 Black Swan 事件中显著降低 MaxDD
3. **可解释性**: ML 提供特征归因，LLM 提供 Reasoning 主题
4. **跨市场**: A股市场 LLM 优势更明显，美股 ML 略占优

---

## 项目结构

```
src/
├── data/              # 数据获取与对齐
│   ├── market_data.py     # OHLCV 数据获取 (akshare, yfinance)
│   ├── news_data.py       # 新闻/情绪数据生成
│   ├── data_aligner.py    # 时间对齐与缺失值处理
│   └── fetch_real_news.py # 真实新闻获取
├── models/
│   ├── ml_track/      # 机器学习轨道
│   │   ├── features.py    # 特征工程 (50+ 技术指标)
│   │   └── baselines.py   # 基准模型 (LR, LSTM, LightGBM)
│   ├── llm_track/     # 大语言模型轨道
│   │   ├── prompts.py     # Chain-of-Thought 提示模板 (A股)
│   │   ├── us_prompts.py  # 美股提示模板
│   │   └── agent.py       # LLM 智能体 (DeepSeek, Qwen, Ollama)
│   └── model_manager.py   # 模型管理器
├── orchestrator/      # 编排器
│   ├── fusion_engine.py   # 信号融合引擎
│   ├── signal_converter.py # 信号转换器
│   └── comparator.py      # 对比分析器
├── execution/         # 回测执行引擎
│   ├── bt_engine.py       # Backtrader 封装
│   ├── base_strategy.py   # 基础策略
│   ├── a_share_strategy.py # A股策略
│   └── us_market_strategy.py # 美股策略
├── evaluation/        # 多维度评估
│   ├── metrics_calculator.py  # 金融与工程指标计算
│   ├── visualizer.py     # 论文图表生成
│   ├── report_generator.py # 报告生成
│   ├── trade_analyzer.py     # MAE/MFE 交易质量分析
│   ├── market_state_analyzer.py # 市场状态切割
│   ├── ml_explainer.py       # SHAP 特征归因
│   ├── llm_explainer.py      # Reasoning 主题分析
│   ├── cross_market_analyzer.py # 跨市场对比
│   ├── attribution_comparator.py # 归因对比
│   └── advanced_visualizer.py # 高级可视化
├── utils/             # 工具函数
│   ├── logger.py          # 日志工具
│   ├── config_loader.py   # 配置加载
│   └── time_utils.py      # 交易日对齐、防未来函数
└── config/            # 配置模块
    └── market_config.py   # 市场配置
config/                # 配置文件
├── llm_config.yaml        # LLM 配置
├── data_config.yaml       # 数据配置
├── ml_config.yaml         # ML 配置
└── backtest_config.yaml   # 回测配置
scripts/               # 数据处理脚本
├── fetch_financial_news.py   # 金融新闻获取
├── fetch_csi300_constituents.py # 沪深300成分股
├── prepare_us_news.py    # 美股新闻准备
└── train_ml_models.py    # ML 模型训练脚本
docs/
├── figures/           # 论文图表 (PNG, 300 DPI)
├── cache/llm_responses/ # LLM 离线缓存 (.jsonl)
└── implementation/    # 实施记录
    └── BACKTEST_FIX_IMPLEMENTATION.md  # 架构实施记录
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

| 维度 | LR | LSTM | LightGBM | DeepSeek | Qwen | 本地LLM |
|------|-----|------|----------|----------|------|---------|
| 收益能力 | ★★★ | ★★★★ | ★★★★ | ★★★ | ★★★ | ★★ |
| 风险控制 | ★★ | ★★★ | ★★★ | ★★★★ | ★★★★ | ★★★ |
| 执行速度 | ★★★★★ | ★★★ | ★★★★ | ★★ | ★★ | ★ |
| 运营成本 | ★★★★★ | ★★★★★ | ★★★★★ | ★★ | ★★ | ★★★★ |

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
| `OPENAI_API_KEY` | OpenAI API 密钥 (用于 Qwen 等) |
| `SILICONFLOW_API_KEY` | SiliconFlow API 密钥 |
| `OLLAMA_HOST` | Ollama 服务地址 (默认: http://localhost:11434) |

---

## 论文核心贡献

通过多轨道对比，本项目回答以下核心问题：

1. **拟合 vs 推理**: 传统ML和LLM哪种范式更适合量化交易？
2. **速度 vs 智能**: 低延迟的拟合 vs 高延迟的推理，哪个更优？
3. **云端 vs 本地**: LLM的智能是否值得额外的API成本？
4. **模型规模**: 不同参数量的LLM在量化任务中的表现差异？
5. **黑天鹅事件**: LLM的语义理解能否在极端行情中提供保护？

> "DualTrack 不是交易策略，而是量化交易技术对比的**工程基础设施（Testbed）**。它建立了一个公平的竞技场，让'拟合'与'推理'在同一个时间线上展开对决。"

---

## 项目状态

| 阶段 | 状态 | 描述 |
|------|------|------|
| Phase 1: 数据层 | ✅ 完成 | OHLCV获取、新闻数据、时间对齐 |
| Phase 2: ML轨道 | ✅ 完成 | LR、LSTM、LightGBM 基准模型 |
| Phase 3: LLM轨道 | ✅ 完成 | DeepSeek、Qwen、Ollama 多模型支持 |
| Phase 4: 编排器 | ✅ 完成 | 多轨道独立信号转换 |
| Phase 5: 执行引擎 | ✅ 完成 | Backtrader 回测封装 |
| Phase 6: 评估 | ✅ 完成 | 多维度指标、可视化 |
| Phase 7: CLI & Docker | ✅ 完成 | 多轨道CLI、容器化 |
| Phase 8: 高级评估 | ✅ 完成 | 四大评估维度、论文级可视化 |

**当前版本**: v2.2 高级学术评估平台
**最后更新**: 2026-03-11
**维护者**: DualTrack Research Team

---

## License

MIT License
