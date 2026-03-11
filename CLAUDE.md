# Dual-Track Quant Research: Project Guidelines

## 1. Project Overview
本项目是一个双轨制（Dual-Track）量化交易回测框架，核心目标是严格对比传统机器学习（Fitting）与大语言模型智能体（Semantic Reasoning）在量化交易中的 ROI 和鲁棒性，特别是在黑天鹅事件下的表现。

## 2. Package Manager (重要)

**本项目严格使用 `uv` 作为包管理器，禁止使用 pip！**

| 操作 | 正确命令 | 错误命令 |
|------|----------|----------|
| 安装依赖 | `uv sync` | ❌ `pip install -r requirements.txt` |
| 添加新依赖 | `uv add <package>` | ❌ `pip install <package>` |
| 运行脚本 | `uv run python <script.py>` | ❌ `python <script.py>` |
| 更新依赖 | `uv lock --upgrade` | ❌ 手动编辑 requirements.txt |

**原因**: uv 比 pip 快 10-100 倍，且自动管理虚拟环境。

## 3. Technology Stack
- **Core Frameworks**: Backtrader (Execution), scikit-learn/LightGBM (ML Track), DeepSeek/Qwen/Ollama API (LLM Track)。
- **Multi-Market Support**: A 股 (CSI300) 和美股 (QQQ/NASDAQ-100)。
- **Hardware Optimization**: 针对 Apple Silicon (M系列芯片) 优化，PyTorch 模型必须优先挂载至 `mps` 设备。

## 4. Architecture Strict Rules
- **No Look-ahead Bias (绝对禁止未来函数)**: 在特征工程（计算 $t$ 时刻因子）和模型预测时，绝对禁止使用 $t+1$ 及之后的数据。数据对齐必须严格使用 `ffill`（向前填充）。
- **Data Structures**:
  - ML Track 信号输出: `[timestamp, symbol, model_name, signal_strength_0_to_1]`
  - LLM Track 信号输出: `[timestamp, symbol, llm_signal, reasoning, latency_ms]` 且必须离线缓存为 `.jsonl`。
  - Orchestrator 目标仓位输出: 必须是严格的 `Dict[str, float]` 格式，如 `{'CSI300': 0.8}`。
- **Defensive Programming**:
  - LLM 输出极不稳定，必须包含严格的 JSON 解析容错机制（截断置信度、设置默认 hold 信号）。
  - Backtrader 兼容性：由于使用 Python 3.12+，必须在回测引擎入口处理 `collections.Iterable` 的 Monkey Patch。

---

## ⚠️ Docker for Mac / Apple Silicon 警告

**由于 Docker for Mac 无法穿透调用 Apple Silicon MPS (Metal Performance Shaders)，对于 M 系列芯片 Mac 用户，建议直接使用本地 `uv` 虚拟环境运行回测以确保 LSTM 训练速度。Docker 镜像仅用于 Linux/CUDA 云端部署。**

```bash
# Apple Silicon Mac 推荐使用方式
uv sync
uv run python main.py run --symbol CSI300

# Docker 仅用于 Linux/CUDA 环境
docker build -t dualtrack-quant .
```

---

## 5. Completed Modules (Phase 1-7)

### Phase 1: Data Layer
**路径**: `src/data/`

| 文件 | 功能 | 核心特性 |
|-----|------|---------|
| `market_data.py` | OHLCV 价格获取 | akshare (CSI300), yfinance (NASDAQ-100/QQQ) |
| `news_data.py` | 新闻/情绪数据生成 | Mock 新闻模板，情感标签 |
| `data_aligner.py` | 时间对齐与缺失值处理 | `ffill` 前向填充，跨市场时间对齐 |

**验证**: `tests/test_data_module.py`

---

### Phase 2: ML Track
**路径**: `src/models/ml_track/`

| 文件 | 功能 | 核心特性 |
|-----|------|---------|
| `features.py` | 技术因子计算 | 50+ 指标 (RSI, MACD, Bollinger, ATR 等)，严格 shift() 防止未来函数 |
| `baselines.py` | 基准模型集成 | LogisticRegression, LightGBM, LSTM (MPS 加速) |

**验证**: `tests/test_ml_track.py` - 未来函数审计、MPS 设备验证

---

### Phase 3: LLM Track
**路径**: `src/models/llm_track/`

| 文件 | 功能 | 核心特性 |
|-----|------|---------|
| `prompts.py` | A股 Chain-of-Thought 提示模板 | 5 步推理框架 |
| `us_prompts.py` | 美股提示模板 | 针对美股市场优化 |
| `agent.py` | LLM 智能体 | DeepSeekExecutor, QwenExecutor, OllamaExecutor, SiliconFlowExecutor；JSON 解析容错；`.jsonl` 离线缓存 |

**支持模型**:
- DeepSeek V3.2 (云端)
- DeepSeek V3.2 Reasoning (云端，推理模式)
- DeepSeek R1 14B (本地)
- DeepSeek R1 8B (本地)
- Qwen3.5 397B (云端)
- Qwen3.5-9B (云端)

**验证**: `tests/test_llm_track.py` - JSON 解析容错、执行器连通性

---

### Phase 4: Orchestrator
**路径**: `src/orchestrator/fusion_engine.py`

**功能**: 双轨信号融合引擎

**融合规则**:

| 市场状态 | 波动率阈值 | ML 权重 | LLM 权重 | 特殊行为 |
|---------|-----------|--------|---------|---------|
| Normal | < 2% | 70% | 30% | 常规融合 |
| High Volatility | 2% ~ 5% | 50% | 50% | 增强 LLM 风险提示 |
| Black Swan | > 5% | 0% | 100% | LLM 否决权，强制清仓 |

**优化特性 (Phase 7 新增)**:
- **调仓死区 (Rebalancing Dead Zone)**: 阈值 5%，避免微调仓的手续费磨损
- **LLM 信号衰减**: 超过 3 个交易日无新新闻，信号权重逐渐衰减

**验证**: `tests/test_orchestrator.py` - 否决机制验证 (ML +0.8 → LLM 否决 → -1.0)

---

### Phase 5: Execution
**路径**: `src/execution/bt_engine.py`

**功能**: Backtrader 回测引擎封装

| 组件 | 功能 |
|-----|------|
| `PandasDataFeed` | OHLCV 数据加载 |
| `DualTrackStrategy` | 目标仓位执行策略 |
| `BacktestEngine` | Cerebro 封装，分析器配置 |
| `BacktestResult` | 结果数据类 |

**兼容性修复**: Python 3.12+ `collections.Iterable` Monkey Patch

**验证**: `tests/test_bt_engine.py` - 订单成交验证、分析器提取

---

### Phase 6: Evaluation
**路径**: `src/evaluation/`

| 文件 | 功能 | 核心特性 |
|-----|------|---------|
| `metrics_calculator.py` | 多维度指标计算 | 金融指标 (Sharpe, MaxDD, WinRate)；工程指标 (Latency, Cost-per-Alpha) |
| `visualizer.py` | 论文图表生成 | 资金曲线对比图、回撤热力图、延迟箱线图 |
| `trade_analyzer.py` | 交易质量分析 | MAE/MFE 分析、入场/持仓/出场效率 |
| `market_state_analyzer.py` | 市场状态分析 | Normal/High-Vol/Black-Swan 切割 |
| `ml_explainer.py` | ML 特征归因 | SHAP 重要性分析 |
| `llm_explainer.py` | LLM Reasoning 分析 | 词云生成、主题提取、质量评分 |
| `cross_market_analyzer.py` | 跨市场对比 | A股 vs 美股雷达图 |
| `attribution_comparator.py` | 归因对比器 | ML vs LLM 统一归因 |
| `advanced_visualizer.py` | 高级可视化 | 论文级图表 (300 DPI) |

**验证**: `tests/test_evaluation.py` - 指标计算验证、图表落盘检查 (assert >10KB)

**输出图表**:
```
docs/figures/
├── equity_curves.png      # 三策略资金曲线对比
├── drawdown_heatmap.png   # 最大回撤热力图
├── latency_boxplot.png    # 延迟分布箱线图
├── trade_quality_comparison.png   # 交易质量对比
├── market_state_heatmap.png       # 市场状态热力图
├── ml_shap_importance.png         # ML SHAP 归因
├── llm_reasoning_wordcloud.png    # LLM Reasoning 词云
└── cross_market_radar.png         # 跨市场雷达图
```

---

### Phase 7: CLI & Docker (Complete)
**主入口**: `main.py`

| 子命令 | 功能 |
|-------|------|
| `python main.py run` | 执行完整回测流水线 (Phase 1-6) |
| `python main.py evaluate` | 重新生成评估图表 |
| `python main.py evaluate-advanced` | 生成高级学术评估报告 (Phase 8) |
| `python main.py cache-build` | 构建 LLM 离线缓存 |

**模型训练脚本**: `scripts/train_ml_models.py`

```bash
# 训练 A股模型
uv run python scripts/train_ml_models.py --symbol CSI300

# 训练美股模型
uv run python scripts/train_ml_models.py --symbol QQQ

# 训练所有市场
uv run python scripts/train_ml_models.py --all
```

**模型保存位置**: `models/{symbol.lower()}_{model_type}/`

**容器化**:
- `Dockerfile`: Linux/CUDA 部署镜像
- `docker-compose.yml`: 服务编排配置
- `requirements.txt`: 依赖导出（由 uv 生成）

---

## 6. Module Organization
```
src/
├── data/              # Phase 1: 数据获取与对齐
│   ├── market_data.py     # OHLCV 数据获取 (akshare, yfinance)
│   ├── news_data.py       # 新闻/情绪数据生成
│   ├── data_aligner.py    # 时间对齐与缺失值处理
│   └── fetch_real_news.py # 真实新闻获取
├── models/
│   ├── ml_track/      # Phase 2: 机器学习轨道
│   │   ├── features.py    # 特征工程 (50+ 技术指标)
│   │   └── baselines.py   # 基准模型 (LR, LSTM, LightGBM)
│   ├── llm_track/     # Phase 3: 大语言模型轨道
│   │   ├── prompts.py     # Chain-of-Thought 提示模板 (A股)
│   │   ├── us_prompts.py  # 美股提示模板
│   │   └── agent.py       # LLM 智能体 (DeepSeek, Qwen, Ollama)
│   └── model_manager.py   # 模型管理器
├── orchestrator/      # Phase 4: 编排器
│   ├── fusion_engine.py   # 信号融合引擎
│   ├── signal_converter.py # 信号转换器
│   └── comparator.py      # 对比分析器
├── execution/         # Phase 5: 回测执行引擎
│   ├── bt_engine.py       # Backtrader 封装
│   ├── base_strategy.py   # 基础策略
│   ├── a_share_strategy.py # A股策略
│   └── us_market_strategy.py # 美股策略
├── evaluation/        # Phase 6: 多维度评估
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
└── output/            # 回测结果输出
```

## 7. Testing Requirements
所有模块必须包含对应的验证测试，位于 `tests/` 目录：

| 模块 | 测试文件 | 核心验证点 |
|-----|---------|-----------|
| Data (Phase 1) | `test_data_module.py` | 数据获取、对齐、缺失值处理 |
| ML Track (Phase 2) | `test_ml_track.py` | 未来函数审计、MPS 设备验证 |
| LLM Track (Phase 3) | `test_llm_track.py` | JSON 解析容错、执行器连通性 |
| Orchestrator (Phase 4) | `test_orchestrator.py` | 否决机制、信号融合逻辑 |
| Execution (Phase 5) | `test_bt_engine.py` | 订单成交验证、分析器提取 |
| Evaluation (Phase 6) | `test_evaluation.py` | 指标计算、图表落盘检查 |
| Advanced Evaluation | `test_advanced_evaluation.py` | MAE/MFE、市场状态、SHAP 归因 |
| US Market | `test_us_market.py` | 美股市场回测验证 |
| US News | `test_us_news_data.py` | 美股新闻数据处理 |
| Backtest | `test_backtest.py` | 回测集成测试 |

## 8. Code Style
- **Language**: Python 3.12+
- **Type Hints**: 所有函数必须包含类型注解
- **Docstrings**: 使用 Google 风格的中文文档字符串
- **Imports**: 标准库 → 第三方库 → 本地模块（按此顺序）
- **Line Length**: 最大 100 字符

## 9. LLM Configuration
项目支持多种 LLM 后端：

| 后端 | 环境变量 | 用途 |
|-----|---------|-----|
| DeepSeek (云端) | `DEEPSEEK_API_KEY` | 高质量推理、低成本 |
| Qwen (云端) | `OPENAI_API_KEY` / `DASHSCOPE_API_KEY` | 通义千问系列 |
| SiliconFlow | `SILICONFLOW_API_KEY` | 多模型代理平台 |
| Ollama (本地) | `OLLAMA_HOST` | 免费、低延迟、隐私保护 |
| Mock | - | 离线测试、无需 API |

## 10. Key Metrics for Paper
论文对比分析必须包含以下指标：

**金融指标:**
- Sharpe Ratio (风险调整收益)
- Maximum Drawdown (最大回撤)
- Win Rate (胜率)
- Sortino Ratio / Calmar Ratio

**工程指标:**
- Average Latency (平均推理延迟)
- Throughput (吞吐量)
- Cost-per-Alpha (每个 Alpha 信号成本)

## 11. Output Directories
```
docs/
├── figures/           # 论文图表 (PNG, 300 DPI)
│   ├── equity_curves.png
│   ├── drawdown_heatmap.png
│   └── latency_boxplot.png
├── cache/
│   └── llm_responses/ # LLM 离线缓存 (.jsonl)
└── output/            # 回测结果输出
```

---

*Last Updated: 2026-03-11*
*Project Status: Phase 8 Complete (Advanced Academic Evaluation Framework)*