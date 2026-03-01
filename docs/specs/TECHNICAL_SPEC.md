# DualTrack Quant Research - 技术规范文档

**版本**: 1.0
**日期**: 2026-03-01

---

## 1. 架构设计规范

### 1.1 数据流架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据输入层                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ 市场数据(OHLCV)│  │   新闻数据    │  │  宏观经济数据  │          │
│  │  (akshare)   │  │  (Mock/Real) │  │   (可选)     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼────────────────┼────────────────┼──────────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        数据处理层                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    DataAligner                          │   │
│  │  • 时间对齐  • 缺失值填充(ffill)  • 频率统一化            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      双轨信号生成层                               │
├───────────────────────────┬─────────────────────────────────────┤
│      ML Track             │           LLM Track                 │
├───────────────────────────┼─────────────────────────────────────┤
│  ┌─────────────────────┐  │  ┌─────────────────────────────┐   │
│  │   FeatureEngineer   │  │  │   SentimentPromptBuilder    │   │
│  │  • 技术指标计算      │  │  │  • Chain-of-Thought 模板    │   │
│  │  • 收益率因子        │  │  │  • JSON格式规范             │   │
│  │  • 波动率指标        │  │  └─────────────────────────────┘   │
│  └─────────────────────┘  │              │                      │
│           │               │              ▼                      │
│           ▼               │  ┌─────────────────────────────┐   │
│  ┌─────────────────────┐  │  │      LLMTradingAgent        │   │
│  │  MLStrategyPortfolio │  │  ├───────────────────────────┤   │
│  │  • LogisticRegression│  │  │  Executors:               │   │
│  │  • LSTM (MPS优化)    │  │  │  - OllamaExecutor         │   │
│  │  • LightGBM          │  │  │  - DeepSeekExecutor       │   │
│  └─────────────────────┘  │  │  - MockExecutor            │   │
│           │               │  └─────────────────────────────┘   │
│           ▼               │              │                      │
│  [timestamp, symbol,      │              ▼                      │
│   model_name,             │  ┌─────────────────────────────┐   │
│   signal_strength_0_to_1] │  │    CacheManager (.jsonl)    │   │
│                           │  │  • 离线缓存  • 断点续传       │   │
│                           │  └─────────────────────────────┘   │
│                           │              │                      │
│                           │              ▼                      │
│                           │  [timestamp, symbol,               │
│                           │   llm_signal, reasoning,           │
│                           │   latency_ms]                      │
└───────────────────────────┴─────────────────────────────────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      信号融合与决策层                             │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  SignalFusionEngine                     │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  市场状态检测:                                          │   │
│  │    • Normal (vol < 2%): ML 70% / LLM 30%               │   │
│  │    • High Vol (2% ~ 5%): ML 50% / LLM 50%              │   │
│  │    • Black Swan (> 5%): LLM 100% (否决权)              │   │
│  │                                                         │   │
│  │  优化特性:                                              │   │
│  │    • 调仓死区 (5%阈值)                                  │   │
│  │    • LLM信号衰减 (72小时)                               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  [symbol: TargetPosition] → Dict[str, float] 目标仓位            │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                        执行与回测层                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Backtrader Engine (bt_engine.py)           │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │  • PandasDataFeed: OHLCV数据加载                        │   │
│  │  • DualTrackStrategy: 目标仓位执行                      │   │
│  │  • 佣金/滑点/印花税模拟                                 │   │
│  │  • 分析器: Sharpe, DrawDown, Returns, Trades            │   │
│  └─────────────────────────────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  BacktestResult {initial_cash, final_value, equity_curve, ...} │
└─────────────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      评估与可视化层                               │
├─────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────┐  ┌──────────────────────────────┐  │
│  │   MetricsCalculator    │  │      Visualizer              │  │
│  │  • 金融指标计算         │  │  • 资金曲线图                │  │
│  │  • 工程指标计算         │  │  • 回撤热力图                │  │
│  │  • 多策略对比           │  │  • 延迟分布图                │  │
│  └────────────────────────┘  └──────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 模块依赖关系

```
main.py
├── src.data
│   ├── market_data (akshare, yfinance)
│   ├── news_data
│   └── data_aligner (pandas, numpy)
│
├── src.models.ml_track
│   ├── features (pandas, numpy, sklearn)
│   └── baselines (sklearn, torch, lightgbm)
│
├── src.models.llm_track
│   ├── prompts
│   └── agent (requests, openai)
│
├── src.orchestrator
│   └── fusion_engine (pandas, numpy)
│
├── src.execution
│   └── bt_engine (backtrader)
│
└── src.evaluation
    ├── metrics_calculator (pandas, numpy)
    └── visualizer (matplotlib, seaborn)
```

---

## 2. 数据规范

### 2.1 OHLCV 数据格式

```python
# 标准DataFrame格式
{
    "index": DatetimeIndex,  # 日期时间索引
    "open": float64,         # 开盘价
    "high": float64,         # 最高价
    "low": float64,          # 最低价
    "close": float64,        # 收盘价
    "volume": int64,         # 成交量
    "symbol": str,           # 可选: 股票代码
}

# 示例
                    open     high      low    close     volume    symbol
date
2020-01-02      3850.50   3870.20   3840.10   3865.30   12500000   CSI300
2020-01-03      3865.30   3880.50   3855.20   3872.10   13200000   CSI300
```

### 2.2 新闻数据格式

```python
# 标准DataFrame格式
{
    "timestamp": datetime64,  # 新闻时间戳
    "symbol": str,            # 相关股票代码
    "title": str,             # 新闻标题
    "content": str,           # 新闻内容
    "source": str,            # 新闻来源
    "sentiment": str,         # 可选: 情感标签
}

# 示例
               timestamp    symbol                                             title  \
0  2020-01-02 09:30:00     CSI300   央行降准释放流动性，市场迎来重大利好
1  2020-01-02 14:00:00     CSI300   科技股集体上涨，半导体板块领涨
```

### 2.3 ML Track 信号格式

```python
# DataFrame输出格式
{
    "timestamp": datetime64,
    "symbol": str,
    "model_name": str,           # "LogisticRegression" | "LSTM" | "LightGBM"
    "signal_strength_0_to_1": float64,  # 0.0 ~ 1.0
    "latency_ms": float64,
}

# 示例
            timestamp    symbol          model_name  signal_strength_0_to_1  latency_ms
0 2020-01-02 09:30:00    CSI300  LogisticRegression                    0.65         1.2
1 2020-01-02 09:30:00    CSI300              LSTM                    0.72        15.5
2 2020-01-02 09:30:00    CSI300          LightGBM                    0.68         2.1
```

### 2.4 LLM Track 信号格式

```python
# DataFrame输出格式
{
    "timestamp": datetime64,
    "symbol": str,
    "llm_signal": str,      # "buy" | "sell" | "hold"
    "confidence": float64,  # 0.0 ~ 1.0
    "reasoning": str,       # 推理过程
    "latency_ms": float64,
    "model": str,           # 使用的模型名称
    "parse_success": bool,  # JSON解析是否成功
}

# 示例
            timestamp    symbol llm_signal  confidence                                           reasoning  latency_ms          model  parse_success
0 2020-01-02 09:30:00    CSI300        buy         0.8  央行降准释放流动性，利好股市，建议买入。                850.0  qwen2.5:7b           True
```

### 2.5 融合信号格式

```python
# FusedSignal 数据类
@dataclass
class FusedSignal:
    symbol: str
    ml_signal: float        # -1 ~ 1
    llm_signal: float       # -1 ~ 1
    ml_confidence: float    # 0 ~ 1
    llm_confidence: float   # 0 ~ 1
    ml_latency_ms: float
    llm_latency_ms: float
    fused_weight: float     # -1 ~ 1
    fusion_source: str      # "ml_dominant" | "llm_veto" | "weighted"
    market_regime: str      # "normal" | "high_volatility" | "black_swan"
    reasoning: str
    timestamp: datetime
```

### 2.6 目标仓位格式

```python
# TargetPosition 数据类
@dataclass
class TargetPosition:
    symbol: str
    weight: float           # -1 ~ 1，负数表示做空
    signal_source: str      # "ml_dominant" | "llm_veto" | "fused"
    confidence: float
    reasoning: str
    timestamp: datetime
    latency_metrics: LatencyMetrics
    market_regime: str
    metadata: dict

# Orchestrator 输出格式: Dict[str, float]
{
    "CSI300": 0.8,    # 80% 仓位
    "QQQ": -0.3,      # -30% 仓位（做空）
}
```

---

## 3. 接口规范

### 3.1 CLI 接口

```bash
# 主命令
dualtrack [GLOBAL_OPTIONS] <COMMAND> [COMMAND_OPTIONS]

# Global Options
--verbose, -v       # 详细输出模式
--version           # 显示版本
--help              # 显示帮助

# Commands

## run - 执行完整回测
 dualtrack run [OPTIONS]

Options:
  --symbol, -s TEXT       # 交易标的 (CSI300/QQQ) [default: CSI300]
  --start TEXT            # 回测开始日期 [default: 2020-01-01]
  --end TEXT              # 回测结束日期 [default: 2024-01-01]
  --initial-cash FLOAT    # 初始资金 [default: 100000.0]
  --commission FLOAT      # 佣金率 [default: 0.0002]
  --output-dir TEXT       # 输出目录 [default: docs/output]
  --config FILE           # 配置文件路径

## evaluate - 生成评估图表
dualtrack evaluate [OPTIONS]

Options:
  --log-file FILE         # 回测日志文件路径
  --output-dir TEXT       # 输出目录 [default: docs/figures]

## cache-build - 构建LLM缓存
dualtrack cache-build [OPTIONS]

Options:
  --symbol, -s TEXT       # 交易标的 [default: CSI300]
  --start TEXT            # 开始日期
  --end TEXT              # 结束日期
  --news-file FILE        # 新闻数据文件路径
  --output-dir TEXT       # 缓存输出目录
  --executor TEXT         # LLM执行器 (ollama/deepseek/mock)
```

### 3.2 Python API 接口

```python
# 数据获取
from src.data.market_data import MarketDataFetcher

fetcher = MarketDataFetcher()
csi300 = fetcher.fetch_csi300(start_date="2020-01-01", end_date="2024-01-01")

# 特征工程
from src.models.ml_track.features import FeatureEngineer

engineer = FeatureEngineer()
features = engineer.compute_all_features(ohlcv_data)
feature_names = engineer.get_feature_names()

# ML Track
from src.models.ml_track.baselines import MLStrategyPortfolio

portfolio = MLStrategyPortfolio()
portfolio.fit(features_df, target_col="target_label")
ml_signals = portfolio.predict(features_df, symbol="CSI300")

# LLM Track
from src.models.llm_track.agent import LLMTradingAgent

agent = LLMTradingAgent(executor_type="ollama")
llm_signals = agent.batch_analyze(news_list, symbol="CSI300", cache_path="cache.jsonl")

# 信号融合
from src.orchestrator.fusion_engine import SignalFusionEngine

fusion = SignalFusionEngine()
target_positions = fusion.generate_target_positions(
    ml_signals=ml_signals,
    llm_signals=llm_signals,
    volatility=0.02,
)

# 回测
from src.execution.bt_engine import BacktestEngine, DualTrackStrategy

engine = BacktestEngine(initial_cash=100000, commission=0.0002)
engine.add_data(ohlcv_data, name="CSI300")
engine.add_strategy(DualTrackStrategy, target_positions=target_positions)
result = engine.run()

# 评估
from src.evaluation.metrics_calculator import MetricsCalculator

calculator = MetricsCalculator()
evaluation = calculator.evaluate(
    strategy_name="DualTrack",
    equity_curve=result.equity_curve,
    latency_log=[],
    num_signals=len(target_positions),
)
```

---

## 4. 配置规范

### 4.1 环境变量

```bash
# LLM配置
DEEPSEEK_API_KEY="your-api-key-here"      # DeepSeek API密钥
OLLAMA_HOST="http://localhost:11434"      # Ollama服务地址

# 数据配置
DATA_DIR="./data"                          # 数据目录
CACHE_DIR="./docs/cache"                   # 缓存目录

# 回测配置
INITIAL_CASH="100000"                      # 初始资金
COMMISSION="0.0002"                        # 佣金率
```

### 4.2 配置文件

```yaml
# config/default.yaml
project:
  name: "dualtrack-quant-research"
  version: "0.1.0"
  debug: false

data:
  symbols: ["CSI300", "QQQ"]
  start_date: "2020-01-01"
  end_date: "2024-01-01"
  frequency: "daily"

ml_track:
  models: ["logistic_regression", "lstm", "lightgbm"]
  feature_engineering:
    normalize: true
    drop_na: true
  training:
    test_size: 0.2
    random_state: 42

llm_track:
  executor: "ollama"  # or "deepseek"
  model: "qwen2.5:7b"
  temperature: 0.7
  cache_enabled: true

fusion:
  ml_weight_normal: 0.7
  ml_weight_high_vol: 0.4
  volatility_threshold: 0.03
  rebalance_threshold: 0.05
  signal_decay_hours: 72

backtest:
  initial_cash: 100000.0
  commission: 0.0002
  slippage: 0.0
```

---

## 5. 测试规范

### 5.1 单元测试结构

```
tests/
├── conftest.py              # 共享fixture
├── test_data_module.py      # 数据模块测试
├── test_ml_track.py         # ML Track测试
├── test_llm_track.py        # LLM Track测试
├── test_orchestrator.py     # 融合引擎测试
├── test_bt_engine.py        # 回测引擎测试
└── test_evaluation.py       # 评估模块测试
```

### 5.2 关键测试用例

```python
# tests/test_ml_track.py

class TestFeatureEngineer:
    """特征工程测试"""

    def test_no_look_ahead_bias(self):
        """验证无未来函数"""
        # 在 t 时刻计算因子时，不使用 t+1 及之后的数据
        pass

    def test_feature_range(self):
        """验证特征值范围"""
        # RSI 在 [0, 100]
        # 信号强度在 [0, 1]
        pass

    def test_nan_handling(self):
        """验证NaN处理"""
        # 前向依赖的因子在序列开头有NaN
        # 目标变量在序列结尾有NaN
        pass


class TestMLModels:
    """ML模型测试"""

    def test_model_convergence(self):
        """验证模型收敛"""
        # 训练损失应下降
        pass

    def test_prediction_format(self):
        """验证预测格式"""
        # 输出应为 [timestamp, symbol, model_name, signal_strength_0_to_1]
        pass

    def test_mps_device(self):
        """验证MPS设备使用"""
        # LSTM应使用MPS加速
        pass


# tests/test_llm_track.py

class TestLLMParser:
    """LLM解析器测试"""

    def test_json_extraction(self):
        """测试JSON提取"""
        # 能从Markdown代码块提取JSON
        # 能处理带前缀的JSON
        pass

    def test_fault_tolerance(self):
        """测试容错能力"""
        # 无效JSON返回默认hold信号
        # 置信度超出范围被截断到[0,1]
        pass


class TestLLMCache:
    """LLM缓存测试"""

    def test_cache_hit(self):
        """测试缓存命中"""
        # 缓存命中时延迟应为0
        pass

    def test_cache_persistence(self):
        """测试缓存持久化"""
        # 缓存能正确保存和加载
        pass


# tests/test_orchestrator.py

class TestSignalFusion:
    """信号融合测试"""

    def test_regime_detection(self):
        """测试市场状态检测"""
        # 正常波动率
        # 高波动率
        # 黑天鹅事件
        pass

    def test_veto_mechanism(self):
        """测试否决机制"""
        # ML强烈买入 + LLM强烈卖出 = 卖出
        pass

    def test_rebalancing_dead_zone(self):
        """测试调仓死区"""
        # 仓位变化小于阈值时不调仓
        pass
```

---

## 6. 性能规范

### 6.1 延迟要求

| 组件 | P50 延迟 | P95 延迟 | P99 延迟 |
|------|----------|----------|----------|
| Feature Engineering | < 10ms | < 50ms | < 100ms |
| ML Inference | < 5ms | < 20ms | < 50ms |
| LLM Cache Hit | < 1ms | < 5ms | < 10ms |
| LLM Cache Miss (Ollama) | < 2s | < 5s | < 10s |
| LLM Cache Miss (DeepSeek) | < 1s | < 3s | < 5s |
| Signal Fusion | < 5ms | < 20ms | < 50ms |
| Backtest Execution | < 100ms/日 | - | - |

### 6.2 资源使用

| 组件 | 内存 | CPU | GPU |
|------|------|-----|-----|
| Data Loading | < 500MB | 单核 | - |
| Feature Engineering | < 1GB | 多核 | - |
| LSTM Training | < 4GB | - | MPS/CUDA |
| LLM Inference (Ollama) | < 8GB | 多核 | - |
| Backtest | < 1GB | 单核 | - |

### 6.3 缓存效率

- Cache Hit Rate: > 80%
- Cache Size: ~10MB/1000条新闻
- Cache Load Time: < 1s/10000条

---

## 7. 错误处理规范

### 7.1 错误等级

```python
class ErrorLevel(Enum):
    CRITICAL = "critical"    # 导致程序终止
    ERROR = "error"          # 功能不可用，但程序继续
    WARNING = "warning"      # 功能降级，程序继续
    INFO = "info"            # 提示信息
```

### 7.2 错误处理策略

```python
# 数据获取失败 → 使用缓存数据或模拟数据
try:
    data = fetcher.fetch_csi300(...)
except DataFetchError:
    logger.warning("数据获取失败，使用缓存数据")
    data = load_cached_data()

# LLM API失败 → 使用Mock执行器
try:
    response = deepseek.execute(...)
except APIError:
    logger.warning("DeepSeek API失败，切换到Mock")
    response = mock.execute(...)

# 模型加载失败 → 重新训练
try:
    portfolio.load_models(path)
except FileNotFoundError:
    logger.warning("模型文件不存在，重新训练")
    portfolio.fit(data)
```

---

## 8. 文档规范

### 8.1 代码注释

```python
def compute_rsi(df: pd.DataFrame, periods: list[int] | None = None) -> pd.DataFrame:
    """
    计算相对强弱指标 (RSI)。

    RSI = 100 - 100 / (1 + RS)
    RS = 平均上涨幅度 / 平均下跌幅度

    Args:
        df: 包含 'close' 列的 DataFrame。
        periods: RSI 计算周期列表，默认为 [6, 14, 24]。

    Returns:
        添加了 RSI 列的 DataFrame。

    Raises:
        ValueError: 当输入数据缺少 'close' 列时抛出。

    Example:
        >>> df = pd.DataFrame({"close": [100, 102, 101, 103, 105]})
        >>> result = compute_rsi(df)
        >>> print(result["rsi_14"])
    """
```

### 8.2 日志规范

```python
import logging

logger = logging.getLogger(__name__)

# 信息级别
logger.info("开始训练模型: %s", model_name)
logger.info("数据加载完成: %d 条记录", len(df))

# 警告级别
logger.warning("检测到缺失值: %d 条", nan_count)
logger.warning("使用降级方案: %s", fallback_method)

# 错误级别
logger.error("模型训练失败: %s", str(e))
logger.error("API请求超时: %d 秒", timeout)

# 调试级别
logger.debug("特征矩阵形状: %s", X.shape)
logger.debug("信号值: %.4f", signal)
```

---

## 9. 版本控制规范

### 9.1 提交信息格式

```
<type>(<scope>): <subject>

<body>

<footer>
```

Type:
- `feat`: 新功能
- `fix`: 修复
- `docs`: 文档
- `style`: 格式
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

Example:
```
feat(ml-track): add LSTM model with MPS support

- Implement LSTM network with configurable layers
- Add MPS device detection for Apple Silicon
- Include sequence creation and batching

Closes #123
```

### 9.2 分支策略

```
main          # 稳定版本
├── dev       # 开发分支
│   ├── feature/ml-track-lstm
│   ├── feature/llm-cache
│   └── fix/future-bias
├── release/v0.1.0
└── hotfix/data-aligner
```

---

**文档维护**: 随代码更新同步更新
**负责人**: [待填写]
**审核人**: [待填写]
