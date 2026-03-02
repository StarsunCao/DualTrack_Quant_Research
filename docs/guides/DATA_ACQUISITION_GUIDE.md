# DualTrack 数据获取与配置指南

**版本**: 1.0
**日期**: 2026-03-01
**分类**: 工程实施指南
**适用范围**: DualTrack五轨道量化对比平台

---

## 目录

1. [概述](#1-概述)
2. [ML Track数据需求](#2-ml-track数据需求)
3. [LLM Track数据需求](#3-llm-track数据需求)
4. [数据获取方案详解](#4-数据获取方案详解)
5. [推荐实施路径](#5-推荐实施路径)
6. [数据质量验证](#6-数据质量验证)
7. [成本与风险评估](#7-成本与风险评估)

---

## 1. 概述

### 1.1 数据在DualTrack中的定位

DualTrack的核心定位是**"完成对比实验的工程基础设施（Testbed）"**，数据质量直接决定实验结论的可信度。五轨道架构需要以下数据支撑：

```
┌─────────────────────────────────────────────────────────────────┐
│                     DualTrack 数据架构                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  OHLCV价格    │  │   新闻文本    │  │  宏观指标     │          │
│  │  (必需)       │  │  (LLM必需)   │  │   (可选)     │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                  │                  │
│         └─────────────────┼──────────────────┘                  │
│                           ▼                                      │
│              ┌─────────────────────┐                           │
│              │    DataAligner      │                           │
│              │  • 时间对齐         │                           │
│              │  • 缺失值填充(ffill)│                           │
│              │  • 交易日对齐       │                           │
│              └─────────────────────┘                           │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  ML Tracks  │  │  LLM Tracks │  │  Comparator │             │
│  │  (50+因子)  │  │ (语义分析)  │  │  (对比分析) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 数据需求总览

| 数据类型 | ML Track | LLM Track | 优先级 | 获取难度 |
|----------|----------|-----------|--------|----------|
| OHLCV价格 | ✅ 必需 | ⚠️ 参考 | P0 | 低 |
| 新闻文本 | ❌ 不需要 | ✅ 必需 | P0 | 中 |
| 宏观指标 | ⚠️ 可选 | ⚠️ 可选 | P2 | 中 |
| 财报数据 | ⚠️ 可选 | ⚠️ 可选 | P1 | 高 |
| 社交情绪 | ❌ 不需要 | ⚠️ 可选 | P2 | 高 |

---

## 2. ML Track数据需求

### 2.1 当前实现

**数据源**:
- **akshare**: `stock_zh_index_daily(symbol="sh000300")` - 沪深300
- **yfinance**: `Ticker("QQQ").history()` - NASDAQ-100代理

**数据字段**:
```python
{
    "date": DatetimeIndex,   # 日期索引
    "open": float64,         # 开盘价
    "high": float64,         # 最高价
    "low": float64,          # 最低价
    "close": float64,        # 收盘价
    "volume": int64,         # 成交量
    "symbol": str,           # 标的代码
}
```

### 2.2 特征工程输出

基于纯OHLCV数据，`FeatureEngineer`计算**50+技术因子**：

| 因子类别 | 具体指标 | 计算公式 | 用途 |
|----------|----------|----------|------|
| **收益率** | return_1d/5d/10d/20d | `pct_change(n)` | 多周期趋势 |
| **动量** | momentum_5d/10d/20d | `close/close.shift(n) - 1` | 价格动量 |
| **均线** | ma_5/10/20/60 | `rolling(n).mean()` | 趋势判断 |
| **MACD** | macd/signal/histogram | EMA(12)-EMA(26) | 趋势反转 |
| **RSI** | rsi_6/14/24 | 100-100/(1+RS) | 超买超卖 |
| **布林带** | bb_upper/lower/width | MA±2σ | 波动率通道 |
| **ATR** | atr_5d/10d/20d | True Range均值 | 止损设置 |
| **成交量** | OBV/VWAP/volume_ratio | 量价关系 | 资金流向 |
| **价格形态** | upper_shadow/lower_shadow | K线形态 | 市场情绪 |

### 2.3 数据充分性评估

#### ✅ 对于学术研究：足够
- 覆盖技术分析的主要维度
- 支持LR/LSTM/LightGBM等模型有效运行
- 与同类学术论文的数据维度相当

#### ⚠️ 对于生产系统：不足

| 缺失数据类型 | 影响 | 建议接入方式 |
|--------------|------|--------------|
| **基本面因子** | 无法评估公司内在价值 | 接入akshare财务数据接口 |
| **资金流向** | 无法识别主力行为 | 北向资金、融资融券数据 |
| **宏观指标** | 无法判断经济周期 | 央行利率、CPI、PMI |
| **产业链数据** | 无法识别行业轮动 | 行业指数、供应链数据 |

### 2.4 扩展建议（按优先级）

```python
# P1: 基本面因子（价值/质量因子）
from akshare import stock_financial_report_em
# PE/PB/ROE/营收增长率

# P2: 资金流向因子
from akshare import stock_hsgt_hist_em
# 北向资金净流入

# P3: 宏观因子
from akshare import macro_china_cpi
# CPI同比/环比
```

---

## 3. LLM Track数据需求

### 3.1 当前Prompt设计分析

当前`SentimentPromptBuilder`设计的Chain-of-Thought框架：

```
用户输入:
├── 市场背景 (market_context)
│   └── 当前指数涨跌、成交额
└── 新闻文本 (news_text)
    └── 政策/公司/行业新闻

LLM推理步骤:
1. 信息提取 → 关键信息点、涉及主体
2. 情绪分析 → 情感倾向、影响方向
3. 影响评估 → 影响程度、持续时间
4. 决策推理 → 交易建议、确信程度
5. 结构化输出 → JSON格式信号
```

### 3.2 可扩展的数据维度

基于当前架构，LLM可以消费更多数据类型：

| 数据类型 | 具体来源 | LLM分析价值 | 接入难度 |
|----------|----------|-------------|----------|
| **央行政策** | 央行公告、FOMC纪要 | 宏观流动性判断 | 低 |
| **行业研报** | 券商深度报告 | 行业景气度评估 | 中 |
| **公司财报** | 季报/年报MD&A | 基本面深度分析 | 中 |
| **ESG报告** | 可持续发展报告 | 长期风险评估 | 高 |
| **分析师预期** | 盈利预测调整 | 市场情绪指标 | 中 |
| **供应链数据** | 订单、库存 | 行业先行指标 | 高 |
| **地缘政治** | 冲突、贸易政策 | 黑天鹅预警 | 中 |

### 3.3 数据质量对LLM的影响

| 数据质量维度 | 低质量表现 | 对LLM的影响 | 改进方案 |
|--------------|------------|-------------|----------|
| **时效性** | 新闻延迟T+1 | 信号滞后 | 接入实时API |
| **相关性** | 噪音新闻多 | 决策噪音 | 增加过滤规则 |
| **覆盖度** | 仅单一来源 | 观点偏差 | 多源交叉验证 |
| **结构化** | 非结构化文本 | 解析困难 | 使用统一schema |

---

## 4. 数据获取方案详解

### 4.1 沪深市场数据方案

#### 方案A：全自动方案（推荐）

```python
from src.data.market_data import MarketDataFetcher
from src.data.fetch_real_news import RealNewsFetcher

# 价格数据（自动）
fetcher = MarketDataFetcher()
csi300 = fetcher.fetch_csi300("2020-01-01", "2024-01-01")

# 新闻数据（自动）
news_fetcher = RealNewsFetcher()
news = news_fetcher.fetch_all_news("2020-01-01", "2024-01-01")
```

**覆盖范围**:
- ✅ OHLCV: akshare自动获取
- ✅ 新闻: CCTV新闻联播（宏观政策）
- ✅ 公告: A股重大事项公告

**优势**: 零成本、中文源、政策敏感
**局限**: 新闻覆盖度有限（缺少社交媒体、外资观点）

#### 方案B：手动导入方案

适用场景：已有历史数据，需要导入系统

```python
# scripts/load_real_data.py
def load_and_clean_csi300(csv_path: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    加载并清洗手动下载的沪深300数据。

    支持格式:
    - 同花顺、东方财富、通达信导出的CSV
    - Wind、Choice金融终端数据

    必需列: date, open, high, low, close, volume
    """
    df = pd.read_csv(csv_path)

    # 列名标准化
    column_mapping = {
        '日期': 'date', 'Date': 'date',
        '开盘': 'open', 'Open': 'open',
        '最高': 'high', 'High': 'high',
        '最低': 'low', 'Low': 'low',
        '收盘': 'close', 'Close': 'close',
        '成交量': 'volume', 'Volume': 'volume',
        '成交额': 'amount', 'Amount': 'amount',
    }
    df = df.rename(columns=column_mapping)

    # 数据清洗...
    return df
```

### 4.2 美股(NASDAQ-100)数据方案

#### OHLCV数据（已有）

```python
# yfinance自动获取
fetcher = MarketDataFetcher()
nasdaq = fetcher.fetch_nasdaq100("2020-01-01", "2024-01-01")
```

#### 新闻数据（需扩展）

| 数据源 | 类型 | 费用 | 推荐度 | 接入代码示例 |
|--------|------|------|--------|--------------|
| **NewsAPI** | 综合新闻 | 免费/月$449 | ⭐⭐⭐ | `requests.get(newsapi_url)` |
| **Alpha Vantage** | 金融新闻 | 免费/月$49 | ⭐⭐ | `av_api.get_news()` |
| **FMP** | 美股专业 | 月$19-79 | ⭐⭐⭐⭐ | `fmp_api.get_news_sentiment()` |
| **Bloomberg API** | 机构级 | 企业定价 | ⭐⭐⭐⭐⭐ | 需商业授权 |
| **Twitter/X API** | 社交情绪 | 月$100 | ⭐⭐ | `tweepy.Client()` |

推荐接入方案：

```python
# 低成本方案：NewsAPI（适合论文实验）
import requests

def fetch_us_news(symbol: str, api_key: str, start_date: str, end_date: str):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{symbol} OR NASDAQ OR Fed",
        "from": start_date,
        "to": end_date,
        "language": "en",
        "sortBy": "publishedAt",
        "apiKey": api_key,
    }
    response = requests.get(url, params=params)
    return response.json()["articles"]

# 专业方案：Financial Modeling Prep（适合实盘）
from fmp_api import FMPClient  # 需安装fmp-api

def fetch_fmp_news(symbol: str, api_key: str):
    client = FMPClient(api_key)
    return client.get_news_sentiment(symbol)
```

### 4.3 方案对比矩阵

| 维度 | 沪深(akshare) | 美股(NewsAPI) | 美股(FMP) |
|------|---------------|---------------|-----------|
| **成本** | 免费 | 免费/限量 | $19-79/月 |
| **覆盖度** | 中（官方源） | 高（综合） | 高（金融专业） |
| **时效性** | T+0 | 实时 | 实时 |
| **质量** | 高（官方） | 中（需过滤） | 高（专业） |
| **中文支持** | 是 | 否 | 否 |
| **接入难度** | 低 | 低 | 中 |

---

## 5. 推荐实施路径

### Phase 1: 基础实验（Week 1-2）

**目标**：快速验证五轨道架构可行性

```bash
# 1. 自动获取沪深300数据（已有功能）
python -c "
from src.data.market_data import MarketDataFetcher
fetcher = MarketDataFetcher()
df = fetcher.fetch_csi300('2020-01-01', '2024-01-01')
print(f'获取 {len(df)} 条价格数据')
"

# 2. 自动获取新闻（已有功能）
python -c "
from src.data.fetch_real_news import RealNewsFetcher
fetcher = RealNewsFetcher()
news = fetcher.fetch_all_news('2025-01-01', '2025-03-01')
print(f'获取 {len(news)} 条新闻')
"

# 3. 运行完整回测
python main.py run --track all --compare --symbol CSI300
```

**交付物**：
- ✅ 沪深300完整回测结果
- ✅ 五轨道对比报告
- ✅ 基础数据验证通过

### Phase 2: 美股扩展（Week 3）

**目标**：接入QQQ/NASDAQ-100数据

```bash
# 1. 获取美股价格数据（已有）
python -c "
from src.data.market_data import MarketDataFetcher
fetcher = MarketDataFetcher()
df = fetcher.fetch_qqq('2020-01-01', '2024-01-01')
print(f'获取 {len(df)} 条QQQ价格数据')
"

# 2. 接入NewsAPI获取美股新闻（需实现）
python scripts/fetch_us_news.py --symbol QQQ --start 2020-01-01 --end 2024-01-01

# 3. 运行美股回测
python main.py run --track all --compare --symbol QQQ
```

**交付物**：
- ✅ QQQ完整回测结果
- ✅ 跨市场对比分析

### Phase 3: 数据增强（Week 4+，可选）

**目标**：提升数据质量，探索扩展因子

| 任务 | 优先级 | 预期收益 |
|------|--------|----------|
| 接入北向资金数据 | P2 | ML Track增加资金因子 |
| 接入FMP专业新闻 | P2 | LLM Track提升美股质量 |
| 接入财报数据 | P3 | 增加基本面维度 |
| 接入宏观指标 | P3 | 增加宏观择时 |

---

## 6. 数据质量验证

### 6.1 自动化验证流程

```python
# tests/test_data_quality.py
class TestDataQuality:
    """数据质量验证测试"""

    def test_ohlcv_completeness(self):
        """验证OHLCV数据无缺失"""
        df = fetcher.fetch_csi300("2020-01-01", "2024-01-01")
        assert df.isnull().sum().sum() == 0, "存在缺失值"

    def test_no_future_data(self):
        """验证无未来数据"""
        # 特征工程后检查最后一行日期
        features = engineer.compute_all_features(df)
        assert features.index.max() <= pd.Timestamp.now()

    def test_news_timeline_consistency(self):
        """验证新闻时间线与交易日对齐"""
        news_dates = pd.to_datetime(news_df["timestamp"]).dt.date
        trade_dates = set(ohlcv_df.index.date)
        # 新闻日期应为交易日子集
        assert all(d in trade_dates for d in news_dates if d in trade_dates)

    def test_signal_no_look_ahead(self):
        """验证信号无未来函数"""
        # 信号日期 <= 交易日
        for signal_date, trade_date in zip(signals.index, ohlcv_df.index):
            assert signal_date <= trade_date
```

### 6.2 数据质量检查清单

| 检查项 | 检查方法 | 通过标准 |
|--------|----------|----------|
| **价格连续性** | `df.index.to_series().diff().max() < 7 days` | 无超过7天的断档 |
| **价格合理性** | `close > 0` | 所有价格为正 |
| **涨跌停检查** | `abs(daily_return) < 0.25` | 无超过25%的异常波动 |
| **成交量检查** | `volume > 0` | 成交量为正 |
| **新闻时效** | `news_date <= trade_date` | 信号时间不晚于交易日 |
| **信号对齐** | `len(signals) == len(ohlcv)` | 信号数量匹配 |

---

## 7. 成本与风险评估

### 7.1 数据成本估算

#### 免费方案（当前配置）

| 数据源 | 费用 | 限制 | 适用场景 |
|--------|------|------|----------|
| akshare (沪深) | $0 | 限速 | 学术研究 |
| yfinance (美股) | $0 | 非官方 | 学术研究 |
| CCTV新闻 | $0 | 仅宏观 | 政策敏感型策略 |
| Mock新闻 | $0 | 模拟数据 | 系统测试 |

**总计**: $0/月

#### 专业方案（生产环境）

| 数据源 | 费用 | 用途 |
|--------|------|------|
| Financial Modeling Prep | $79/月 | 美股新闻+数据 |
| Wind/Choice | ¥5000+/月 | A股专业数据 |
| Bloomberg | 企业定价 | 全球机构级数据 |

**总计**: $79-5000+/月

### 7.2 风险评估

| 风险 | 可能性 | 影响 | 应对方案 |
|------|--------|------|----------|
| **akshare接口变更** | 中 | 高 | 定期更新适配器 |
| **NewsAPI限流** | 高 | 中 | 实现指数退避重试 |
| **数据缺失** | 低 | 高 | 使用前向填充(ffill) |
| **未来函数泄露** | 低 | 极高 | 严格时间戳对齐检查 |
| **新闻噪音** | 高 | 中 | LLM置信度过滤 |

### 7.3 数据备份策略

```bash
# 建议的数据备份结构
data/
├── raw/                    # 原始数据
│   ├── csi300_daily.parquet
│   ├── nasdaq100_daily.parquet
│   └── real_csi300_news_3m.csv
├── backup/                 # 定期备份
│   ├── 2026-03-01/
│   └── 2026-02-01/
└── cache/                  # LLM缓存
    └── llm_responses/
        └── llm_cache_CSI300.jsonl
```

---

## 附录A：快速启动命令

```bash
# 1. 安装依赖
uv sync

# 2. 获取数据
python -c "
from src.data.market_data import MarketDataFetcher
from src.data.fetch_real_news import RealNewsFetcher

# 价格数据
m = MarketDataFetcher()
m.fetch_csi300('2020-01-01', '2024-01-01')
m.fetch_qqq('2020-01-01', '2024-01-01')

# 新闻数据
n = RealNewsFetcher()
n.fetch_all_news('2025-01-01', '2025-03-01')
"

# 3. 运行回测
python main.py run --track all --compare --symbol CSI300
```

## 附录B：外部数据API参考

| 服务 | 文档链接 | 注册地址 |
|------|----------|----------|
| NewsAPI | https://newsapi.org/docs | https://newsapi.org/register |
| FMP | https://site.financialmodelingprep.com/developer/docs | 官网注册 |
| Alpha Vantage | https://www.alphavantage.co/documentation/ | https://www.alphavantage.co/support/#api-key |
| AKShare | https://www.akshare.xyz/ | 开源库，直接使用 |

---

**文档维护**: 随数据接口更新同步更新
**负责人**: DualTrack Research Team
**审核状态**: 待审查
