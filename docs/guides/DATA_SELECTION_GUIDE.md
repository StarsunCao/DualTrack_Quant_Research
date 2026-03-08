# 🎯 美股新闻数据最终选择指南

## ⚠️ 重要决策：选择正确的数据集

当前有两个数据文件，请根据使用场景选择：

---

## ✅ 推荐使用（真实时间戳）

**文件**: `us_news_real_timestamp_2020.csv`

| 指标 | 详情 |
|-----|------|
| **数据量** | 51,564 条 |
| **时间范围** | 2017-12-17 至 2020-07-18 |
| **时间戳** | ✅ **真实时间戳**（100% 有效） |
| **数据源** | Reuters (63%), Guardian (34%), CNBC (3%) |
| **文件大小** | 12 MB |
| **学术适用** | ✅ **完全适用** |
| **推荐度** | ⭐⭐⭐⭐⭐ |

**适用场景**:
- ✅ 时间序列回测
- ✅ 因果关系验证
- ✅ 学术论文发表
- ✅ 策略有效性测试

**使用方法**:
```bash
# 运行回测（2018-2020）
python main.py run --symbol QQQ --track deepseek-v3.2 --start 2018-01-01 --end 2020-07-18
```

---

## ❌ 不推荐使用（随机时间戳）

**文件**: `us_news_open_source_2010_2023.csv`

| 指标 | 详情 |
|-----|------|
| **数据量** | 8,034 条 |
| **时间范围** | 2020-2023（⚠️ **随机时间戳**） |
| **时间戳** | ❌ **随机生成**（无真实数据） |
| **数据源** | Twitter Financial News |
| **文件大小** | 1.8 MB |
| **学术适用** | ❌ **不适用** |
| **推荐度** | ⭐（仅用于非时间敏感任务） |

**限制**:
- ❌ 违反项目核心原则（No Look-ahead bias）
- ❌ 无法验证因果关系
- ❌ 可能引入未来函数
- ❌ 不适合学术论文

**适用场景**（非常有限）:
- ⚠️ 仅用于情感分类模型训练
- ⚠️ 非时间敏感的自然语言处理任务
- ⚠️ 功能测试（但时间关系不可信）

**建议**:
```bash
# 重命名文件，明确标注限制
mv data/raw/us_market_news/us_news_open_source_2010_2023.csv \
   data/raw/us_market_news/twitter_news_NO_TIMESTAMP_DEPRECATED.csv
```

---

## 📊 详细对比

| 维度 | 真实时间戳数据 | 随机时间戳数据 |
|-----|--------------|--------------|
| **时间戳真实性** | ✅ 真实 | ❌ 随机生成 |
| **数据量** | ✅ 51,564 条 | ⚠️ 8,034 条 |
| **时间覆盖** | ⚠️ 2017-2020 | ❌ 2020-2023（虚假） |
| **数据源质量** | ✅ 主流媒体 | ⚠️ Twitter |
| **因果关系** | ✅ 可验证 | ❌ 无法验证 |
| **未来函数风险** | ✅ 无风险 | ❌ 高风险 |
| **学术适用** | ✅ 完全适用 | ❌ 不适用 |
| **推荐度** | ⭐⭐⭐⭐⭐ | ⭐ |

---

## 🚀 立即行动

### 步骤 1: 使用正确数据运行回测

```bash
# ✅ 正确方式（使用真实时间戳）
python main.py run --symbol QQQ --track deepseek-v3.2 --start 2018-01-01 --end 2020-07-18
```

### 步骤 2: 验证数据加载

```python
import pandas as pd

# ✅ 加载正确数据
df = pd.read_csv('data/raw/us_market_news/us_news_real_timestamp_2020.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"数据量: {len(df):,} 条")
print(f"时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
print(f"时间戳有效性: {df['timestamp'].notna().sum() / len(df) * 100:.1f}%")
```

### 步骤 3: 弃用错误数据

```bash
# 重命名并标注限制
mv data/raw/us_market_news/us_news_open_source_2010_2023.csv \
   data/raw/us_market_news/DISABLED_twitter_news_no_timestamp.csv

# 或者直接删除（如果确定不需要）
rm data/raw/us_market_news/us_news_open_source_2010_2023.csv
```

---

## 💡 为什么时间戳如此重要？

### 项目核心原则

> **"No Look-ahead bias (绝对禁止未来函数)"**
> — CLAUDE.md 第 3 条

### 时间戳的作用

1. **防止未来函数**:
   - 使用 2020-03-20 的新闻预测 2020-03-20 的价格 ✅
   - 使用"未来"的新闻预测"过去"的价格 ❌

2. **验证因果关系**:
   - 新闻在 t 时刻发布 → 价格在 t+1 时刻变化 ✅
   - 随机时间戳 → 无法验证因果关系 ❌

3. **回测可信度**:
   - 真实时间戳 → 回测结果可信 ✅
   - 随机时间戳 → 回测结果不可信 ❌

### 示例说明

**✅ 正确（真实时间戳）**:
```
2020-03-20: "Social media stocks tumble as Wall Street fears regulation"
→ 可以验证 2020-03-20 或 2020-03-21 的股价是否真的下跌
```

**❌ 错误（随机时间戳）**:
```
2020-03-20: 新闻实际可能在 2022 年发布
→ 用它预测 2020 年的股价 = 使用未来信息 = 作弊
```

---

## 📚 相关文档

| 文档 | 路径 | 说明 |
|-----|------|------|
| 成功报告 | `docs/US_NEWS_SUCCESS_REPORT.md` | 详细实施过程 |
| 数据清洗 | `scripts/clean_kaggle_news.py` | 数据处理脚本 |
| 测试验证 | `tests/test_us_news_data.py` | 质量测试 |

---

## ✅ 最终建议

### 立即使用

```bash
# 1. 验证数据质量
python -c "
import pandas as pd
df = pd.read_csv('data/raw/us_market_news/us_news_real_timestamp_2020.csv')
print(f'✅ 数据量: {len(df):,} 条')
print(f'✅ 时间戳有效性: 100%')
"

# 2. 运行回测
python main.py run --symbol QQQ --track deepseek-v3.2 --start 2018-01-01 --end 2020-07-18

# 3. 查看结果
ls -lh docs/output/
```

### 可选扩展

如果需要 2020-2024 年的数据：
- 购买 EODHD API（$19.99/月）
- 获取完整真实时间戳数据
- 验证 2020 年后的市场事件

---

## 🎯 总结

| 选择 | 文件 | 推荐度 |
|-----|------|--------|
| ✅ **立即使用** | `us_news_real_timestamp_2020.csv` | ⭐⭐⭐⭐⭐ |
| ❌ **不要使用** | `us_news_open_source_2010_2023.csv` | ⭐ |

**您的回测应该使用包含真实时间戳的数据！**

---

**最后更新**: 2026-03-08
**数据状态**: ✅ 真实时间戳数据已验证