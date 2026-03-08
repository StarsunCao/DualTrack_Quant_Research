# 美股新闻数据快速使用指南

## 当前状态

✅ **数据已就绪**: 8,034 条美股新闻数据已下载并清洗完成

## 数据文件位置

```
data/raw/us_market_news/us_news_open_source_2010_2023.csv
```

## 快速开始

### 1. 查看数据

```bash
# 查看前 10 行
head -n 10 data/raw/us_market_news/us_news_open_source_2010_2023.csv

# 统计数据量
wc -l data/raw/us_market_news/us_news_open_source_2010_2023.csv

# 查看数据质量报告
python scripts/clean_and_merge_news.py
```

### 2. 运行测试

```bash
# 验证数据质量
python tests/test_us_news_data.py
```

### 3. 集成到回测系统

```bash
# 运行美股回测（使用真实新闻）
python main.py run --symbol QQQ --track deepseek-v3.2 --start 2020-01-01 --end 2023-12-31
```

## 数据格式

| 列名 | 类型 | 说明 | 示例 |
|-----|------|------|------|
| timestamp | datetime | 时间戳（合成） | 2020-01-01 14:52:01 |
| title | string | 新闻标题 | $BYND - JPMorgan reels in expectations... |
| content | string | 新闻内容 | （同标题） |
| source | string | 数据源 | huggingface_twitter |

## 注意事项

⚠️ **重要提示**:
- 时间戳为 2020-2023 年随机分布（用于测试）
- 数据来源：Twitter Financial News
- 数据量：8,034 条

## 获取更多数据（可选）

### 配置 Kaggle API

```bash
# 1. 访问 https://www.kaggle.com/
# 2. My Account -> API -> Create New API Token
# 3. 下载 kaggle.json
# 4. 执行以下命令：

mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 5. 重新运行下载脚本
python scripts/download_open_source_news.py
python scripts/clean_and_merge_news.py
```

### 购买真实数据（推荐用于论文发表）

**EODHD API**: $19.99/月
- 真实时间戳
- 完整 2020-2024 历史数据
- AI 情绪评分

详见：`docs/KAGGLE_SETUP.md`

## 下一步

1. **功能测试**: 集成到回测系统
2. **LLM 测试**: 验证 DeepSeek/Qwen 对英文新闻的理解
3. **优化**: 根据需要补充更多数据

---

**相关文档**:
- 实施总结: `docs/US_NEWS_IMPLEMENTATION_SUMMARY.md`
- 配置指南: `docs/KAGGLE_SETUP.md`
- 测试用例: `tests/test_us_news_data.py`