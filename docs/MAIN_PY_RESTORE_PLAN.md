# main.py 恢复计划

## 当前状态

main.py 已通过 `git restore` 恢复，但缺少优化修改。

## 需要重新应用的修改

### 修改 1: 数据聚合参数优化 (约第180-240行)

**位置**: `run_backtest()` 函数中的数据聚合部分

**需要修改**:
```python
# 旧代码（需要找到并替换）
daily_news = DataAligner.aggregate_daily_news(
    news_data,
    max_news_per_day=15,
    source_col="source" if "source" in news_data.columns else None
)

# 新代码（推荐配置）
daily_news = DataAligner.aggregate_daily_news(
    news_data,
    max_news_per_day=20,
    max_content_length=150,
    source_col="source" if "source" in news_data.columns else None,
    filter_notices=True  # 智能筛选重要公告
)
```

### 修改 2: 添加 siliconflow 执行器支持 (第739行)

**位置**: `cache_build()` 函数的 executor 参数

**需要修改**:
```python
# 旧代码
@click.option("--executor", default="ollama", type=click.Choice(["ollama", "deepseek", "mock"]), help="LLM 执行器类型")

# 新代码
@click.option("--executor", default="ollama", type=click.Choice(["ollama", "deepseek", "siliconflow", "mock"]), help="LLM 执行器类型")
```

### 修改 3: 提示词优先级修复 (cache_build 函数中)

**位置**: `cache_build()` 函数中的 news_text 参数

**需要修改**:
```python
# 旧代码（优先 structured_summary，仅包含标题）
news_text=news.get("structured_summary", news.get("aggregated_content", news.get("content", "")))

# 新代码（优先 aggregated_content，包含标题+内容）
news_text=news.get("aggregated_content", news.get("structured_summary", news.get("content", "")))
```

### 修改 4: 移除 LLMTradingAgent 的 use_simple_format 参数

**位置**: cache_build() 函数中

**需要修改**:
```python
# 旧代码
llm_agent = LLMTradingAgent(executor_type=executor, model=model, use_simple_format=True)

# 新代码
llm_agent = LLMTradingAgent(executor_type=executor, model=model)
```

## 验证步骤

1. 应用所有修改
2. 运行语法检查: `python -m py_compile main.py`
3. 运行简单测试: `python main.py cache-build --help`
4. 运行硅基流动测试验证提示词

## 其他已完成的修改

✅ `src/data/data_aligner.py` - 已添加智能筛选和内容包含
✅ `src/models/llm_track/agent.py` - 已修复优先级
✅ `scripts/fetch_complete_data.py` - 已添加宏观数据和北向资金

---

**下一步**: 按照此计划系统性地重新应用所有修改。