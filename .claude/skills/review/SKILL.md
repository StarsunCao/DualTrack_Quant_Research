# Code Review Skill

审查项目代码质量、性能和符合性。

## 触发词
`/review`

## 用途
对项目的指定模块或整个代码库进行全面审查，检查：
- 代码质量和最佳实践
- 潜在的 Bug 和安全问题
- 性能优化机会
- 文档要求符合性

## 使用方法

```bash
# 审查整个项目
/review

# 审查指定模块
/review src/models/ml_track/

# 审查特定文件
/review src/execution/bt_engine.py
```

## 审查标准

### 1. 数据安全（最高优先级）
- 未来函数检查（禁止使用未来数据）
- 数据对齐正确性
- 缺失值处理

### 2. 错误处理
- 异常捕获完整性
- 错误消息清晰度
- 降级策略

### 3. 性能优化
- 算法复杂度
- 内存使用效率
- 并发安全性

### 4. 代码风格
- 类型注解完整性
- 文档字符串质量
- 命名规范

## 输出格式

审查报告将包含：
- 问题等级（🔴 高 / 🟡 中 / 🟢 低）
- 具体问题描述
- 代码位置（文件:行号）
- 改进建议
- 示例代码修复

## 示例输出

```
🔍 代码审查报告 - src/models/llm_track/agent.py

🔴 高优先级问题 (2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

问题 1: 缓存文件损坏未处理
位置: Line 97-102
风险: JSONL 文件损坏会导致整个缓存加载失败

建议修复:
```python
if cache_file.exists():
    corrupted_lines = 0
    with open(cache_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line.strip())
                self.cache[record["cache_key"]] = record
            except json.JSONDecodeError as e:
                corrupted_lines += 1
                print(f"⚠️ 缓存文件第 {line_num} 行损坏: {e}")
    if corrupted_lines > 0:
        print(f"⚠️ 共跳过 {corrupted_lines} 行损坏缓存")
```

✅ 审查完成，报告已更新至 docs/CODE_REVIEW_REPORT.md
```

## 相关文档
- `docs/CODE_REVIEW_REPORT.md` - 审查报告存储位置
- `CLAUDE.md` - 项目代码规范