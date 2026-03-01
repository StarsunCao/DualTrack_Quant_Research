# DualTrack 项目文档中心

**版本**: 2.0
**日期**: 2026-03-01
**状态**: 已整理归档

---

## 📁 文档目录结构

```
docs/
├── README.md                          # 本文档 - 文档索引
├── guides/                            # 指导文档（How-to）
│   ├── BACKTEST_FIX_GUIDE.md         # 回测框架修复指南
│   ├── CODE_REFACTOR_GUIDE.md        # 代码重构指南
│   └── ADD_TRADE_DETAILS.md          # 添加交易记录功能指南
├── specs/                             # 技术规范（Specification）
│   ├── TECHNICAL_SPEC.md             # 技术规范文档
│   ├── PROJECT_ROADMAP.md            # 项目路线图
│   └── EXPERIMENT_PLAN.md            # 实验执行计划
├── reviews/                           # 代码审查与架构（Review）
│   ├── CORRECTED_UNDERSTANDING.md    # 核心概念修正说明
│   ├── COMPARISON_FRAMEWORK_REFACTOR.md  # 对比框架重构说明
│   ├── REFACTOR_IMPLEMENTATION_PLAN.md   # 重构实施计划
│   ├── BUG_FIX_SUMMARY.md            # Bug修复汇总（原bug_fixes/）
│   ├── NEW_ISSUES_FOUND.md           # 新发现问题记录
│   ├── REPAIR_VERIFICATION_REPORT.md # 修复验证报告
│   ├── CODE_REVIEW_REPORT.md         # 代码审查报告（原code_review/）
│   └── CODE_REVIEW_REPORT_FOLLOWUP.md # 代码审查跟进报告
├── implementation/                    # 实施记录（Implementation）
│   └── BACKTEST_FIX_IMPLEMENTATION.md    # 回测修复实施记录
├── output/                            # 回测输出（Output）
│   └── track_results/                # 各轨道详细交易记录
├── figures/                           # 论文图表（Figures）
├── cache/                             # 缓存数据（Cache）
│   └── llm_responses/                # LLM响应缓存
└── reference/                         # 外部参考文档
    └── Masters_practice_Cao Xinyang_321793.pdf
```

---

## 📚 文档分类说明

### guides/ - 指导文档
面向开发者的操作指南，回答"怎么做"。

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [BACKTEST_FIX_GUIDE.md](guides/BACKTEST_FIX_GUIDE.md) | 回测框架修复的详细步骤 | 修复回测问题时参考 |
| [CODE_REFACTOR_GUIDE.md](guides/CODE_REFACTOR_GUIDE.md) | 代码重构的具体方案 | 重构代码时参考 |
| [ADD_TRADE_DETAILS.md](guides/ADD_TRADE_DETAILS.md) | 添加交易记录功能的指南 | 实现交易记录保存时参考 |

### specs/ - 技术规范
面向架构师和研究员的设计规范，回答"是什么"和"为什么"。

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [TECHNICAL_SPEC.md](specs/TECHNICAL_SPEC.md) | 技术架构、数据格式、API规范 | 开发新功能时参考 |
| [PROJECT_ROADMAP.md](specs/PROJECT_ROADMAP.md) | 项目里程碑和进度规划 | 项目规划时参考 |
| [EXPERIMENT_PLAN.md](specs/EXPERIMENT_PLAN.md) | 实验设计和执行计划 | 设计实验时参考 |

### reviews/ - 架构审查
记录关键设计决策、架构修正、Bug修复和代码审查。

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [CORRECTED_UNDERSTANDING.md](reviews/CORRECTED_UNDERSTANDING.md) | DualTrack核心概念的修正说明 | 理解项目架构时参考 |
| [COMPARISON_FRAMEWORK_REFACTOR.md](reviews/COMPARISON_FRAMEWORK_REFACTOR.md) | 对比框架重构说明 | 理解五轨道设计时参考 |
| [REFACTOR_IMPLEMENTATION_PLAN.md](reviews/REFACTOR_IMPLEMENTATION_PLAN.md) | 重构实施计划 | 了解重构历史时参考 |
| [BUG_FIX_SUMMARY.md](reviews/BUG_FIX_SUMMARY.md) | Bug修复汇总 | 查看已修复问题 |
| [NEW_ISSUES_FOUND.md](reviews/NEW_ISSUES_FOUND.md) | 新发现问题记录 | 了解待修复问题 |
| [REPAIR_VERIFICATION_REPORT.md](reviews/REPAIR_VERIFICATION_REPORT.md) | 修复验证报告 | 验证修复结果 |
| [CODE_REVIEW_REPORT.md](reviews/CODE_REVIEW_REPORT.md) | 代码审查报告 | 查看审查意见 |
| [CODE_REVIEW_REPORT_FOLLOWUP.md](reviews/CODE_REVIEW_REPORT_FOLLOWUP.md) | 代码审查跟进 | 查看审查后续 |

### implementation/ - 实施记录
记录具体的代码实施过程。

| 文档 | 内容 | 适用场景 |
|------|------|----------|
| [BACKTEST_FIX_IMPLEMENTATION.md](implementation/BACKTEST_FIX_IMPLEMENTATION.md) | 回测修复的具体实施记录 | 查看实施细节时参考 |

---

## 🎯 快速导航

### 如果你是新加入的开发者
1. 先看 [reviews/CORRECTED_UNDERSTANDING.md](reviews/CORRECTED_UNDERSTANDING.md) - 理解项目核心概念
2. 再看 [specs/TECHNICAL_SPEC.md](specs/TECHNICAL_SPEC.md) - 了解技术架构
3. 最后看 [guides/BACKTEST_FIX_GUIDE.md](guides/BACKTEST_FIX_GUIDE.md) - 了解当前任务

### 如果你要修复回测问题
直接参考 [guides/BACKTEST_FIX_GUIDE.md](guides/BACKTEST_FIX_GUIDE.md)

### 如果你要实现新功能
参考 [specs/TECHNICAL_SPEC.md](specs/TECHNICAL_SPEC.md) 中的接口规范

### 如果你要运行实验
参考 [specs/EXPERIMENT_PLAN.md](specs/EXPERIMENT_PLAN.md)

---

## 📂 输出目录说明

### docs/output/
回测运行后自动生成的输出文件（图表已移至 docs/figures/）：

```
output/
└── track_results/              # 各轨道详细记录（实现ADD_TRADE_DETAILS后）
    ├── lr/
    │   ├── trades.csv
    │   ├── positions.csv
    │   └── report.txt
    ├── lstm/
    ├── lgb/
    ├── llm-cloud/
    └── llm-local/
```

### docs/figures/
回测图表和论文用图：

```
figures/
├── equity_curves_CSI300.png    # 资金曲线对比
├── drawdown_heatmap.png        # 回撤热力图
├── latency_boxplot.png         # 延迟箱线图
├── rolling_sharpe.png          # 滚动夏普比率
└── underwater.png              # 回撤水下曲线
```

### docs/cache/
预计算的LLM响应缓存：

```
cache/
└── llm_responses/
    └── llm_cache_CSI300.jsonl
```

---

## 📝 文档维护规范

### 新增文档时
1. 根据类型放入对应目录（guides/specs/reviews/implementation）
2. 在本文档的对应表格中添加条目
3. 保持文件名使用大写下划线格式

### 修改文档时
1. 更新文档头部的版本号和日期
2. 在变更日志中添加修改说明
3. 如涉及架构变更，同步更新相关文档

### 删除文档时
1. 确保没有其他文档引用该文件
2. 在本文档的表格中标记为"已废弃"
3. 保留至少一个版本的历史备份

---

## 🔗 相关链接

- 主程序: `main.py`
- 项目配置: `pyproject.toml`
- 代码目录: `src/`
- 测试目录: `tests/`
- 数据目录: `data/`

---

**维护者**: Claude Code
**最后更新**: 2026-03-01
