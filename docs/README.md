# 文档目录说明

本目录包含 DualTrack Quant Research 项目的所有文档，按以下结构组织：

```
docs/
├── README.md                    # 本说明文件
├── code_review/                 # 代码审查报告
│   ├── CODE_REVIEW_REPORT.md           # 初次代码审查报告
│   └── CODE_REVIEW_REPORT_FOLLOWUP.md  # 复审报告（验证修复情况）
├── bug_fixes/                   # Bug修复记录
│   ├── BUG_FIX_SUMMARY.md              # 修复摘要（快速概览）
│   ├── CODE_FIX_RECORD.md              # 详细修复记录
│   └── NEW_ISSUES_FOUND.md             # 新发现的问题（待修复）
├── reference/                   # 参考资料
│   └── Masters_practice_Cao Xinyang_321793.pdf  # 项目需求文档
├── figures/                     # 生成的图表
├── output/                      # 回测输出结果
└── cache/                       # 缓存数据
```

---

## 各目录说明

### code_review/ - 代码审查报告
存放所有代码审查相关的报告文件。

| 文件 | 说明 |
|-----|------|
| `CODE_REVIEW_REPORT.md` | 初次代码审查报告，包含发现的问题列表 |
| `CODE_REVIEW_REPORT_FOLLOWUP.md` | 复审报告，验证前期问题的修复情况 |

### bug_fixes/ - Bug修复记录
存放Bug修复相关的记录文件。

| 文件 | 说明 |
|-----|------|
| `BUG_FIX_SUMMARY.md` | 修复摘要，快速概览所有修复内容 |
| `CODE_FIX_RECORD.md` | 详细修复记录，包含代码片段和验证结果 |
| `NEW_ISSUES_FOUND.md` | 复审中新发现的问题（待修复） |

### reference/ - 参考资料
存放项目相关的参考文档。

| 文件 | 说明 |
|-----|------|
| `Masters_practice_Cao Xinyang_321793.pdf` | 项目需求/论文文档 |

### figures/ - 图表输出
存放回测生成的可视化图表（资金曲线、回撤热力图等）。

### output/ - 回测输出
存放回测结果数据文件。

### cache/ - 缓存数据
存放 LLM 响应缓存等临时数据文件。

---

## 使用建议

1. **查看审查结果**：先阅读 `code_review/CODE_REVIEW_REPORT_FOLLOWUP.md` 了解最新状态
2. **查看修复详情**：参考 `bug_fixes/BUG_FIX_SUMMARY.md` 了解修复内容
3. **查看待修复问题**：查看 `bug_fixes/NEW_ISSUES_FOUND.md` 了解待办事项

---

## 项目规划文档 (新增)

| 文档 | 描述 | 适用读者 |
|------|------|----------|
| [PROJECT_ROADMAP.md](./PROJECT_ROADMAP.md) | 项目路线图与里程碑规划 | 项目经理、研究人员 |
| [EXPERIMENT_PLAN.md](./EXPERIMENT_PLAN.md) | 详细实验执行计划 | 实验员、研究人员 |
| [TECHNICAL_SPEC.md](./TECHNICAL_SPEC.md) | 技术规范与API文档 | 开发人员、研究人员 |

---

## 更新记录

- **2026-03-01**: 整理文档目录结构，将报告文件分类存放
- **2026-03-01**: 新增项目规划文档 (PROJECT_ROADMAP.md, EXPERIMENT_PLAN.md, TECHNICAL_SPEC.md)
