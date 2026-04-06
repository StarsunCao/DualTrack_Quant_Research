# 论文写作指南

**最后更新**: 2026-03-18
**目标**: 80-100 页英文学术论文

---

## 一、论文结构概览

| 章节 | 页数 | 字数 | 占比 |
|------|------|------|------|
| Abstract | 1-2 | 300-500 | 2% |
| Chapter 1: Introduction | 12-15 | 3,000-4,000 | 14% |
| Chapter 2: Related Work | 20-25 | 5,000-6,000 | 23% |
| Chapter 3: Methodology | 20-25 | 5,000-6,000 | 23% |
| Chapter 4: Experimental Setup | 10-12 | 2,500-3,000 | 11% |
| Chapter 5: Results and Analysis | 18-22 | 4,500-5,500 | 20% |
| Chapter 6: Discussion | 6-8 | 1,500-2,000 | 7% |
| Chapter 7: Conclusion | 2-3 | 500-800 | 3% |
| References | 3-5 | - | 4% |
| **总计** | **~100** | **~25,000** | 100% |

---

## 二、语言风格规范

### 2.1 学术英语特点

1. **正式性**: 避免口语化表达
   - ❌ "We think that LLMs are better..."
   - ✅ "We hypothesize that LLMs demonstrate superior performance..."

2. **精确性**: 使用具体数据支撑论点
   - ❌ "LLMs performed well..."
   - ✅ "DeepSeek V3.2 achieved a Sharpe ratio of 0.66, outperforming the best ML model (LSTM, 0.17) by 288%."

3. **客观性**: 避免主观判断
   - ❌ "This is an amazing result..."
   - ✅ "This result is statistically significant at the α=0.05 level."

### 2.2 时态规范

| 章节 | 推荐时态 | 示例 |
|------|---------|------|
| Abstract | Present / Past | "This thesis presents..." / "We conducted experiments..." |
| Introduction | Present | "Machine learning plays an important role..." |
| Related Work | Present Perfect / Past | "Previous studies have shown..." / "Zhang et al. (2023) proposed..." |
| Methodology | Past | "We implemented three baseline models..." |
| Experiments | Past | "We conducted backtests on two markets..." |
| Results | Past | "DeepSeek V3.2 achieved the highest Sharpe ratio..." |
| Discussion | Present | "These findings suggest that..." |
| Conclusion | Present | "Our work demonstrates that..." |

### 2.3 常用学术短语

#### 表达贡献
- "To the best of our knowledge, this is the first study to..."
- "Our work makes the following contributions..."
- "We propose a novel framework for..."

#### 表达结果
- "Our experimental results demonstrate that..."
- "The results indicate a significant improvement in..."
- "As shown in Table 1, ..."

#### 表达对比
- "Consistent with prior work [citation], we observe..."
- "In contrast to [citation], our approach..."
- "Unlike traditional methods, ..."

#### 表达限制
- "One limitation of this study is..."
- "Further research is needed to..."
- "Our findings should be interpreted with caution because..."

---

## 三、引用规范

### 3.1 引用格式

本论文采用 IEEE 格式：

```bibtex
@article{yang2023fingpt,
  title={FinGPT: Open-Source Financial Large Language Models},
  author={Yang, Honglin and Liu, Xiao and Wang, Zhiwei},
  journal={arXiv preprint arXiv:2306.06031},
  year={2023}
}
```

文中引用:
- 单篇: `\cite{yang2023fingpt}`
- 多篇: `\cite{yang2023fingpt, zhang2025alphaborgebench}`
- 页内: `(Yang et al., 2023)` 或 `Yang et al. (2023) showed that...`

### 3.2 引用原则

1. **关键概念必须引用**: 每个重要概念的首次提及需引用原始文献
2. **避免过度自引**: 自引比例控制在 10% 以内
3. **引用权威来源**: 优先引用顶级期刊/会议论文
4. **时效性**: 近 5 年文献占比 > 50%

---

## 四、图表规范

### 4.1 图表要求

| 项目 | 要求 |
|------|------|
| 分辨率 | 300 DPI |
| 格式 | PNG / PDF |
| 字体 | 与正文一致 |
| 编号 | 按章节编号 (Figure 1.1, Table 2.1) |

### 4.2 图表标题

- **图标题**: 在图下方
- **表标题**: 在表上方
- **格式**: 简洁、描述性

示例:
```
Figure 5.1: SHAP Feature Importance Analysis for CSI300 LightGBM Model

Table 3.1: Market State Classification and Signal Weights
```

### 4.3 图表引用

在正文中必须引用所有图表:
- "As shown in Figure 5.1, momentum indicators contribute most..."
- "Table 3.1 summarizes the market state classification rules."

---

## 五、写作技巧

### 5.1 段落结构

每个段落应遵循 C-E-E 结构:
1. **Claim**: 主题句，陈述核心观点
2. **Evidence**: 证据，数据或引用支撑
3. **Explanation**: 解释，说明证据如何支撑观点

示例:
```
[Claim] LLM-based strategies demonstrate superior risk-adjusted returns in emerging markets.
[Evidence] On CSI300, DeepSeek V3.2 achieved a Sharpe ratio of 0.66 compared to 0.17 for LSTM.
[Explanation] This suggests that the reasoning capability of LLMs provides advantages in markets
where textual information plays a significant role in price formation.
```

### 5.2 章节过渡

章节之间应有清晰的逻辑过渡:

```
// Chapter 2 结尾
"Despite significant progress in both ML and LLM approaches, systematic comparison remains lacking."

// Chapter 3 开头
"To address this gap, we propose the Dual-Track framework..."
```

### 5.3 学术论证

**假设-验证-结论** 模式:

```
[Hypothesis] We hypothesize that LLM strategies provide better risk control during black swan events.

[Verification] We analyze performance during the COVID-19 market crash (March 2020)...

[Conclusion] Contrary to our hypothesis, ML models showed lower maximum drawdown, suggesting...
```

---

## 六、检查清单

### 写作前
- [ ] 阅读完核心文献 (10+ 篇)
- [ ] 完成数据分析，准备好所有图表
- [ ] 明确每章的核心论点

### 写作中
- [ ] 每段有明确的主题句
- [ ] 所有论断有数据或引用支撑
- [ ] 图表编号和引用正确
- [ ] 参考文献格式统一

### 写作后
- [ ] 拼写检查 (Grammarly / LanguageTool)
- [ ] 语法检查
- [ ] 格式检查 (字体、行距、页边距)
- [ ] 引用完整性检查
- [ ] 图表清晰度检查

---

## 七、常用工具

### 写作工具
- **Overleaf**: 在线 LaTeX 编辑器
- **Grammarly**: 语法检查
- **Zotero**: 文献管理

### 数据可视化
- **Matplotlib**: Python 绑图
- **SHAP**: 特征归因
- **seaborn**: 统计可视化

### 版本控制
- **Git**: 论文版本管理
- **GitHub**: 备份与协作

---

## 八、参考文献写作资源

1. **Academic Phrasebank**: https://www.phrasebank.manchester.ac.uk/
2. **IEEE Citation Style**: https://ieeeauthorcenter.ieee.org/
3. **LaTeX Guide**: https://www.overleaf.com/learn

---

*最后更新: 2026-03-18*