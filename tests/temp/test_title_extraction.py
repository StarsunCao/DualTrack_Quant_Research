"""测试优化后的新闻格式（正文缩短+增加条数）。"""

import pandas as pd
from src.data.data_aligner import DataAligner

# 创建测试数据（模拟真实的数据格式）
test_news = pd.DataFrame([
    # 公告：content字段包含type标签，需要提取
    {
        "timestamp": "2020-01-02",
        "source": "notice",
        "title": "荃银高科:关于向银行申请并购贷款的公告",
        "content": "重大事项 | 荃银高科: 荃银高科:关于向银行申请并购贷款的公告"
    },
    {
        "timestamp": "2020-01-02",
        "source": "notice",
        "title": "科锐国际:关于控股股东部分股份解除质押的公告",
        "content": "重大事项 | 科锐国际: 科锐国际:关于控股股东部分股份解除质押的公告"
    },
    {
        "timestamp": "2020-01-02",
        "source": "notice",
        "title": "金龙羽:关于项目中标公告",
        "content": "重大事项 | 金龙羽: 金龙羽:关于项目中标公告"
    },
    {
        "timestamp": "2020-01-02",
        "source": "notice",
        "title": "诚迈科技:风险提示及澄清公告",
        "content": "风险提示 | 诚迈科技: 诚迈科技:风险提示及澄清公告"
    },
    # CCTV新闻：长正文，测试缩短效果
    {
        "timestamp": "2020-01-02",
        "source": "cctv",
        "title": "习近平主持召开中央财经委员会第六次会议",
        "content": "中共中央总书记、国家主席、中央军委主席、中央财经委员会主任习近平1月3日下午主持召开中央财经委员会第六次会议，研究黄河流域生态保护和高质量发展，大力推动成渝地区双城经济圈建设。习近平在会上发表重要讲话强调，黄河流域生态保护和高质量发展是国家战略，要牢固树立大局意识，加强统筹协调，确保黄河流域生态保护和高质量发展取得实效。"
    },
    {
        "timestamp": "2020-01-02",
        "source": "cctv",
        "title": "跑出新时代的中国风采",
        "content": "的一天，新的一年，新的起点，新的希望，我们万众一心加油干。让我们勠力同心，朝着春天的梦想勇毅笃行。新的一年，我们要继续奋斗，勇往直前，为实现中华民族伟大复兴的中国梦而不懈努力。"
    },
    # 北向资金
    {
        "timestamp": "2020-01-02",
        "source": "northbound",
        "title": "北向资金净流入50亿",
        "content": "北向资金今日净流入50亿元，连续5日净流入"
    },
])

# 使用聚合函数
print("=" * 80)
print("测试优化方案：正文缩短到100字符 + 增加新闻条数")
print("=" * 80)

result = DataAligner.aggregate_daily_news(
    test_news,
    max_news_per_day=30,  # 增加到30条
    max_content_length=100,  # 缩短正文到100字符
    filter_notices=False
)

print("\n聚合结果:")
print(f"日期数: {len(result)}")
print(f"\naggregated_content 详细内容:")
content = result['aggregated_content'].iloc[0]
print(content)

print("\n" + "=" * 80)
print("优化效果分析:")
print(f"总长度: {len(content)} 字符")
print(f"新闻条数: {content.count('|')+1} 条")
print(f"平均每条: {len(content)/(content.count('|')+1):.0f} 字符")

# 分析各来源
items = content.split('|')
not_items = [item for item in items if '[NOT]' in item]
cct_items = [item for item in items if '[CCT]' in item]
nor_items = [item for item in items if '[NOR]' in item]

print(f"\n按来源统计:")
print(f"  公告: {len(not_items)}条, 平均{sum(len(i) for i in not_items)/len(not_items):.0f}字符")
if cct_items:
    print(f"  CCTV: {len(cct_items)}条, 平均{sum(len(i) for i in cct_items)/len(cct_items):.0f}字符")
    # 显示CCTV示例
    print(f"    正文示例: {cct_items[0]}")
if nor_items:
    print(f"  北向: {len(nor_items)}条, 平均{sum(len(i) for i in nor_items)/len(nor_items):.0f}字符")

print("=" * 80)
print("\n验证要点:")
print("1. CCTV正文缩短到~100字符（之前150-180字符）")
print("2. 核心信息完整（时间、事件、主题）")
print("3. 公告数量增加到20条")
print("4. 总长度保持~1800字符")
print("=" * 80)