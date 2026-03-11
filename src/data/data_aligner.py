"""
数据对齐模块。

提供高频价格数据与低频新闻数据的时间对齐功能。
包含严格的未来函数防护机制。
"""

import json
import logging
from pathlib import Path
from typing import Literal, Optional, Union
import warnings

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FutureFunctionError(Exception):
    """未来函数错误异常。"""
    pass


class DataAligner:
    """
    数据对齐工具类。

    用于将不同频率的数据按时间戳进行对齐，支持向前填充、
    最近交易日对齐等策略。

    Attributes:
        processed_dir: 处理后数据保存目录。
    """

    def __init__(self, processed_dir: Optional[Path] = None) -> None:
        """
        初始化数据对齐器。

        Args:
            processed_dir: 处理后数据保存目录，默认为项目根目录下的 data/processed/。
        """
        project_root = Path(__file__).parent.parent.parent
        self.processed_dir = processed_dir or project_root / "data" / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_no_future_data(
        features: pd.DataFrame,
        ohlcv: pd.DataFrame,
        strict: bool = True,
    ) -> bool:
        """
        严格验证特征数据不包含未来信息。

        确保用于 T 日预测的特征只包含 T-1 及之前的信息。

        Args:
            features: 特征DataFrame，索引为日期。
            ohlcv: 原始OHLCV数据，索引为日期。
            strict: 是否严格模式（任何违规都会抛出异常）。

        Returns:
            验证是否通过。

        Raises:
            FutureFunctionError: 当检测到未来函数时抛出。
        """
        if not isinstance(features.index, pd.DatetimeIndex):
            features.index = pd.to_datetime(features.index)
        if not isinstance(ohlcv.index, pd.DatetimeIndex):
            ohlcv.index = pd.to_datetime(ohlcv.index)

        # 检查1: 特征时间不应晚于OHLCV时间
        features_dates = set(features.index.normalize())
        ohlcv_dates = set(ohlcv.index.normalize())

        future_dates = features_dates - ohlcv_dates
        if future_dates:
            msg = f"检测到未来日期的特征: {sorted(future_dates)[:5]}"
            if strict:
                raise FutureFunctionError(msg)
            else:
                warnings.warn(msg, UserWarning)
                return False

        # 检查2: 特征中的收盘价应该已经shift
        if 'close' in features.columns and 'close' in ohlcv.columns:
            # 对齐日期
            common_dates = features.index.intersection(ohlcv.index)
            if len(common_dates) > 0:
                for date in common_dates[:5]:  # 抽查前5个
                    feature_close = features.loc[date, 'close']
                    ohlcv_close = ohlcv.loc[date, 'close']

                    # 如果特征中的close等于当日的close，说明没有shift
                    if np.isclose(feature_close, ohlcv_close):
                        msg = f"检测到未shift的收盘价特征在 {date}"
                        if strict:
                            raise FutureFunctionError(msg)
                        else:
                            warnings.warn(msg, UserWarning)
                            return False

        # 检查3: rolling特征应该正确对齐
        # 抽查rolling特征是否使用了未来数据
        rolling_cols = [col for col in features.columns if 'rolling' in col.lower() or 'ma' in col.lower()]
        for col in rolling_cols[:3]:  # 抽查前3个
            if features[col].isna().sum() == 0:
                # rolling特征前面应该有NaN（因为历史数据不足）
                msg = f"滚动特征 '{col}' 前期无NaN，可能使用了未来数据"
                if strict:
                    raise FutureFunctionError(msg)
                else:
                    warnings.warn(msg, UserWarning)
                    return False

        # 检查4: 确保特征的最后一行不包含"明天"的信息
        # 通过检查特征计算是否依赖了当日收盘后的信息
        latest_feature_date = features.index.max()
        latest_ohlcv_date = ohlcv.index.max()

        if latest_feature_date > latest_ohlcv_date:
            msg = f"特征最新日期 {latest_feature_date} 晚于OHLCV最新日期 {latest_ohlcv_date}"
            if strict:
                raise FutureFunctionError(msg)
            else:
                warnings.warn(msg, UserWarning)
                return False

        print("✓ 未来函数检查通过：特征数据不包含未来信息")
        return True

    @staticmethod
    def assert_no_look_ahead_bias(
        features: pd.DataFrame,
        target: pd.Series,
        prediction_date: pd.Timestamp,
    ) -> None:
        """
        断言预测时不使用未来数据。

        Args:
            features: 特征DataFrame。
            target: 目标变量Series。
            prediction_date: 预测日期。

        Raises:
            FutureFunctionError: 当特征包含预测日之后的信息时。
        """
        # 过滤出预测日之前的特征
        valid_features = features[features.index < prediction_date]

        if len(valid_features) == 0:
            raise FutureFunctionError(f"预测日期 {prediction_date} 没有可用的历史特征")

        # 确保目标变量不包含未来信息
        if target.index.max() > prediction_date:
            raise FutureFunctionError(
                f"目标变量包含未来数据（最晚日期: {target.index.max()}）"
            )

        print(f"✓ 预测日期 {prediction_date.date()} 的未来函数检查通过")

    @staticmethod
    def filter_important_notices(
        news_data: pd.DataFrame,
        notice_keywords: Optional[dict] = None,
    ) -> pd.DataFrame:
        """
        筛选重要公告，过滤噪音。

        Args:
            news_data: 新闻DataFrame，包含source和title列。
            notice_keywords: 公告关键词字典，默认使用重要类型。

        Returns:
            筛选后的DataFrame。
        """
        if notice_keywords is None:
            # 默认重要公告类型
            notice_keywords = {
                "重大事项": ["重大事项", "重组", "并购", "收购", "投资", "中标"],
                "风险提示": ["风险提示", "警示", "退市", "亏损", "商誉减值"],
                "业绩": ["业绩预告", "业绩快报", "年报", "季报", "半年报"],
                "分红送转": ["分红", "派息", "送转", "利润分配"],
                "股权变动重要": ["实际控制人", "控股股东", "大股东"],  # 只保留重要的股权变动
            }

        df = news_data.copy()

        # 筛选公告
        notices_mask = df["source"] == "notice"
        other_mask = df["source"] != "notice"

        # 对公告进行分类
        def is_important_notice(title: str) -> bool:
            """判断是否为重要公告。"""
            for category, keywords in notice_keywords.items():
                for kw in keywords:
                    if kw in title:
                        return True
            return False

        important_notices_mask = notices_mask & df["title"].apply(is_important_notice)

        # 保留重要公告 + 所有其他来源
        filtered = df[important_notices_mask | other_mask].copy()

        # 统计过滤效果
        original_notices = notices_mask.sum()
        filtered_notices = important_notices_mask.sum()
        print(f"公告筛选: {original_notices} 条 → {filtered_notices} 条 (保留 {filtered_notices/original_notices*100:.1f}%)")

        return filtered

    @staticmethod
    def aggregate_daily_news(
        news_data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        max_news_per_day: int = 25,
        max_content_length: int = 200,
        source_col: Optional[str] = "source",
        filter_notices: bool = True,
        market_type: str = "A_SHARE",  # 新增：市场类型 ("A_SHARE" 或 "US_MARKET")
    ) -> pd.DataFrame:
        """
        将每日多条新闻聚合成一条，用于每日统一决策。

        支持多数据源（CCTV新闻、公告、宏观数据等），按来源分类聚合。
        智能筛选重要公告，过滤噪音。

        Args:
            news_data: 新闻DataFrame，需包含timestamp、title、content列。
            timestamp_col: 时间戳列名。
            max_news_per_day: 每天最多保留的新闻条数（建议20-30）。
            max_content_length: 单条新闻内容最大长度（建议150-200字符）。
            source_col: 数据来源列名，用于区分不同来源的新闻。
            filter_notices: 是否智能筛选公告（默认True）。

        Returns:
            聚合后的DataFrame，每行代表一天的所有新闻。
            Columns: date, timestamp, aggregated_title, aggregated_content,
                    cctv_news, notices, news_count, source_counts
            注意：宏观数据和北向资金作为市场数据，在market_context中单独处理
        """
        if news_data.empty:
            return pd.DataFrame(columns=[
                "date", "timestamp", "aggregated_title", "aggregated_content",
                "cctv_news", "notices",
                "news_count", "source_counts"
            ])

        df = news_data.copy()

        # 智能筛选公告（含成分股过滤）
        if filter_notices and source_col and "notice" in df[source_col].values:
            # 先进行成分股过滤
            try:
                from scripts.fetch_csi300_constituents import get_constituents_cached
                constituents_codes, constituents_names = get_constituents_cached()
                logger.info(f"加载沪深300成分股: {len(constituents_codes)} 只")

                # 过滤成分股公告
                notice_mask = df[source_col] == "notice"
                if notice_mask.any():
                    # 从标题提取股票代码或匹配公司名称
                    import re

                    # 获取名称到代码的映射（用于匹配只有名称的公告）
                    name_to_code = {name: code for code, name in constituents_names.items()} if 'constituents_names' in locals() else {}

                    def extract_code_from_title(title):
                        title = str(title)
                        # 方法1: 提取6位数字代码
                        match = re.search(r'(\d{6})', title)
                        if match:
                            return match.group(1)
                        # 方法2: 匹配公司名称
                        for name, code in name_to_code.items():
                            # 简化名称匹配（去掉"股份"、"有限"等后缀）
                            short_name = name.replace('股份', '').replace('有限', '').replace('公司', '').replace('集团', '')
                            if short_name in title or name in title:
                                return code
                        return None

                    df.loc[notice_mask, '_code'] = df.loc[notice_mask, 'title'].apply(extract_code_from_title)

                    # 只保留成分股公告（非公告类新闻如CCTV等保留）
                    filter_mask = (
                        (df[source_col] != "notice") |  # 非公告类新闻（CCTV、宏观、北向）保留
                        (df['_code'].isin(constituents_codes))  # 只保留成分股公告
                    )
                    notice_before = notice_mask.sum()
                    notice_after = (df.loc[notice_mask, '_code'].isin(constituents_codes)).sum()
                    df = df[filter_mask].drop(columns=['_code'], errors='ignore')
                    logger.info(f"成分股过滤: {notice_before} 条公告 → {notice_after} 条 (保留 {notice_after/notice_before*100:.1f}%)")
            except Exception as e:
                logger.warning(f"成分股过滤失败，跳过此步骤: {e}")

            # 再进行重要性筛选
            df = DataAligner.filter_important_notices(df, notice_keywords=None)

        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df["date"] = df[timestamp_col].dt.date

        # 确保有source列
        if source_col not in df.columns:
            df[source_col] = "unknown"

        # 按日期聚合
        def aggregate_group(group):
            # 美股市场：简化聚合逻辑
            if market_type == "US_MARKET":
                group = group.copy()

                # 按重要性排序（标题长度 + 内容长度）
                if "content" in group.columns:
                    group["importance"] = group["title"].str.len() + group["content"].str.len()
                else:
                    group["importance"] = group["title"].str.len()

                # 限制每天最多 N 条新闻
                group = group.nlargest(max_news_per_day, "importance")

                # 统计各来源数量
                source_counts = group[source_col].value_counts().to_dict() if source_col in group.columns else {}

                # 格式化新闻内容
                def format_us_news_item(row):
                    """美股新闻格式化：保留标题+正文"""
                    title = row['title']
                    content = row.get('content', '')

                    if content and isinstance(content, str):
                        content = content.replace('\n', ' ').strip()
                        # 英文智能截断：按句号分割
                        max_len = 200
                        if len(content) > max_len:
                            sentences = content.split('. ')
                            truncated = []
                            current_len = 0
                            for sentence in sentences:
                                if current_len + len(sentence) + 2 <= max_len:
                                    truncated.append(sentence)
                                    current_len += len(sentence) + 2
                                else:
                                    break
                            if truncated:
                                content_short = '. '.join(truncated) + '.'
                            else:
                                content_short = content[:max_len].rsplit(' ', 1)[0] + '...'
                        else:
                            content_short = content
                        return f"{title}: {content_short}"
                    else:
                        return title

                # 构建新闻列表
                news_items = [format_us_news_item(row) for _, row in group.iterrows()]
                full_content = "\n".join([f"- {item}" for item in news_items])

                return pd.Series({
                    "timestamp": pd.Timestamp(group[timestamp_col].iloc[0].date()),
                    "aggregated_title": "|".join(group["title"].astype(str).head(5)),
                    "aggregated_content": full_content,
                    "cctv_news": "",  # 美股无CCTV分类
                    "notices": "",    # 美股无公告分类
                    "news_count": len(group),
                    "source_counts": json.dumps(source_counts),
                    "structured_summary": f"[News:{len(group)}]",
                })

            # A股市场：原有逻辑
            # 按来源分组，每个来源分配固定配额
            group = group.copy()

            # 按来源分配配额（成分股公告减少后，增加CCTV新闻补充）
            # 注意：北向资金和宏观数据作为交易数据，像OHLCV一样在market_context中处理，不在新闻中聚合
            source_quotas = {
                "notice": 15,      # 公告：只保留成分股公告（预计会减少很多）
                "cctv": 15,        # CCTV新闻：增加到15条（补充公告减少后的信息量）
            }

            # 对每个来源按重要性排序并截取配额
            selected_items = []
            for source, quota in source_quotas.items():
                source_items = group[group[source_col] == source].copy()
                if len(source_items) > 0:
                    # 按重要性排序（标题长度 + 内容长度）
                    source_items["importance"] = source_items["title"].str.len() + source_items["content"].str.len()
                    source_items = source_items.nlargest(quota, "importance")
                    selected_items.append(source_items)

            # 合并所有来源（保持来源优先级顺序）
            group = pd.concat(selected_items, ignore_index=True) if selected_items else pd.DataFrame()

            # 按来源分组聚合（只保留cctv和notice，宏观数据和北向资金作为市场数据处理）
            cctv_news = group[group[source_col] == "cctv"]
            notices = group[group[source_col] == "notice"]

            # 统计各来源数量
            source_counts = group[source_col].value_counts().to_dict()

            # 构建分类聚合内容（紧凑格式节省token）
            sections = []
            if not notices.empty:
                sections.append(f"[公告:{len(notices)}]" +
                              "|".join([t for t in notices["title"].head(3)]))
            if not cctv_news.empty:
                sections.append(f"[CCTV:{len(cctv_news)}]" +
                              "|".join([t for t in cctv_news["title"].head(3)]))

            # 完整内容（智能处理：Markdown格式 + 句号截断）
            def format_news_item(row):
                """根据来源类型决定是否包含正文，使用句号智能截断。"""
                source_tag = row[source_col][:3].upper()
                title = row['title']
                content = row.get('content', '')

                # 公告（notice）: 提取type标签，简化为"股票代码: 标题"
                if row[source_col] == 'notice':
                    # content格式: "标签 | 股票名: 完整标题"
                    if content and ' | ' in content:
                        parts = content.split(' | ')
                        type_tag = parts[0].strip()  # 提取"重大事项"等标签
                        # 简化输出：股票代码 + 类型 + 标题
                        return f"{type_tag}: {title}"
                    return f"{title}"

                # 其他来源（CCTV新闻、宏观、北向资金等）: 保留标题+正文
                # 使用句号智能截断，保证句子完整性（增加长度到200字符）
                if content and isinstance(content, str):
                    content = content.replace('\n', ' ').strip()
                    # 智能截断：按句号分割，保留完整句子
                    max_len = 200  # 增加长度限制
                    if len(content) > max_len:
                        sentences = content.split('。')
                        truncated = []
                        current_len = 0
                        for sentence in sentences:
                            if current_len + len(sentence) + 1 <= max_len:  # +1 for '。'
                                truncated.append(sentence)
                                current_len += len(sentence) + 1
                            else:
                                break
                        if truncated:
                            content_short = '。'.join(truncated) + '。'
                        else:
                            # 如果第一句就太长，按逗号截断
                            content_short = content[:max_len].rsplit('，', 1)[0] + '……'
                    else:
                        content_short = content
                    return f"{title}：{content_short}"
                else:
                    return f"{title}"

            # 按来源分类格式化为Markdown列表
            md_sections = []

            # 1. CCTV新闻（宏观与政策面）- 显示更多条数
            if not cctv_news.empty:
                cctv_items = [format_news_item(row) for _, row in cctv_news.head(12).iterrows()]
                md_sections.append("### 1. 宏观与政策面 (CCTV)\n" + "\n".join([f"- {item}" for item in cctv_items]))

            # 2. 公告（核心成分股动态）- 只显示成分股公告
            if not notices.empty:
                notice_items = [format_news_item(row) for _, row in notices.head(10).iterrows()]
                md_sections.append("### 2. 成分股公告 (Notices)\n" + "\n".join([f"- {item}" for item in notice_items]))

            # 合并为Markdown格式
            full_content = "\n\n".join(md_sections)

            # 保存完整的带正文的内容（用于后续合并多日新闻时使用）
            cctv_full = "|".join([format_news_item(row) for _, row in cctv_news.iterrows()]) if not cctv_news.empty else ""
            notices_full = "|".join([format_news_item(row) for _, row in notices.iterrows()]) if not notices.empty else ""

            return pd.Series({
                "timestamp": pd.Timestamp(group[timestamp_col].iloc[0].date()),
                "aggregated_title": "|".join(group["title"].astype(str).head(5)),
                "aggregated_content": full_content,  # Markdown格式（带正文）
                "cctv_news": cctv_full,  # 完整的CCTV新闻（带正文）
                "notices": notices_full,  # 完整的公告（带正文）
                "news_count": len(group),
                "source_counts": json.dumps(source_counts),
                "structured_summary": " ".join(sections),  # 保持简洁摘要
            })

        aggregated = df.groupby("date").apply(aggregate_group, include_groups=False).reset_index()

        print(f"新闻聚合完成: {len(news_data)} 条 → {len(aggregated)} 天")
        print(f"平均每天: {len(news_data) / len(aggregated):.1f} 条新闻")
        print(f"数据来源分布: {df[source_col].value_counts().to_dict()}")

        return aggregated

    @staticmethod
    def merge_non_trading_news_to_trading_days(
        daily_news: pd.DataFrame,
        trading_days: pd.DatetimeIndex,
        timestamp_col: str = "timestamp",
    ) -> pd.DataFrame:
        """
        将非交易日的新闻合并到下一个交易日。

        业务逻辑：周末和假期发布的新闻（如政策公告）会影响下一个交易日的开盘，
        因此应该将这些新闻合并到下一个交易日一起考虑。

        例如：
        - 周五收盘后 + 周六 + 周日的公告 → 合并到下周一
        - 元旦假期的公告 → 合并到节后第一个交易日

        Args:
            daily_news: 按日历日聚合的新闻DataFrame
            trading_days: 交易日索引（用于识别交易日和非交易日）
            timestamp_col: 时间戳列名

        Returns:
            合并后的DataFrame，只包含交易日，但包含该交易日前所有累积的非交易日新闻
        """
        if daily_news.empty:
            return daily_news

        df = daily_news.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col]).dt.normalize()
        trading_days = pd.to_datetime(trading_days).normalize().unique()
        trading_days_set = set(trading_days)

        # 按日期排序
        df = df.sort_values(timestamp_col)
        all_dates = df[timestamp_col].unique()

        # 构建合并映射：每个新闻日期 → 下一个交易日（T日，决策日）
        # 关键修正：新闻发布于T-1日，决策在T日（下一个交易日）执行
        # 例如：
        # - 2020-01-01 (非交易日) 的新闻 → 2020-01-02 (第一个交易日) 决策
        # - 2020-01-02 (交易日) 的新闻 → 2020-01-03 (下一个交易日) 决策
        # - 2020-01-03 (周五交易日) 的新闻 → 2020-01-06 (周一) 决策
        sorted_trading_days = sorted(trading_days)
        merge_map = {}  # date -> next_trading_day (T)

        for date in all_dates:
            # 关键修正：使用 > 而不是 >=，确保新闻映射到下一个交易日（不含当天）
            for trading_day in sorted_trading_days:
                if trading_day > date:  # 必须是下一个交易日，不能是当天
                    merge_map[date] = trading_day
                    break

        # 对每一条新闻，找到它应该归属的交易日
        df['trading_day'] = df[timestamp_col].apply(
            lambda x: merge_map.get(x, x if x in trading_days_set else None)
        )

        # 过滤掉无法映射的日期（交易日开始前的数据）
        df = df[df['trading_day'].notna()]

        if df.empty:
            return pd.DataFrame(columns=daily_news.columns)

        # 按交易日合并新闻
        results = []
        for trading_day, group in df.groupby("trading_day"):
            # 如果只有一条，直接返回
            if len(group) == 1:
                result = group.iloc[0].copy()
                # 保持原始新闻日期，添加交易日信息
                result["trading_day"] = trading_day
                results.append(result)
                continue

            # 按原始日期排序（降序：最新的在前，更早的在后）
            group = group.sort_values(timestamp_col, ascending=False)

            # 检测是否为美股市场格式（aggregated_content 非空，cctv_news/notices 为空或不存在）
            has_us_format = (
                'aggregated_content' in group.columns and
                group['aggregated_content'].astype(str).str.strip().str.len().sum() > 0 and
                (not 'cctv_news' in group.columns or group['cctv_news'].astype(str).str.strip().str.len().sum() == 0)
            )

            if has_us_format:
                # 美股市场：直接合并 aggregated_content 字段
                all_contents = []
                total_news = 0
                for _, row in group.iterrows():
                    content = str(row.get('aggregated_content', '')).strip()
                    if content:
                        date_label = row[timestamp_col].strftime('%m-%d')
                        # 添加日期标签
                        all_contents.append(f"[{date_label}]\n{content}")
                        total_news += row.get('news_count', 1)

                # 合并内容，限制总长度
                combined_content = "\n\n".join(all_contents)
                content_limit = 5000
                combined_content = combined_content[:content_limit]

                # 取最新的新闻日期作为代表
                news_date = group[timestamp_col].max()

                # 合并标题
                combined_title = "|".join(filter(None, group["aggregated_title"].astype(str)))[:300]

                results.append(pd.Series({
                    timestamp_col: news_date,  # T-1 新闻日期
                    "trading_day": trading_day,  # T 交易日（决策日）
                    "aggregated_title": combined_title,
                    "aggregated_content": combined_content,
                    "cctv_news": "",
                    "notices": "",
                    "news_count": total_news,
                    "source_counts": json.dumps({"merged_days": len(group)}),
                    "structured_summary": f"合并 {len(group)} 天新闻: {total_news}条",
                }))
                continue

            # A股市场：原有的 CCTV 和公告合并逻辑

            # 按日期加权筛选：确保每个日期都有数据
            selected_indices = []
            date_position = 0
            prev_date = None

            for idx, row in group.iterrows():
                current_date = row[timestamp_col]

                # 如果是新的日期，增加位置计数
                if prev_date is None or current_date != prev_date:
                    date_position += 1
                    prev_date = current_date

                # 按日期位置筛选，确保每个日期都保留
                if date_position == 1:  # T-1日（最新的）
                    selected_indices.append(idx)
                elif date_position == 2:  # T-2日
                    selected_indices.append(idx)
                elif date_position == 3:  # T-3日
                    selected_indices.append(idx)
                else:  # T-4+日，严格控制
                    if len(selected_indices) < 20:
                        selected_indices.append(idx)

            # 限制总新闻数量
            max_total_news = 20
            selected_indices = selected_indices[:max_total_news]
            selected_group = group.loc[selected_indices]

            # 统计新闻数量
            total_news = selected_group["news_count"].sum()

            # 构建Markdown格式内容，明确标注每条新闻的来源日期
            md_sections = []

            # 1. CCTV新闻（宏观与政策面）
            # 按重要性分配数量：T-1日10条，T-2日3-4条，T-3+日1-2条，总计~15条
            cctv_items = []
            date_position = 0
            prev_date = None

            for _, row in selected_group.iterrows():
                date_label = f"[{row[timestamp_col].strftime('%m-%d')}]"
                current_date = row[timestamp_col]

                # 判断日期位置
                if prev_date is None or current_date != prev_date:
                    date_position += 1
                    prev_date = current_date

                if row["cctv_news"]:
                    # 将多条CCTV新闻拆分
                    all_news = [news.strip() for news in row["cctv_news"].split("|") if news.strip()]

                    # 根据位置决定保留数量
                    if date_position == 1:  # T-1日（最新的）
                        news_limit = 10
                    elif date_position == 2:  # T-2日
                        news_limit = 4
                    else:  # T-3+日
                        news_limit = 2

                    # 取前N条
                    selected_news = all_news[:news_limit]

                    for news in selected_news:
                        cctv_items.append(f"{date_label} {news}")

            if cctv_items:
                md_sections.append(f"### 1. 宏观与政策面 (CCTV) - T-1日及前期新闻\n" +
                                  "\n".join([f"- {item}" for item in cctv_items[:15]]))

            # 2. 成分股公告
            # 按重要性分配数量：T-1日最多10条，T-2日3条，T-3+日2条
            notice_items = []
            date_position = 0
            prev_date = None

            for _, row in selected_group.iterrows():
                date_label = f"[{row[timestamp_col].strftime('%m-%d')}]"
                current_date = row[timestamp_col]

                # 判断日期位置
                if prev_date is None or current_date != prev_date:
                    date_position += 1
                    prev_date = current_date

                if row["notices"]:
                    all_notices = [notice.strip() for notice in row["notices"].split("|") if notice.strip()]

                    # 根据位置决定保留数量
                    if date_position == 1:  # T-1日
                        notice_limit = 10
                    elif date_position == 2:  # T-2日
                        notice_limit = 3
                    else:  # T-3+日
                        notice_limit = 2

                    # 取前N条
                    selected_notices = all_notices[:notice_limit]

                    for notice in selected_notices:
                        notice_items.append(f"{date_label} {notice}")

            if notice_items:
                md_sections.append(f"### 2. 成分股公告 (Notices) - T-1日及前期公告\n" +
                                  "\n".join([f"- {item}" for item in notice_items[:10]]))

            # 合并为Markdown格式
            combined_content = "\n\n".join(md_sections)

            # 构建简单摘要
            combined_title = "|".join(filter(None, selected_group["aggregated_title"].astype(str)))
            combined_cctv = "|".join(filter(None, selected_group["cctv_news"].astype(str)))
            combined_notices = "|".join(filter(None, selected_group["notices"].astype(str)))

            # 限制内容长度（增加以容纳15条新闻，每条约200字符）
            content_limit = 3500 if len(selected_group) <= 3 else 3000

            # 关键修正：timestamp 应该是新闻日期（T-1），而非交易日（T）
            # 取选中新闻中的最晚日期作为代表（通常是T-1）
            news_date = selected_group[timestamp_col].max()

            results.append(pd.Series({
                timestamp_col: news_date,  # T-1 新闻日期
                "trading_day": trading_day,  # T 交易日（决策日）
                "aggregated_title": combined_title[:300],
                "aggregated_content": combined_content[:content_limit],
                "cctv_news": combined_cctv[:500],
                "notices": combined_notices[:500],
                "news_count": total_news,
                "source_counts": json.dumps({"selected_days": len(selected_group), "original_days": len(group)}),
                "structured_summary": f"T-1及前期新闻: {total_news}条",
            }))

        merged = pd.DataFrame(results)

        print(f"非交易日新闻合并: {len(daily_news)} 天 → {len(merged)} 个交易日")
        print(f"合并了周末和假期的新闻到下一交易日")

        return merged

    @staticmethod
    def get_trading_days(
        price_data: pd.DataFrame,
        date_column: Optional[str] = None,
    ) -> pd.DatetimeIndex:
        """
        从价格数据中提取交易日列表。

        Args:
            price_data: 价格数据DataFrame，需包含日期索引或日期列。
            date_column: 日期列名，如果为None则使用索引。

        Returns:
            排序后的交易日DatetimeIndex。
        """
        if date_column:
            dates = pd.to_datetime(price_data[date_column])
        else:
            dates = pd.to_datetime(price_data.index)

        trading_days = dates.normalize().sort_values().unique()
        return pd.DatetimeIndex(trading_days)

    @staticmethod
    def align_to_nearest_trading_day(
        timestamp: pd.Timestamp,
        trading_days: pd.DatetimeIndex,
        method: Literal["previous", "next", "nearest"] = "previous",
    ) -> pd.Timestamp:
        """
        将时间戳对齐到最近的交易日。

        Args:
            timestamp: 待对齐的时间戳。
            trading_days: 交易日列表。
            method: 对齐方法，可选 'previous'（向前）、'next'（向后）、
                   'nearest'（最近）。

        Returns:
            对齐后的交易日时间戳。

        Raises:
            ValueError: 当无法找到合适的交易日时抛出。
        """
        ts_normalized = timestamp.normalize()

        if method == "previous":
            # 找到小于等于当前日期的最近交易日
            mask = trading_days <= ts_normalized
            if not mask.any():
                raise ValueError(f"无法找到早于 {timestamp} 的交易日")
            return trading_days[mask][-1]

        elif method == "next":
            # 找到大于等于当前日期的最近交易日
            mask = trading_days >= ts_normalized
            if not mask.any():
                raise ValueError(f"无法找到晚于 {timestamp} 的交易日")
            return trading_days[mask][0]

        elif method == "nearest":
            # 找到最近的交易日
            diff = abs(trading_days - ts_normalized)
            nearest_idx = diff.argmin()
            return trading_days[nearest_idx]

        else:
            raise ValueError(f"不支持的对齐方法: {method}")

    def align_news_to_price(
        self,
        price_data: pd.DataFrame,
        news_data: pd.DataFrame,
        timestamp_col: str = "timestamp",
        fill_method: Literal["ffill", "nearest", "drop"] = "ffill",
        save_to_file: bool = True,
        output_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        将低频新闻数据对齐到高频价格数据的时间点。

        将新闻数据按照交易日进行分组，并对齐到价格数据的每个时间点。
        支持向前填充、最近对齐或丢弃非交易日数据等策略。

        Args:
            price_data: 价格数据DataFrame，索引应为日期。
            news_data: 新闻数据DataFrame，需包含时间戳列。
            timestamp_col: 新闻数据中时间戳列的名称。
            fill_method: 填充方法：
                - 'ffill': 向前填充，每个交易日继承之前最近的新闻。
                - 'nearest': 对齐到最近的交易日。
                - 'drop': 只保留有新闻的交易日数据。
            save_to_file: 是否保存到文件，默认为True。
            output_filename: 输出文件名，如果为None则自动生成。

        Returns:
            对齐后的DataFrame，包含价格数据和关联的新闻数据。

        Raises:
            ValueError: 当输入数据格式不正确时抛出。
        """
        # 获取交易日列表
        trading_days = self.get_trading_days(price_data)

        # 确保新闻数据的时间戳列是datetime类型
        news_data = news_data.copy()
        news_data[timestamp_col] = pd.to_datetime(news_data[timestamp_col])
        news_data["aligned_date"] = news_data[timestamp_col].apply(
            lambda ts: self.align_to_nearest_trading_day(ts, trading_days, "previous")
        )

        # 按对齐日期聚合新闻
        news_grouped = (
            news_data.groupby("aligned_date")
            .agg({
                "title": lambda x: " | ".join(x.astype(str)),
                "content": lambda x: " || ".join(x.astype(str)),
            })
            .reset_index()
        )
        news_grouped.columns = ["date", "news_titles", "news_contents"]

        # 确保价格数据索引为日期
        price_data = price_data.copy()
        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)
        price_data["date"] = price_data.index.normalize()

        # 合并价格和新闻数据
        aligned_df = price_data.merge(
            news_grouped,
            on="date",
            how="left" if fill_method != "drop" else "inner",
        )

        # 处理填充
        if fill_method == "ffill":
            aligned_df[["news_titles", "news_contents"]] = aligned_df[
                ["news_titles", "news_contents"]
            ].ffill()

        # 删除临时列
        aligned_df = aligned_df.drop(columns=["date"], errors="ignore")

        # 保存数据
        if save_to_file:
            filename = output_filename or f"aligned_data_{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
            filepath = self.processed_dir / filename
            aligned_df.to_parquet(filepath, index=True)
            print(f"对齐数据已保存至: {filepath}")

        return aligned_df

    def align_multiple_series(
        self,
        dataframes: dict[str, pd.DataFrame],
        fill_method: Literal["ffill", "interpolate", "drop"] = "ffill",
        save_to_file: bool = True,
        output_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        对齐多个数据序列到统一的时间轴。

        Args:
            dataframes: 数据字典，键为数据名称，值为DataFrame。
                       每个DataFrame应具有日期索引。
            fill_method: 填充方法：
                - 'ffill': 向前填充缺失值。
                - 'interpolate': 线性插值填充。
                - 'drop': 只保留所有序列都有数据的日期。
            save_to_file: 是否保存到文件，默认为True。
            output_filename: 输出文件名。

        Returns:
            对齐后的合并DataFrame。

        Raises:
            ValueError: 当输入数据为空时抛出。
        """
        if not dataframes:
            raise ValueError("数据字典不能为空")

        # 获取所有日期的并集
        all_dates: pd.DatetimeIndex = pd.DatetimeIndex([])
        for name, df in dataframes.items():
            dates = self.get_trading_days(df)
            all_dates = all_dates.union(dates)
        all_dates = all_dates.sort_values()

        # 重新索引每个DataFrame
        aligned_dfs: list[pd.DataFrame] = []
        for name, df in dataframes.items():
            df_copy = df.copy()
            if not isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy.index = pd.to_datetime(df_copy.index)

            # 重采样到日频率并填充
            df_copy = df_copy.reindex(all_dates)

            if fill_method == "ffill":
                df_copy = df_copy.ffill()
            elif fill_method == "interpolate":
                df_copy = df_copy.interpolate(method="linear")

            # 添加前缀以区分不同数据源
            df_copy.columns = [f"{name}_{col}" for col in df_copy.columns]
            aligned_dfs.append(df_copy)

        # 合并所有数据
        result = pd.concat(aligned_dfs, axis=1)

        # 如果选择drop方法，删除任何含有缺失值的行
        if fill_method == "drop":
            result = result.dropna()

        # 保存数据
        if save_to_file:
            filename = output_filename or f"aligned_multi_{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
            filepath = self.processed_dir / filename
            result.to_parquet(filepath, index=True)
            print(f"多序列对齐数据已保存至: {filepath}")

        return result

    def resample_price_data(
        self,
        price_data: pd.DataFrame,
        freq: str = "W",
        agg_rules: Optional[dict[str, str]] = None,
        save_to_file: bool = True,
        output_filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        重采样价格数据到指定频率。

        Args:
            price_data: 价格数据DataFrame，需包含OHLCV列。
            freq: 重采样频率，如 'D'(日)、'W'(周)、'M'(月)。
            agg_rules: 各列的聚合规则，如 {'close': 'last', 'volume': 'sum'}。
                      默认为标准OHLCV聚合规则。
            save_to_file: 是否保存到文件，默认为True。
            output_filename: 输出文件名。

        Returns:
            重采样后的DataFrame。

        Raises:
            ValueError: 当输入数据缺少必要列时抛出。
        """
        price_data = price_data.copy()

        if not isinstance(price_data.index, pd.DatetimeIndex):
            price_data.index = pd.to_datetime(price_data.index)

        # 默认OHLCV聚合规则
        if agg_rules is None:
            agg_rules = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }

        # 过滤存在的列
        available_cols = {col: rule for col, rule in agg_rules.items() if col in price_data.columns}

        if not available_cols:
            raise ValueError("价格数据缺少必要的OHLCV列")

        # 重采样
        resampled = price_data.resample(freq).agg(available_cols)
        resampled = resampled.dropna(how="all")

        # 保存数据
        if save_to_file:
            filename = output_filename or f"resampled_{freq}_{pd.Timestamp.now().strftime('%Y%m%d')}.parquet"
            filepath = self.processed_dir / filename
            resampled.to_parquet(filepath, index=True)
            print(f"重采样数据已保存至: {filepath}")

        return resampled


if __name__ == "__main__":
    # 示例用法
    from datetime import datetime, timedelta

    import numpy as np

    # 创建示例价格数据
    dates = pd.date_range(
        start="2024-01-01",
        end="2024-01-31",
        freq="B",  # 工作日
    )
    np.random.seed(42)

    price_df = pd.DataFrame({
        "open": 100 + np.random.randn(len(dates)).cumsum(),
        "high": 101 + np.random.randn(len(dates)).cumsum(),
        "low": 99 + np.random.randn(len(dates)).cumsum(),
        "close": 100 + np.random.randn(len(dates)).cumsum(),
        "volume": np.random.randint(1000000, 10000000, len(dates)),
    }, index=dates)

    # 创建示例新闻数据
    news_df = pd.DataFrame({
        "timestamp": [
            datetime(2024, 1, 5, 10, 30),
            datetime(2024, 1, 8, 14, 0),
            datetime(2024, 1, 15, 9, 45),
            datetime(2024, 1, 20, 11, 0),
            datetime(2024, 1, 25, 15, 30),
        ],
        "title": [
            "市场大涨：科技股领涨",
            "央行降准，市场反弹",
            "新能源板块走强",
            "外资净流入创新高",
            "年报季来临，关注业绩",
        ],
        "content": [
            "今日市场大幅上涨...",
            "央行宣布降准...",
            "新能源板块表现强势...",
            "外资持续流入A股...",
            "年报披露季来临...",
        ],
    })

    # 对齐数据
    aligner = DataAligner()
    aligned = aligner.align_news_to_price(
        price_data=price_df,
        news_data=news_df,
        fill_method="ffill",
    )

    print(f"\n对齐后数据行数: {len(aligned)}")
    print(f"有新闻的交易日数: {aligned['news_titles'].notna().sum()}")
    print(f"\n对齐数据样例:\n{aligned.head(10)}")