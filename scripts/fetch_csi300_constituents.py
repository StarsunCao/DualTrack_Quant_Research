"""
获取沪深300成分股列表。

由于akshare只提供当前成分股，暂时使用当前成分股作为近似。
沪深300每半年调整一次（6月、12月），每次调整比例不超过10%，
因此使用当前成分股过滤历史公告的偏差在可接受范围内。

TODO: 后续可通过以下方式获取历史成分股：
1. 中证指数公司官网爬取历史调整公告
2. 使用Wind/Choice等专业数据接口
3. 从公告数据中提取沪深300调整公告
"""

import pandas as pd
import akshare as ak
from pathlib import Path
from typing import Set, Optional
import logging

logger = logging.getLogger(__name__)


def fetch_csi300_constituents(
    save_path: Optional[Path] = None,
    include_names: bool = True
) -> Set[str]:
    """
    获取沪深300当前成分股代码列表。

    Args:
        save_path: 保存路径，如果为None则不保存
        include_names: 是否同时返回股票名称字典

    Returns:
        成分股代码集合，如 {'000001', '000002', ...}
        如果 include_names=True，则返回 (代码集合, {代码: 名称}字典)
    """
    try:
        logger.info("正在获取沪深300成分股列表...")

        # 使用akshare获取成分股
        df = ak.index_stock_cons_csindex(symbol='000300')

        # 提取成分股代码（6位数字，如 '000001'）
        code_col = '成分券代码'
        if code_col not in df.columns:
            # 尝试其他可能的列名
            for col in ['股票代码', 'code', 'symbol']:
                if col in df.columns:
                    code_col = col
                    break

        codes = set(df[code_col].astype(str).str.zfill(6))

        logger.info(f"成功获取 {len(codes)} 只沪深300成分股")

        # 保存到文件
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            result_df = df[['日期', code_col, '成分券名称']].copy()
            result_df.columns = ['date', 'code', 'name']
            result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
            logger.info(f"成分股列表已保存至: {save_path}")

        if include_names:
            name_dict = dict(zip(df[code_col].astype(str).str.zfill(6), df['成分券名称']))
            return codes, name_dict
        else:
            return codes

    except Exception as e:
        logger.error(f"获取成分股列表失败: {e}")
        # 返回空集合，避免后续处理失败
        if include_names:
            return set(), {}
        else:
            return set()


def load_constituents_from_file(file_path: Path) -> Set[str]:
    """
    从文件加载成分股列表。

    Args:
        file_path: 成分股CSV文件路径

    Returns:
        成分股代码集合
    """
    file_path = Path(file_path)

    if not file_path.exists():
        logger.warning(f"成分股文件不存在: {file_path}")
        return set()

    df = pd.read_csv(file_path, dtype={'code': str})
    codes = set(df['code'].str.zfill(6))

    logger.info(f"从文件加载 {len(codes)} 只成分股")
    return codes


def filter_notices_by_constituents(
    notices: pd.DataFrame,
    constituents_codes: Set[str],
    code_col: str = '股票代码',
    keep_no_code: bool = False
) -> pd.DataFrame:
    """
    根据成分股列表过滤公告。

    Args:
        notices: 公告DataFrame
        constituents_codes: 成分股代码集合
        code_col: 股票代码列名
        keep_no_code: 是否保留无法识别代码的公告（如系统性的宏观公告）

    Returns:
        过滤后的公告DataFrame
    """
    if notices.empty:
        return notices

    df = notices.copy()

    # 从title或content中提取股票代码
    # 公告标题通常包含股票代码，如 "000001 平安银行: 关于XXX的公告"
    def extract_code(row):
        title = str(row.get('title', ''))
        content = str(row.get('content', ''))

        # 尝试从标题提取代码（前6位数字）
        import re
        code_match = re.search(r'(\d{6})', title)
        if code_match:
            return code_match.group(1)

        # 尝试从内容提取
        code_match = re.search(r'(\d{6})', content)
        if code_match:
            return code_match.group(1)

        return None

    # 提取代码
    df['_extracted_code'] = df.apply(extract_code, axis=1)

    # 过滤：保留成分股公告
    if keep_no_code:
        # 保留成分股 + 无法识别代码的公告
        mask = (df['_extracted_code'].isin(constituents_codes)) | (df['_extracted_code'].isna())
    else:
        # 只保留成分股公告
        mask = df['_extracted_code'].isin(constituents_codes)

    filtered = df[mask].drop(columns=['_extracted_code'])

    logger.info(f"公告过滤: {len(notices)} 条 → {len(filtered)} 条 (保留 {len(filtered)/len(notices)*100:.1f}%)")

    return filtered


# 用于缓存成分股列表的全局变量
_CACHED_CONSTITUENTS = None
_CACHED_NAMES = None


def get_constituents_cached(
    cache_file: Optional[Path] = Path("data/processed/csi300_constituents.csv"),
    force_reload: bool = False
) -> tuple[Set[str], dict]:
    """
    获取成分股列表（带缓存）。

    优先从缓存文件加载，如果不存在则从API获取并缓存。

    Args:
        cache_file: 缓存文件路径
        force_reload: 是否强制重新获取

    Returns:
        (成分股代码集合, {代码: 名称}字典)
    """
    global _CACHED_CONSTITUENTS, _CACHED_NAMES

    # 如果已缓存且不强制刷新
    if not force_reload and _CACHED_CONSTITUENTS is not None:
        return _CACHED_CONSTITUENTS, _CACHED_NAMES

    cache_file = Path(cache_file) if cache_file else None

    # 尝试从文件加载
    if cache_file and cache_file.exists() and not force_reload:
        try:
            df = pd.read_csv(cache_file, dtype={'code': str})
            _CACHED_CONSTITUENTS = set(df['code'].str.zfill(6))
            _CACHED_NAMES = dict(zip(df['code'].str.zfill(6), df['name']))
            logger.info(f"从缓存加载 {len(_CACHED_CONSTITUENTS)} 只成分股")
            return _CACHED_CONSTITUENTS, _CACHED_NAMES
        except Exception as e:
            logger.warning(f"缓存文件读取失败: {e}")

    # 从API获取
    _CACHED_CONSTITUENTS, _CACHED_NAMES = fetch_csi300_constituents(
        save_path=cache_file,
        include_names=True
    )

    return _CACHED_CONSTITUENTS, _CACHED_NAMES


if __name__ == "__main__":
    # 测试获取成分股
    logging.basicConfig(level=logging.INFO)

    codes, names = get_constituents_cached(force_reload=True)

    print(f"\n沪深300成分股数量: {len(codes)}")
    print(f"\n前10只成分股:")
    for i, (code, name) in enumerate(list(names.items())[:10]):
        print(f"  {code}: {name}")

    # 测试过滤功能
    test_notices = pd.DataFrame([
        {"title": "000001 平安银行: 关于重大事项的公告", "content": ""},
        {"title": "600519 贵州茅台: 2024年业绩预告", "content": ""},
        {"title": "300001 特锐德: 关于股份回购的公告", "content": ""},  # 创业板，不在沪深300
        {"title": "央行: 下调存款准备金率", "content": ""},  # 宏观公告，无代码
    ])

    print("\n测试公告过滤:")
    filtered = filter_notices_by_constituents(test_notices, codes, keep_no_code=True)
    print(f"原始: {len(test_notices)} 条")
    print(f"过滤后: {len(filtered)} 条")
    print("保留的公告:")
    for title in filtered['title']:
        print(f"  - {title}")