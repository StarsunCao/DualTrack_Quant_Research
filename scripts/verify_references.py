#!/usr/bin/env python3
"""
参考文献验证脚本。

验证参考文献的真实性，防止 AI 幻觉。
支持 arXiv 论文、DOI 论文和期刊论文的验证。

用法:
    uv run python scripts/verify_references.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 请求头
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
}


@dataclass
class PaperInfo:
    """论文信息数据类。"""
    index: int
    title: str
    authors: List[str]
    year: int
    source: str  # arxiv, journal, conference
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    verified: bool = False
    local_file: Optional[str] = None
    notes: str = ""


def parse_references_file(file_path: str) -> List[Dict[str, Any]]:
    """
    解析参考文献 Markdown 文件。

    Args:
        file_path: references.md 文件路径。

    Returns:
        解析后的参考文献列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    papers = []

    # 匹配 IEEE 格式的参考文献
    # 例如: [1] H. Yang, X. Liu, and Z. Wang, "FinGPT: Open-Source Financial Large Language Models," *arXiv preprint arXiv:2306.06031*, 2023.
    # 改进的正则表达式
    lines = content.split('\n')

    for line in lines:
        line = line.strip()
        # 跳过空行和标题行
        if not line or line.startswith('#') or line.startswith('**') or line.startswith('---') or line.startswith('```'):
            continue

        # 匹配 [数字] 开头的行
        import re
        match = re.match(r'\[(\d+)\]\s+(.+?),\s+"(.+?)",?\s+\*(.+?)\*?,?\s+(\d{4})', line)
        if match:
            index, authors, title, source, year = match.groups()

            # 提取 arXiv ID
            arxiv_id = None
            arxiv_pattern = r'arXiv[:\s]+(\d{4}\.\d{4,5})'
            arxiv_match = re.search(arxiv_pattern, source)
            if arxiv_match:
                arxiv_id = arxiv_match.group(1)
            # 也检查标题中的 arXiv
            if not arxiv_id:
                arxiv_match = re.search(arxiv_pattern, title)
                if arxiv_match:
                    arxiv_id = arxiv_match.group(1)

            papers.append({
                'index': int(index),
                'authors': authors.strip(),
                'title': title.strip(),
                'source': source.strip(),
                'year': int(year),
                'arxiv_id': arxiv_id
            })

    logger.info(f"Parsed {len(papers)} references from {file_path}")
    return papers


def verify_arxiv_paper(arxiv_id: str) -> Optional[Dict[str, Any]]:
    """
    验证 arXiv 论文真实性。

    Args:
        arxiv_id: arXiv ID (如 '2306.06031')。

    Returns:
        论文信息字典，失败返回 None。
    """
    try:
        # 使用 arXiv API
        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        response = requests.get(url, headers=HEADERS, timeout=30)

        if response.status_code != 200:
            return None

        # 解析 XML 响应
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)

        # 定义命名空间
        ns = {'atom': 'http://www.w3.org/2005/Atom'}

        entries = root.findall('atom:entry', ns)

        if not entries:
            return None

        entry = entries[0]

        # 提取信息
        title_elem = entry.find('atom:title', ns)
        title = title_elem.text.strip() if title_elem is not None else "Unknown"

        authors = []
        for author in entry.findall('atom:author', ns):
            name_elem = author.find('atom:name', ns)
            if name_elem is not None:
                authors.append(name_elem.text)

        published_elem = entry.find('atom:published', ns)
        year = int(published_elem.text[:4]) if published_elem is not None else 0

        return {
            'title': title,
            'authors': authors,
            'year': year,
            'arxiv_id': arxiv_id,
            'pdf_url': f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            'abs_url': f"https://arxiv.org/abs/{arxiv_id}"
        }

    except Exception as e:
        logger.error(f"Error verifying arXiv paper {arxiv_id}: {e}")
        return None


def download_arxiv_pdf(arxiv_id: str, output_dir: str, filename: str) -> bool:
    """
    下载 arXiv 论文 PDF。

    Args:
        arxiv_id: arXiv ID。
        output_dir: 输出目录。
        filename: 输出文件名。

    Returns:
        是否下载成功。
    """
    try:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        output_path = Path(output_dir) / filename

        if output_path.exists():
            logger.info(f"PDF already exists: {output_path}")
            return True

        response = requests.get(pdf_url, headers=HEADERS, timeout=60, stream=True)

        if response.status_code != 200:
            logger.error(f"Failed to download PDF: {pdf_url}")
            return False

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info(f"Downloaded: {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error downloading PDF {arxiv_id}: {e}")
        return False


def verify_all_references(references_path: str, output_dir: str) -> Dict[str, Any]:
    """
    验证所有参考文献。

    Args:
        references_path: references.md 文件路径。
        output_dir: PDF 输出目录。

    Returns:
        验证结果字典。
    """
    papers = parse_references_file(references_path)

    results = {
        'total': len(papers),
        'verified': 0,
        'failed': 0,
        'skipped': 0,
        'papers': []
    }

    papers_dir = Path(output_dir) / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    for paper in papers:
        index = paper['index']
        title = paper['title']
        arxiv_id = paper['arxiv_id']

        result = {
            'index': index,
            'title': title,
            'authors': paper['authors'],
            'year': paper['year'],
            'arxiv_id': arxiv_id,
            'verified': False,
            'local_file': None,
            'notes': ''
        }

        if arxiv_id:
            logger.info(f"\nVerifying [{index}] {title[:50]}... (arXiv: {arxiv_id})")

            # 验证论文
            info = verify_arxiv_paper(arxiv_id)

            if info:
                result['verified'] = True
                results['verified'] += 1

                # 下载 PDF
                filename = f"{info['authors'][0].split()[-1].lower()}_{info['year']}_{arxiv_id}.pdf"
                if download_arxiv_pdf(arxiv_id, str(papers_dir), filename):
                    result['local_file'] = str(papers_dir / filename)

                # 检查标题匹配度
                if title.lower() not in info['title'].lower() and info['title'].lower() not in title.lower():
                    result['notes'] = f"Title mismatch: API returns '{info['title'][:50]}...'"
            else:
                results['failed'] += 1
                result['notes'] = "Verification failed"
        else:
            # 非 arXiv 论文，标记为需要手动验证
            results['skipped'] += 1
            result['notes'] = "Non-arXiv paper - manual verification required"

        results['papers'].append(result)

    return results


def generate_verification_report(results: Dict[str, Any], output_path: str) -> None:
    """
    生成验证报告。

    Args:
        results: 验证结果。
        output_path: 输出文件路径。
    """
    report = f"""# 参考文献真实性验证报告

**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**验证总数**: {results['total']}
**验证成功**: {results['verified']}
**验证失败**: {results['failed']}
**跳过验证**: {results['skipped']}

---

## 验证状态汇总

| # | 标题 | arXiv ID | 状态 | 本地文件 | 备注 |
|---|------|----------|------|---------|------|
"""

    for paper in results['papers']:
        status = "✅ 已验证" if paper['verified'] else "❌ 失败" if paper['notes'] == "Verification failed" else "⏭️ 跳过"
        local = paper['local_file'].split('/')[-1] if paper['local_file'] else "-"
        report += f"| {paper['index']} | {paper['title'][:40]}... | {paper['arxiv_id'] or '-'} | {status} | {local} | {paper['notes'][:30]} |\n"

    report += """
---

## 统计信息

"""

    # 计算验证率
    if results['total'] > 0:
        verification_rate = results['verified'] / results['total'] * 100
        report += f"- **验证成功率**: {verification_rate:.1f}%\n"
        report += f"- **已下载 PDF**: {sum(1 for p in results['papers'] if p['local_file'])} 篇\n"

    report += """
---

## 注意事项

1. arXiv 论文通过 arXiv API 自动验证
2. 期刊论文需要通过学校图书馆或 Google Scholar 手动验证
3. 部分最新论文可能尚未被 arXiv API 收录

---

*报告由 verify_references.py 自动生成*
"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"\nVerification report saved to: {output_path}")


def main():
    """主函数。"""
    logger.info("="*60)
    logger.info("  Reference Verification Script")
    logger.info("="*60)

    references_path = "docs/reference/references.md"
    output_dir = "docs/reference"

    if not Path(references_path).exists():
        logger.error(f"References file not found: {references_path}")
        return

    # 运行验证
    results = verify_all_references(references_path, output_dir)

    # 生成报告
    report_path = Path(output_dir) / "verification_status.md"
    generate_verification_report(results, str(report_path))

    # 打印摘要
    logger.info("\n" + "="*60)
    logger.info("  Verification Summary")
    logger.info("="*60)
    logger.info(f"Total papers: {results['total']}")
    logger.info(f"Verified: {results['verified']}")
    logger.info(f"Failed: {results['failed']}")
    logger.info(f"Skipped: {results['skipped']}")

    if results['total'] > 0:
        rate = results['verified'] / results['total'] * 100
        logger.info(f"\nVerification rate: {rate:.1f}%")


if __name__ == "__main__":
    main()