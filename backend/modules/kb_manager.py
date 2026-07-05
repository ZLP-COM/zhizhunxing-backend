'''职准星 - 知识库管理模块（ChromaDB 向量检索 + 文件降级兜底）

核心职责：
  1. ChromaDB 初始化、文本分块（1200字符/块，200字符重叠）、嵌入检索
  2. 内置降级机制：向量库不可用时自动读取本地 knowledge/data 静态规则
  3. 两个检索函数：search_jd_for_recommendations（推理用）、search_kb_for_gap_analysis（差距分析用）
  4. 30 份本地 JD 批量入库，启动自动检测缺失知识库文件
  5. 统一入口 get_kb_retrieval_result(query)
'''

import os
import re
import json
import hashlib
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

# ChromaDB 导入已注释（避免无此模块时报错）
# import chromadb
# from chromadb.config import Settings
# from chromadb.errors import ChromaDBError

from ..core import config

logger = logging.getLogger(__name__)


# ============================================================
# ChromaDB 管理器（已注释，保留空类占位）
# ============================================================

class KnowledgeBaseManager:
    '''ChromaDB 知识库管理器（当前不可用，降级使用文件检索）'''

    def __init__(self, persist_path: Optional[str] = None):
        self.persist_path = persist_path or config.CHROMA_DB_PATH
        self._available = False
        logger.info(f'知识库管理器初始化（降级模式）: {self.persist_path}')

    def is_available(self) -> bool:
        return False

    def search(self, collection: str, query: str, n_results: int = 3, min_score: Optional[float] = None) -> List[str]:
        logger.warning(f'ChromaDB 不可用，无法检索 collection={collection}')
        return []

    def count(self, collection: str) -> int:
        return 0

    def add_documents(self, collection: str, documents: List[str], ids: List[str], metadatas: Optional[List[dict]] = None) -> None:
        logger.warning(f'ChromaDB 不可用，无法添加文档到 collection={collection}')

    def get_collection(self, name: str) -> Optional[Any]:
        return None

    def get_stats(self) -> dict:
        return {
            'judgment_rules': 0,
            'jds': 0,
            'holland_guide': 0,
            'available': False,
        }


# ============================================================
# 文本分块
# ============================================================

def split_by_heading(
    text: str,
    max_chars: Optional[int] = None,
    overlap: Optional[int] = None,
) -> List[str]:
    '''
    按标题（##）分块文本。

    规则：
      - 按 ## 标题分割
      - 单块不超过 max_chars（默认 1200）
      - 块间重叠 overlap 字符（默认 200）

    Args:
        text:    源文本
        max_chars: 单块最大字符数
        overlap:   块间重叠字符数

    Returns:
        分块后的文本列表
    '''
    max_chars = max_chars or config.KB_CHUNK_SIZE
    overlap = overlap or config.KB_CHUNK_OVERLAP

    sections = re.split(r'(?=^##\s)', text, flags=re.MULTILINE)
    chunks: List[str] = []
    current = ''
    for section in sections:
        if not section.strip():
            continue
        if len(current) + len(section) > max_chars and current:
            chunks.append(current.strip())
            current = current[-overlap:] + '\n' + section
        else:
            current += '\n' + section
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text.strip()]


# ============================================================
# 知识库初始化（ChromaDB 不可用，跳过）
# ============================================================

async def init_knowledge_base(kb: Optional[KnowledgeBaseManager] = None) -> bool:
    '''ChromaDB 不可用，跳过知识库初始化，返回 False。'''
    logger.warning('ChromaDB 未安装，跳过知识库初始化（将使用文件降级检索）')
    return False


# ============================================================
# 统一检索入口（降级：本地文件读取）
# ============================================================

async def get_kb_retrieval_result(
    query: str,
    collection: str = 'judgment_rules',
    n_results: int = 3,
    min_score: Optional[float] = None,
) -> List[str]:
    '''ChromaDB 不可用，从本地文件降级检索。'''
    logger.warning(f'ChromaDB 不可用，从本地文件降级检索 collection={collection}')
    if collection == 'judgment_rules':
        return [_read_fallback_rules()]
    if collection == 'jds':
        return _read_fallback_jds(query)
    return []


# ============================================================
# 模块 D 用：综合推理 → JD 检索（降级：本地关键词匹配）
# ============================================================

async def search_jd_for_recommendations(reasoning_result: dict) -> List[dict]:
    '''
    ChromaDB 不可用，使用本地关键词匹配检索 JD。

    3 层策略：
      1. 关键词精确匹配
      2. 岗位方向名模糊匹配（去后缀）
      3. 兜底返回前 2 个 JD

    Args:
        reasoning_result: 综合推理结果（含 recommendations）

    Returns:
        [{
            'job_direction': str,
            'match_reason':  str,
            'match_score':   int,
            'jd_list':       [{'company': str, 'title': str, 'content': str}, ...]
        }]
    '''
    import re
    recommendations = reasoning_result.get('recommendations', [])
    results: List[dict] = []

    # 预读全部 JD
    jd_dir = config.JDS_DIR
    all_jds: List[dict] = []
    if jd_dir.exists():
        for fpath in sorted(jd_dir.glob('*.txt')):
            try:
                content = fpath.read_text(encoding='utf-8')
                company, title = '', fpath.stem
                for line in content.split('\n'):
                    if line.startswith('公司名称') or line.startswith('公司:'):
                        _, _, val = line.partition('：')
                        if not val:
                            _, _, val = line.partition(':')
                        company = val.strip()
                    elif line.startswith('岗位名称') or line.startswith('岗位:'):
                        _, _, val = line.partition('：')
                        if not val:
                            _, _, val = line.partition(':')
                        title = val.strip()
                all_jds.append({
                    'company': company or '未知公司',
                    'title': title,
                    'content': content,
                })
            except Exception:
                continue

    for rec in recommendations:
        direction = rec.get('job_direction', '')
        keywords = rec.get('jd_search_keywords', [])
        q_keywords = ' '.join(keywords).lower().split()

        jd_list: List[dict] = []

        # 策略1：关键词匹配
        for jd in all_jds:
            if any(kw in jd['content'].lower() for kw in q_keywords):
                jd_list.append({'company': jd['company'], 'title': jd['title'], 'content': jd['content']})
                if len(jd_list) >= 2:
                    break

        # 策略2：方向名模糊匹配
        if not jd_list and direction:
            dir_key = re.sub(r'专员|助理|管培生|实习生', '', direction).strip().lower()
            for jd in all_jds:
                lower_title = jd['title'].lower()
                lower_content = jd['content'].lower()
                if dir_key and (dir_key in lower_title or dir_key in lower_content):
                    jd_list.append({'company': jd['company'], 'title': jd['title'], 'content': jd['content']})
                    if len(jd_list) >= 2:
                        break

        # 策略3：兜底
        if not jd_list:
            for jd in all_jds[:2]:
                jd_list.append({'company': jd['company'], 'title': jd['title'], 'content': jd['content']})

        results.append({
            'job_direction': direction,
            'match_reason': rec.get('match_reason', ''),
            'match_score': rec.get('match_score', 0),
            'jd_list': jd_list,
        })

    return results


# ============================================================
# 模块 H 用：差距分析 → 校招规则检索（降级：本地文件）
# ============================================================

async def search_kb_for_gap_analysis(career_fit: str) -> str:
    '''
    用简历解析输出的 career_fit 检索本地校招评判规则文件。

    Args:
        career_fit: 简历解析出的岗位方向关键词

    Returns:
        拼接的上下文文本
    '''
    rules_docs = _read_fallback_rules()
    jd_docs = _read_fallback_jds(career_fit)

    parts: List[str] = []
    if rules_docs:
        parts.append('【校招评判规则（默认）】\n' + rules_docs)
    if jd_docs:
        parts.append('【相似JD参考】\n' + '\n\n'.join(jd_docs))

    context = '\n\n'.join(parts)
    if len(context) > config.KB_CONTEXT_MAX_LENGTH:
        context = context[:config.KB_CONTEXT_MAX_LENGTH]

    return context or '（无知识库检索结果）'


# ============================================================
# 产品说明书检索（降级：关键词匹配）
# ============================================================

async def search_product_manual(query: str, n_results: int = 3) -> str:
    '''ChromaDB 不可用，使用关键词降级匹配。'''
    logger.warning('产品说明书向量检索不可用，使用关键词降级匹配')
    return _fallback_readme_keyword_match(query)


def _fallback_readme_keyword_match(query: str) -> str:
    '''
    读取 README.md 全文做关键词匹配。

    Args:
        query: 用户问题文本

    Returns:
        匹配的 README 片段
    '''
    try:
        manual_path = config.PRODUCT_MANUAL_PATH
        if not manual_path.exists():
            logger.warning(f'产品说明书文件不存在: {manual_path}')
            return ''
        content = manual_path.read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f'读取产品说明书失败: {str(e)}')
        return ''

    stop_words = {'的', '了', '是', '在', '有', '和', '就', '不', '人', '都',
                  '一', '个', '上', '也', '很', '到', '说', '要', '去', '你',
                  '会', '着', '没有', '看', '好', '自己', '这', '我', '他',
                  '它', '她', '们', '什么', '怎么', '如何', '为什么', '吗',
                  '吧', '呢', '啊', '哦', '嗯', 'the', 'a', 'an', 'is', 'are',
                  'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                  'do', 'does', 'did', 'will', 'would', 'could', 'should',
                  'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
                  'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through',
                  'during', 'before', 'after', 'above', 'below', 'between',
                  'and', 'but', 'or', 'nor', 'not', 'so', 'yet', 'both',
                  'either', 'neither', 'each', 'every', 'all', 'any', 'few',
                  'more', 'most', 'other', 'some', 'such', 'no', 'only',
                  'own', 'same', 'than', 'too', 'very', 'just', 'about',
                  'up', 'out', 'if', 'while', 'because', 'until', 'although',
                  'though', 'when', 'where', 'how', 'which', 'who', 'whom',
                  'what', 'this', 'that', 'these', 'those'}

    words = set()
    for w in re.findall(r'[a-zA-Z]{2,}', query.lower()):
        if w not in stop_words:
            words.add(w)
    chinese_chars = re.findall(r'[\u4e00-\u9fff]+', query)
    for seg in chinese_chars:
        for i in range(len(seg)):
            for j in range(i + 2, min(i + 5, len(seg) + 1)):
                words.add(seg[i:j])

    if not words:
        return content[:2000]

    paragraphs = re.split(r'\n\s*\n', content)
    scored: list[tuple[int, str]] = []
    for para in paragraphs:
        para_lower = para.lower()
        score = sum(1 for w in words if w in para_lower)
        if score > 0:
            scored.append((score, para.strip()))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:3]
    if not top:
        return content[:2000]
    result = '\n\n---\n\n'.join(p for _, p in top)
    return result[:2000] if len(result) > 2000 else result


# ============================================================
# JD 文本解析
# ============================================================

def _parse_jd_text(jd_text: str) -> dict:
    '''从 JD 纯文本中提取公司名、岗位名。'''
    company = ''
    title = ''
    for line in jd_text.split('\n'):
        line = line.strip()
        if line.startswith('公司名称') or line.startswith('公司:'):
            _, _, val = line.partition('：')
            if not val:
                _, _, val = line.partition(':')
            company = val.strip()
        elif line.startswith('岗位名称') or line.startswith('岗位:'):
            _, _, val = line.partition('：')
            if not val:
                _, _, val = line.partition(':')
            title = val.strip()
    return {
        'company': company or '未知公司',
        'title': title or '未知岗位',
        'content': jd_text,
    }


# ============================================================
# 降级：本地文件读取
# ============================================================

def _read_fallback_rules() -> str:
    '''读取本地 judgment_rules.md 静态规则。'''
    rules_file = config.JUDGMENT_RULES_FILE
    if rules_file.exists():
        try:
            with open(rules_file, 'r', encoding='utf-8') as f:
                return f.read()[:2000]
        except Exception as e:
            logger.error(f'读取降级规则失败: {str(e)}')
    return ''


def _read_fallback_jds(query: str) -> List[str]:
    '''从本地 JD 目录中检索匹配的 JD 文本。'''
    jd_dir = config.JDS_DIR
    if not jd_dir.exists():
        return []
    keywords = query.lower().split()
    matched: List[str] = []
    for fpath in sorted(jd_dir.glob('*.txt')):
        try:
            content = fpath.read_text(encoding='utf-8')
            if any(kw in content.lower() for kw in keywords):
                matched.append(content)
        except Exception:
            continue
    return matched[:2]


# ============================================================
# 全局单例
# ============================================================

kb_manager = KnowledgeBaseManager()
'''全局知识库管理器实例'''
