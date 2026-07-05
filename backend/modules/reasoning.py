'''职准星 - 综合推理与岗位推荐模块

接收六问测评 6 份分析结果，执行：
  1. LLM 综合推理 → 霍兰德 3 字码 + 3 个推荐岗位方向
  2. 知识库 JD 检索 → 按 jd_search_keywords 匹配真实 JD
  3. LLM 格式化推荐 → 拼接为用户可读报告（含 emoji + JD 摘要）
  4. 缓存推理结果至 session，支持 10 分钟报告复用
  5. 统一入口函数 run_job_reason(session)
'''

import json
import hashlib
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

from ..core import config
from ..core.llm import llm_client, LLMError
from ..core.prompts import (
    REASONING_SYSTEM_PROMPT,
    REASONING_USER_PROMPT_TEMPLATE,
    FORMAT_RECOMMEND_SYSTEM,
    FORMAT_RECOMMEND_USER_TEMPLATE,
)
# ChromaDB 向量库未安装，KB 检索降级为本地文件匹配
# from ..modules.kb_manager import search_jd_for_recommendations
from ..models.session import Session

logger = logging.getLogger(__name__)


# ============================================================
# 推理结果缓存 {session_id: (timestamp, result_dict)}
# ============================================================

_reasoning_cache: Dict[str, tuple] = {}


def _get_cache_key(session: Session) -> str:
    '''基于 session id + 测评答案生成缓存 key'''
    answers = session.assessment_answers or {}
    raw = session.id + json.dumps(answers, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(raw.encode()).hexdigest()


# ============================================================
# 统一入口函数
# ============================================================

async def run_job_reason(session: Session) -> dict:
    '''
    综合推理统一入口函数。

    从 session 读取 6 份测评分析结果，依次执行：
      1. 缓存命中检查
      2. LLM 综合推理（scene=reasoning）
      3. 知识库 JD 检索
      4. LLM 格式化推荐报告
      5. 结果写入 session

    Args:
        session: 当前会话（含 assessment_results）

    Returns:
        {
            'reply':        str,   # 格式化后的推荐报告
            'action':       str,   # 'SHOWING_RECOMMENDATIONS'
            'reasoning':    dict,  # 综合推理结果
            'jd_results':   list,  # JD 检索结果
            'formatted':    str,   # 格式化报告
        }
    '''
    results = session.assessment_results or {}

    # ---- 1. 缓存检查 ----
    cache_key = _get_cache_key(session)
    if cache_key in _reasoning_cache:
        cached_time, cached = _reasoning_cache[cache_key]
        if datetime.now() - cached_time < timedelta(minutes=config.REPORT_CACHE_MINUTES):
            logger.info(f'会话 {session.id}: 命中推理缓存')
            session.reasoning_result = cached['reasoning']
            session.selected_jd_list = cached['jd_results']
            session.formatted_recommendation = cached['formatted']
            session.state = 'SHOWING_RECOMMENDATIONS'
            return cached

    # ---- 2. LLM 综合推理 ----
    reasoning_result = await _execute_reasoning(
        q1_result=results.get('Q1', {}),
        q2_result=results.get('Q2', {}),
        q3_result=results.get('Q3', {}),
        q4_result=results.get('Q4', {}),
        q5_result=results.get('Q5', {}),
        q6_result=results.get('Q6', {}),
        session_id=session.id,
    )
    session.reasoning_result = reasoning_result

    # ---- 3. 知识库 JD 检索（降级：本地关键词匹配） ----
    jd_results = await _local_jd_search(reasoning_result)
    session.selected_jd_list = jd_results

    # ---- 4. 格式化推荐报告 ----
    formatted = await _format_recommendation(reasoning_result, jd_results, session.id)
    session.formatted_recommendation = formatted

    # ---- 5. 更新状态 ----
    session.state = 'SHOWING_RECOMMENDATIONS'

    # ---- 写入缓存 ----
    output = {
        'reply': formatted,
        'action': 'SHOWING_RECOMMENDATIONS',
        'reasoning': reasoning_result,
        'jd_results': jd_results,
        'formatted': formatted,
    }
    _reasoning_cache[cache_key] = (datetime.now(), output)

    logger.info(
        f'会话 {session.id}: 综合推理完成, '
        f'霍兰德={reasoning_result.get("holland_code", "N/A")}'
    )
    return output


# ============================================================
# LLM 综合推理
# ============================================================

async def _execute_reasoning(
    q1_result: dict,
    q2_result: dict,
    q3_result: dict,
    q4_result: dict,
    q5_result: dict,
    q6_result: dict,
    session_id: Optional[str] = None,
) -> dict:
    '''
    综合推理：6 个分析结果 → 霍兰德 3 字码 + 3 岗位方向。

    权重分配（对应需求文档模块 C）：
      Q1(30%) + Q2(15%) + Q3(15%) + Q4(10%) + Q5(20%) + Q6(10%)
    '''
    user_prompt = REASONING_USER_PROMPT_TEMPLATE.format(
        q1_result=json.dumps(q1_result, ensure_ascii=False),
        q2_result=json.dumps(q2_result, ensure_ascii=False),
        q3_result=json.dumps(q3_result, ensure_ascii=False),
        q4_result=json.dumps(q4_result, ensure_ascii=False),
        q5_result=json.dumps(q5_result, ensure_ascii=False),
        q6_result=json.dumps(q6_result, ensure_ascii=False),
    )

    # 降级兜底
    fallback: dict = {
        'holland_code': 'SEC',
        'primary_type': 'S型-社会型',
        'type_description': '喜欢与人打交道，善于沟通协调',
        'recommendations': [
            {
                'rank': 1,
                'job_direction': '人力资源专员',
                'match_reason': '综合评估推荐',
                'match_score': 85,
                'jd_search_keywords': ['人力资源专员', '校招'],
            },
            {
                'rank': 2,
                'job_direction': '行政助理',
                'match_reason': '综合评估推荐',
                'match_score': 80,
                'jd_search_keywords': ['行政助理', '校招'],
            },
            {
                'rank': 3,
                'job_direction': '培训助理',
                'match_reason': '综合评估推荐',
                'match_score': 75,
                'jd_search_keywords': ['培训助理', '校招'],
            },
        ],
    }

    try:
        result = await llm_client.safe_chat_json(
            system_prompt=REASONING_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            scene='reasoning',
            session_id=session_id,
            fallback=fallback,
        )
        # 校验 recommendations 数量
        recs = result.get('recommendations', [])
        if len(recs) < 3:
            logger.warning(f'推荐岗位不足3个({len(recs)})，使用兜底填充')
            for i in range(len(recs), 3):
                recs.append(fallback['recommendations'][i])
            result['recommendations'] = recs
        return result

    except LLMError as e:
        logger.error(f'综合推理LLM调用失败: {str(e)}')
        return fallback
    except Exception as e:
        logger.error(f'综合推理未知异常: {str(e)}')
        return fallback


# ============================================================
# 格式化推荐报告
# ============================================================

async def _format_recommendation(
    reasoning_result: dict,
    jd_results: list,
    session_id: Optional[str] = None,
) -> str:
    '''
    格式化推荐报告：综合推理结果 + 知识库 JD → 用户可读文本。

    LLM 场景 scene=format（温度 0.7），
    失败时使用 _build_fallback_text() 静态降级。
    '''
    user_prompt = FORMAT_RECOMMEND_USER_TEMPLATE.format(
        reasoning=json.dumps(reasoning_result, ensure_ascii=False, indent=2),
        jd_results=json.dumps(jd_results, ensure_ascii=False, indent=2),
    )

    try:
        result = await llm_client.chat(
            system_prompt=FORMAT_RECOMMEND_SYSTEM,
            user_prompt=user_prompt,
            scene='format',
            session_id=session_id,
        )
        return result.get('output', '')
    except LLMError as e:
        logger.error(f'格式化推荐LLM调用失败: {str(e)}')
        return _build_fallback_text(reasoning_result, jd_results)
    except Exception as e:
        logger.error(f'格式化推荐未知异常: {str(e)}')
        return _build_fallback_text(reasoning_result, jd_results)


# ============================================================
# 降级推荐报告
# ============================================================

def _build_fallback_text(reasoning: dict, jd_results: list) -> str:
    '''LLM 不可用时，用静态模板构建推荐报告'''
    lines: List[str] = []
    holland = reasoning.get('holland_code', 'N/A')
    primary = reasoning.get('primary_type', '')
    desc = reasoning.get('type_description', '')

    lines.append(f'\U0001f3af 你的霍兰德代码：**{holland}**')
    if primary:
        lines.append(f'\U0001f4ca 主导类型：{primary}')
    if desc:
        lines.append(f'\U0001f4a1 {desc}')
    lines.append('')

    recs = reasoning.get('recommendations', [])
    lines.append('\U0001f4cb 推荐岗位方向：')
    for i, rec in enumerate(recs, 1):
        name = rec.get('job_direction', f'方向{i}')
        score = rec.get('match_score', '?')
        reason = rec.get('match_reason', '')
        lines.append(f'\n**{i}. {name}**（匹配度：{score}%）')
        if reason:
            lines.append(f'   {reason}')

        # 添加 JD 摘要
        for jd_res in jd_results:
            if jd_res.get('job_direction') == name:
                for jd in jd_res.get('jd_list', []):
                    company = jd.get('company', '')
                    title = jd.get('title', '')
                    if company or title:
                        lines.append(f'   \U0001f4c4 配套JD：{company} - {title}')

    lines.append(
        '\n---\n\U0001f4ac 请告诉我你想深入分析哪个方向？（回复 1/2/3 或方向名称）'
    )

    return '\n'.join(lines)


# ============================================================
# 对 main.py 旧接口的兼容导出
# ============================================================

async def execute_reasoning(
    q1_result: dict,
    q2_result: dict,
    q3_result: dict,
    q4_result: dict,
    q5_result: dict,
    q6_result: dict,
    session_id: Optional[str] = None,
) -> dict:
    '''兼容旧接口：直接调用 _execute_reasoning'''
    return await _execute_reasoning(
        q1_result, q2_result, q3_result,
        q4_result, q5_result, q6_result,
        session_id,
    )


async def format_recommendation(
    reasoning_result: dict,
    jd_results: list,
    session_id: Optional[str] = None,
) -> str:
    '''兼容旧接口：直接调用 _format_recommendation'''
    return await _format_recommendation(reasoning_result, jd_results, session_id)


def _build_fallback_recommendation(reasoning: dict, jd_results: list) -> str:
    '''兼容旧接口：直接调用 _build_fallback_text'''
    return _build_fallback_text(reasoning, jd_results)


# ============================================================
# 本地 JD 降级检索（ChromaDB 不可用时使用）
# ============================================================

async def _local_jd_search(reasoning_result: dict) -> list:
    '''
    不使用 ChromaDB 的 JD 检索降级方案，直接读取本地 JD 文件关键词匹配。

    改进：先用关键词精确匹配，匹配不到时用岗位方向名称做模糊匹配，
    还找不到则返回全部 JD 中的前 2 个作为兜底。

    Args:
        reasoning_result: 综合推理结果（含 recommendations）

    Returns:
        [{'job_direction', 'match_reason', 'match_score', 'jd_list': [...]}]
    '''
    import json
    from pathlib import Path

    recommendations = reasoning_result.get('recommendations', [])
    results = []
    jd_dir = config.JDS_DIR

    # 预先读取所有 JD 文件
    all_jds: list[dict] = []
    if jd_dir.exists():
        for fpath in sorted(jd_dir.glob('*.txt')):
            try:
                content = fpath.read_text(encoding='utf-8')
                company, title = '', ''
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
                    'title': title or fpath.stem,
                    'content': content,
                    'filename': fpath.stem,
                })
            except Exception:
                continue

    for rec in recommendations:
        direction = rec.get('job_direction', '')
        keywords = rec.get('jd_search_keywords', [])
        query = ' '.join(keywords).lower()
        q_keywords = query.split()

        # 策略1：关键词精确匹配
        jd_list = []
        for jd in all_jds:
            content_lower = jd['content'].lower()
            if any(kw in content_lower for kw in q_keywords):
                jd_list.append({'company': jd['company'], 'title': jd['title'], 'content': jd['content']})
                if len(jd_list) >= 2:
                    break

        # 策略2：岗位方向名模糊匹配
        if not jd_list and direction:
            dir_keywords = direction.lower().replace('专员', '').replace('助理', '').replace('管培生', '').replace('实习生', '').strip()
            for jd in all_jds:
                title_lower = jd['title'].lower()
                content_lower = jd['content'].lower()
                if dir_keywords and (dir_keywords in title_lower or dir_keywords in content_lower):
                    jd_list.append({'company': jd['company'], 'title': jd['title'], 'content': jd['content']})
                    if len(jd_list) >= 2:
                        break

        # 策略3：兜底，取前 2 个 JD
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
