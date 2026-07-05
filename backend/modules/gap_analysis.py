'''职准星 - 差距分析工作流模块

5 节点并行 LLM 差距分析工作流：

  节点1 ── 简历解析（LLM）
  节点2 ── JD 拆解（LLM）
  节点1+2 并行执行
      │
  节点3 ── 知识库检索（KB，依赖节点1的 career_fit）
      │
  节点4 ── 差距诊断（LLM 核心节点，依赖节点1+2+3）
      │
  节点5 ── 简历优化（LLM，与节点6并行）
  节点6 ── 逆袭话术（LLM，与节点5并行）

内置并行任务超时控制，单节点失败自动兜底静态评判规则。
统一入口函数 run_gap_analysis(session, resume_text, target_jd)。
'''

import json
import asyncio
import logging
from typing import Any, Dict, Optional

from ..core import config
from ..core.llm import llm_client, LLMError
from ..core.prompts import (
    NODE1_SYSTEM_PROMPT, NODE1_USER_PROMPT_TEMPLATE,
    NODE2_SYSTEM_PROMPT, NODE2_USER_PROMPT_TEMPLATE,
    NODE4_SYSTEM_PROMPT, NODE4_USER_PROMPT_TEMPLATE,
    NODE5_SYSTEM_PROMPT, NODE5_USER_PROMPT_TEMPLATE,
    NODE6_SYSTEM_PROMPT, NODE6_USER_PROMPT_TEMPLATE,
)
from ..modules.kb_manager import search_kb_for_gap_analysis
from ..models.session import Session

logger = logging.getLogger(__name__)


# ============================================================
# 并行执行超时（秒）
# ============================================================

_PARALLEL_TIMEOUT = 45  # 单个 gather 超时


# ============================================================
# 统一入口函数
# ============================================================

async def run_gap_analysis(
    session: Session,
    resume_text: str,
    target_jd: str,
) -> dict:
    '''
    差距分析工作流统一入口函数。

    Args:
        session:     当前会话（用于写入结果）
        resume_text: 简历纯文本
        target_jd:   JD 纯文本

    Returns:
        {
            'gap_data':            dict,   # 差距诊断结果
            'optimized_data':      dict,   # 简历优化结果
            'counterattack_data':  dict,   # 逆袭话术结果
        }
    '''
    logger.info(
        f'会话 {session.id}: 开始差距分析 | 简历={len(resume_text)}字 | JD={len(target_jd)}字'
    )

    # ---- 阶段1：节点1+2 并行 ----
    node1_task = _node1_parse_resume(resume_text, session.id)
    node2_task = _node2_parse_jd(target_jd, session.id)

    try:
        node1_result, node2_result = await asyncio.wait_for(
            asyncio.gather(node1_task, node2_task, return_exceptions=True),
            timeout=_PARALLEL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(f'会话 {session.id}: 节点1+2 并行超时')
        node1_result = _fallback_parsed_resume(resume_text)
        node2_result = _fallback_parsed_jd(target_jd)

    # 异常降级
    if isinstance(node1_result, Exception):
        logger.error(f'节点1(简历解析)异常: {str(node1_result)}')
        node1_result = _fallback_parsed_resume(resume_text)
    if isinstance(node2_result, Exception):
        logger.error(f'节点2(JD拆解)异常: {str(node2_result)}')
        node2_result = _fallback_parsed_jd(target_jd)

    # ---- 节点3：知识库检索 ----
    career_fit = node1_result.get('summary', {}).get('career_fit', '')
    kb_context = await search_kb_for_gap_analysis(career_fit)

    # ---- 节点4：差距诊断（串行，依赖节点1+2+3） ----
    gap_result = await _node4_gap_diagnosis(node1_result, node2_result, kb_context, session.id)

    # ---- 阶段2：节点5+6 并行 ----
    node5_task = _node5_optimize_resume(resume_text, gap_result, session.id)
    node6_task = _node6_counterattack(gap_result, session.id)

    try:
        node5_result, node6_result = await asyncio.wait_for(
            asyncio.gather(node5_task, node6_task, return_exceptions=True),
            timeout=_PARALLEL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.error(f'会话 {session.id}: 节点5+6 并行超时')
        node5_result = _fallback_optimized()
        node6_result = _fallback_counterattack()

    if isinstance(node5_result, Exception):
        logger.error(f'节点5(简历优化)异常: {str(node5_result)}')
        node5_result = _fallback_optimized()
    if isinstance(node6_result, Exception):
        logger.error(f'节点6(逆袭话术)异常: {str(node6_result)}')
        node6_result = _fallback_counterattack()

    # ---- 写入 session ----
    session.gap_analysis_result = gap_result
    session.optimized_resume_result = node5_result
    session.counterattack_result = node6_result

    logger.info(f'会话 {session.id}: 差距分析工作流完成')
    return {
        'gap_data': gap_result,
        'optimized_data': node5_result,
        'counterattack_data': node6_result,
    }


# ============================================================
# 节点1：简历解析（LLM）
# ============================================================

async def _node1_parse_resume(resume_text: str, session_id: Optional[str] = None) -> dict:
    '''简历解析：混乱文本 → 结构化 JSON（education/experience/skills/honors/summary）'''
    user_prompt = NODE1_USER_PROMPT_TEMPLATE.format(resume_text=resume_text)
    fallback = _fallback_parsed_resume(resume_text)

    try:
        return await llm_client.safe_chat_json(
            system_prompt=NODE1_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            scene='diagnosis',
            session_id=session_id,
            fallback=fallback,
        )
    except LLMError as e:
        logger.error(f'节点1 LLM异常: {str(e)}')
        return fallback
    except Exception as e:
        logger.error(f'节点1 未知异常: {str(e)}')
        return fallback


# ============================================================
# 节点2：JD 拆解（LLM）
# ============================================================

async def _node2_parse_jd(jd_text: str, session_id: Optional[str] = None) -> dict:
    '''JD 拆解：JD 全文 → 结构化需求（hard_requirements/core_capabilities/bonus_items/priority）'''
    user_prompt = NODE2_USER_PROMPT_TEMPLATE.format(jd_text=jd_text)
    fallback = _fallback_parsed_jd(jd_text)

    try:
        return await llm_client.safe_chat_json(
            system_prompt=NODE2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            scene='diagnosis',
            session_id=session_id,
            fallback=fallback,
        )
    except LLMError as e:
        logger.error(f'节点2 LLM异常: {str(e)}')
        return fallback
    except Exception as e:
        logger.error(f'节点2 未知异常: {str(e)}')
        return fallback


# ============================================================
# 节点4：差距诊断（LLM 核心节点）
# ============================================================

async def _node4_gap_diagnosis(
    parsed_resume: dict,
    parsed_jd: dict,
    kb_context: str,
    session_id: Optional[str] = None,
) -> dict:
    '''
    差距诊断核心节点。

    输入：简历结构化数据 + JD 结构化需求 + 知识库校招规则
    输出：match_score（教育×30%+经验×40%+技能×30%）+ gap_analysis（红黄绿灯）+ top3_fatal_gaps
    '''
    user_prompt = NODE4_USER_PROMPT_TEMPLATE.format(
        parsed_resume=json.dumps(parsed_resume, ensure_ascii=False),
        parsed_jd=json.dumps(parsed_jd, ensure_ascii=False),
        kb_context=kb_context,
    )
    fallback = _fallback_gap_diagnosis()

    try:
        return await llm_client.safe_chat_json(
            system_prompt=NODE4_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            scene='diagnosis',
            session_id=session_id,
            fallback=fallback,
        )
    except LLMError as e:
        logger.error(f'节点4 LLM异常: {str(e)}')
        return fallback
    except Exception as e:
        logger.error(f'节点4 未知异常: {str(e)}')
        return fallback


# ============================================================
# 节点5：简历优化（LLM）
# ============================================================

async def _node5_optimize_resume(
    original_resume: str,
    gap_diagnosis: dict,
    session_id: Optional[str] = None,
) -> dict:
    '''简历优化：针对差距诊断结果，生成逐条修改对比 + 完整优化版'''
    user_prompt = NODE5_USER_PROMPT_TEMPLATE.format(
        original_resume=original_resume,
        gap_diagnosis=json.dumps(gap_diagnosis, ensure_ascii=False),
    )
    fallback = _fallback_optimized()

    try:
        return await llm_client.safe_chat_json(
            system_prompt=NODE5_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            scene='optimize',
            session_id=session_id,
            fallback=fallback,
        )
    except LLMError as e:
        logger.error(f'节点5 LLM异常: {str(e)}')
        return fallback
    except Exception as e:
        logger.error(f'节点5 未知异常: {str(e)}')
        return fallback


# ============================================================
# 节点6：逆袭话术 + 面试题预测（LLM）
# ============================================================

async def _node6_counterattack(
    gap_diagnosis: dict,
    session_id: Optional[str] = None,
) -> dict:
    '''逆袭话术：基于 top3_fatal_gaps 生成 3 版话术 + 预测面试题'''
    user_prompt = NODE6_USER_PROMPT_TEMPLATE.format(
        gap_diagnosis=json.dumps(gap_diagnosis, ensure_ascii=False),
    )
    fallback = _fallback_counterattack()

    try:
        return await llm_client.safe_chat_json(
            system_prompt=NODE6_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            scene='counterattack',
            session_id=session_id,
            fallback=fallback,
        )
    except LLMError as e:
        logger.error(f'节点6 LLM异常: {str(e)}')
        return fallback
    except Exception as e:
        logger.error(f'节点6 未知异常: {str(e)}')
        return fallback


# ============================================================
# 降级兜底数据
# ============================================================

def _fallback_parsed_resume(resume_text: str) -> dict:
    '''节点1 降级：简历解析失败时的结构化兜底'''
    return {
        'education': {
            'school': None, 'major': None, 'gpa': None,
            'gpa_rank': None, 'degree': '本科', 'relevant_courses': [],
        },
        'experience': {'internships': [], 'student_orgs': []},
        'skills': {
            'hard_skills': [], 'soft_skills': [],
            'certificates': [], 'languages': [],
        },
        'honors': [],
        'summary': {
            'top_strength': '基于简历文本评估中',
            'top_weakness': '基于简历文本评估中',
            'career_fit': resume_text[:80],
        },
    }


def _fallback_parsed_jd(jd_text: str) -> dict:
    '''节点2 降级：JD 拆解失败时的结构化兜底'''
    return {
        'hard_requirements': {
            'education_level': '本科',
            'major_requirement': '相关专业',
            'required_certificates': [],
            'min_experience_months': 0,
        },
        'core_capabilities': {
            'hard_skills': ['Office办公软件'],
            'soft_skills': ['沟通协调'],
            'experience_type': '校招',
        },
        'bonus_items': {
            'preferred_skills': [],
            'preferred_experience': [],
            'culture_fit': ['团队合作'],
        },
        'priority_order': ['专业对口', '基础技能', '综合素质'],
        'jd_category': '校招',
    }


def _fallback_gap_diagnosis() -> dict:
    '''节点4 降级：差距诊断失败时的评分兜底（红黄绿灯结构化数据）'''
    return {
        'match_score': {
            'total': '75.0',
            'education_score': '70.0',
            'experience_score': '75.0',
            'skill_score': '80.0',
            'calculation_detail': '教育70×30% + 经验75×40% + 技能80×30% = 75.0',
        },
        'gap_analysis': {
            'green_items': [
                {'item': '基础条件达标', 'detail': '简历基本信息完整',
                 'evidence': '简历文本已提交'},
            ],
            'yellow_items': [
                {'item': '需进一步优化', 'detail': '建议补充量化成果',
                 'suggestion': '在经历描述中加入具体数字和规模'},
            ],
            'red_items': [
                {'item': '需重点关注', 'detail': '建议对照JD完善技能描述',
                 'urgency': '中'},
            ],
        },
        'top3_fatal_gaps': [
            {
                'gap': '简历信息不够充分',
                'why_fatal': 'HR无法全面评估候选人能力',
                'interview_question': '请详细描述你最相关的一段实习经历及具体成果',
                'counterattack_hint': '建议补充具体项目经验和量化数据',
            },
        ],
        'priority_fix_order': ['补充量化成果', '对齐JD关键词', '完善证书信息'],
    }


def _fallback_optimized() -> dict:
    '''节点5 降级：简历优化失败时的兜底'''
    return {
        'optimized_resume': '（AI优化暂不可用，请参考差距诊断建议自行修改）',
        'changes': [
            {
                'section': '整体',
                'original': '原始简历',
                'optimized': '请补充量化成果、对齐JD关键词',
                'reason': 'LLM优化服务暂不可用',
                'priority': '高',
            },
        ],
        'highlights_added': ['建议突出岗位相关技能'],
    }


def _fallback_counterattack() -> dict:
    '''节点6 降级：逆袭话术失败时的兜底'''
    return {
        'counterattack_scripts': [
            {
                'gap': '待补充',
                'interview_question': '请介绍你的相关经验',
                'script_with_experience': '（AI生成暂不可用，请参考差距诊断中的逆袭方向自行准备）',
                'script_no_experience': '（AI生成暂不可用，请坦诚说明并强调学习意愿）',
                'script_one_liner': '（AI生成暂不可用）',
            },
        ],
        'predicted_interview_questions': [
            {
                'question': '请做一个简短的自我介绍',
                'why_predicted': '面试常规问题',
                'answer_strategy': '突出与目标岗位相关的经历和能力',
            },
        ],
        'interview_tips': [
            '提前了解公司和岗位信息',
            '准备2-3个个人亮点案例',
            '保持自信、诚恳的态度',
        ],
    }


# ============================================================
# 兼容旧接口
# ============================================================

async def execute_gap_analysis(
    resume_text: str,
    jd_text: str,
    session_id: Optional[str] = None,
) -> dict:
    '''兼容 main.py 旧接口：构造临时 session 对象调用 run_gap_analysis'''
    from ..models.session import Session
    temp_session = Session(id=session_id or 'temp')
    return await run_gap_analysis(temp_session, resume_text, jd_text)
