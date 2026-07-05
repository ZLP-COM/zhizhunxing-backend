'''职准星 - 六问兴趣测评模块

实现霍兰德六问测评完整流程：
  1. 顺序控制 Q1 → Q2 → Q3 → Q4 → Q5 → Q6
  2. 每道题用户作答，缓存至 session.assessment_answers
  3. 调用 LLM 做单题性格倾向分析（scene=assessment）
  4. 支持中途退出恢复（从 session.current_question 继续）
  5. 全部作答完毕后输出 6 份分析 JSON，流转至综合推理模块
  6. 支持跳过（用户可跳过任意题目，置信度标记为低）
  7. 统一入口函数 run_assessment_step(session, user_message)
'''

import json
import logging
from typing import Optional, Any, Dict

from ..core import config
from ..core.llm import llm_client, LLMError
from ..core.prompts import (
    ASSESSMENT_QUESTIONS,
    ANALYZE_Q1_SYSTEM, ANALYZE_Q1_USER_TEMPLATE,
    ANALYZE_Q2_SYSTEM, ANALYZE_Q2_USER_TEMPLATE,
    ANALYZE_Q3_SYSTEM, ANALYZE_Q3_USER_TEMPLATE,
    ANALYZE_Q4_SYSTEM, ANALYZE_Q4_USER_TEMPLATE,
    ANALYZE_Q5_SYSTEM, ANALYZE_Q5_USER_TEMPLATE,
    ANALYZE_Q6_SYSTEM, ANALYZE_Q6_USER_TEMPLATE,
)
from ..models.session import Session

logger = logging.getLogger(__name__)


# ============================================================
# 分析 Prompt 映射表
# ============================================================

ANALYZE_MAP: Dict[str, tuple] = {
    'Q1': (ANALYZE_Q1_SYSTEM, ANALYZE_Q1_USER_TEMPLATE),
    'Q2': (ANALYZE_Q2_SYSTEM, ANALYZE_Q2_USER_TEMPLATE),
    'Q3': (ANALYZE_Q3_SYSTEM, ANALYZE_Q3_USER_TEMPLATE),
    'Q4': (ANALYZE_Q4_SYSTEM, ANALYZE_Q4_USER_TEMPLATE),
    'Q5': (ANALYZE_Q5_SYSTEM, ANALYZE_Q5_USER_TEMPLATE),
    'Q6': (ANALYZE_Q6_SYSTEM, ANALYZE_Q6_USER_TEMPLATE),
}


# ============================================================
# 降级兜底数据（LLM 调用失败 / 用户跳过时使用）
# ============================================================

FALLBACK_ANALYSIS: Dict[str, Dict[str, Any]] = {
    'Q1': {
        'dimension': None, 'confidence': '低',
        'evidence': '', 'reasoning': '用户未提供有效回答',
    },
    'Q2': {
        'excluded_dimensions': [], 'confidence': '低',
        'evidence': '', 'reasoning': '用户未提供有效回答',
    },
    'Q3': {
        'dimension': '中性', 'confidence': '低',
        'evidence': '', 'reasoning': '用户未提供有效回答',
    },
    'Q4': {
        'dimension': 'C', 'confidence': '低',
        'evidence': '', 'reasoning': '用户未提供有效回答',
    },
    'Q5': {
        'dimension': None, 'confidence': '低',
        'evidence': '', 'reasoning': '用户未提供有效回答',
    },
    'Q6': {
        'major': '未知', 'mapped_dimensions': [],
        'user_preferred_direction': None, 'reasoning': '用户未提供有效回答',
    },
}


# ============================================================
# 统一入口函数
# ============================================================

async def run_assessment_step(session: Session, user_message: str) -> dict:
    '''
    测评流程单步执行函数。

    接收当前会话上下文 + 用户本轮回答，自动判断当前进度并执行对应步骤。

    Args:
        session:       当前会话（含 current_question、assessment_answers 等字段）
        user_message:  用户本轮输入

    Returns:
        {
            'reply':         str,          # 返回给用户的回复文本（下个问题 / 过渡语 / 完成通知）
            'action':        str,          # 'NEXT_QUESTION' | 'COMPLETED' | 'REPEAT'
            'next_question': int | None,   # 下一个问题编号（1-6）
        }
    '''
    q_num = session.current_question  # 当前待答问题编号（1-6）

    # ---- 处理跳过 ----
    if _is_skip_intent(user_message):
        session.skipped_questions = (session.skipped_questions or []) + [f'Q{q_num}']
        session.assessment_answers = (session.assessment_answers or {})
        session.assessment_answers[f'Q{q_num}'] = ''
        analysis = dict(FALLBACK_ANALYSIS.get(f'Q{q_num}', {}))
        analysis['skipped'] = True
        session.assessment_results = (session.assessment_results or {})
        session.assessment_results[f'Q{q_num}'] = analysis
        logger.info(f'会话 {session.id}: 跳过 Q{q_num}')
        return await _advance_to_next(session, q_num)

    # ---- 短回答引导 ----
    if len(user_message.strip()) < 5:
        logger.info(f'会话 {session.id}: Q{q_num} 回答过短，引导多说')
        return {
            'reply': '能多说一点吗？你的回答越详细，我的分析就越准确哦\u301c',
            'action': 'REPEAT',
            'next_question': q_num,
        }

    # ---- 保存用户回答 ----
    session.assessment_answers = (session.assessment_answers or {})
    session.assessment_answers[f'Q{q_num}'] = user_message

    # ---- LLM 单题分析 ----
    q_id = f'Q{q_num}'
    analysis = await _analyze_single_question(q_id, user_message, session.id)

    # 保存分析结果
    session.assessment_results = (session.assessment_results or {})
    session.assessment_results[q_id] = analysis

    logger.info(
        f'会话 {session.id}: {q_id} 分析完成, '
        f'置信度={analysis.get("confidence", "N/A")}'
    )

    # ---- 推进到下一题或完成 ----
    return await _advance_to_next(session, q_num)


# ============================================================
# 单题 LLM 分析
# ============================================================

async def _analyze_single_question(question_id: str, answer: str, session_id: str) -> dict:
    '''
    调用 LLM 对单题回答做霍兰德维度分析。

    Args:
        question_id:  'Q1' ~ 'Q6'
        answer:       用户回答文本
        session_id:   会话 ID（限流 + 统计用）

    Returns:
        分析结果 dict（各题格式见 prompts.py 输出格式定义）
    '''
    system_prompt, user_template = ANALYZE_MAP.get(question_id, (None, None))
    if not system_prompt:
        logger.warning(f'未知问题ID: {question_id}，使用兜底')
        return dict(FALLBACK_ANALYSIS.get(question_id, {}))

    # 渲染 user prompt（占位符：q1_answer ~ q6_answer）
    q_key = f'q{question_id[-1]}_answer'
    user_prompt = user_template.format(**{q_key: answer})

    fallback = dict(FALLBACK_ANALYSIS.get(question_id, {}))

    try:
        result = await llm_client.safe_chat_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            scene='assessment',
            session_id=session_id,
            fallback=fallback,
        )
        return result
    except LLMError as e:
        logger.error(f'分析{question_id} LLM异常: {str(e)}')
        return fallback
    except Exception as e:
        logger.error(f'分析{question_id} 未知异常: {str(e)}')
        return fallback


# ============================================================
# 进度推进
# ============================================================

async def _advance_to_next(session: Session, current_q: int) -> dict:
    '''
    推进到下一题或通知测评完成。
    '''
    if current_q < 6:
        # 进入下一题
        next_q = current_q + 1
        session.current_question = next_q
        session.state = f'ASSESSMENT_Q{next_q}'
        next_text = get_question_text(next_q)
        logger.info(f'会话 {session.id}: Q{current_q} \u2192 Q{next_q}')
        return {
            'reply': f'收到，继续下一个问题\u301c\n\n{next_text}',
            'action': 'NEXT_QUESTION',
            'next_question': next_q,
        }

    # 全部 6 题完成
    session.current_question = 0
    session.state = 'REASONING'
    logger.info(f'会话 {session.id}: 六问测评全部完成，进入综合推理')
    return {
        'reply': '测评完成！正在为你分析结果，请稍等...',
        'action': 'COMPLETED',
        'next_question': None,
    }


# ============================================================
# 跳过检测
# ============================================================

def _is_skip_intent(text: str) -> bool:
    '''检测用户是否要跳过当前问题'''
    skip_keywords = {'跳过', '不想回答', '不想说', 'pass', '略过', '不想答', '下一个'}
    return text.strip().lower() in skip_keywords


# ============================================================
# 工具函数
# ============================================================

def get_question(question_id: int) -> Optional[dict]:
    '''获取指定编号的问题定义（含 id、text、node）'''
    if 1 <= question_id <= 6:
        return ASSESSMENT_QUESTIONS[question_id - 1]
    return None


def get_question_text(question_id: int) -> str:
    '''获取指定编号的问题文本'''
    q = get_question(question_id)
    return q['text'] if q else ''


def get_all_answers(session: Session) -> Dict[str, str]:
    '''获取当前会话全部已作答内容'''
    return session.assessment_answers or {}


def get_all_results(session: Session) -> Dict[str, Any]:
    '''获取当前会话全部分析结果'''
    return session.assessment_results or {}


def is_assessment_complete(session: Session) -> bool:
    '''检查六问测评是否全部完成'''
    results = session.assessment_results or {}
    return all(f'Q{i}' in results for i in range(1, 7))


def build_assessment_summary(session: Session) -> str:
    '''构建测评进度摘要（用于断点恢复时告知用户当前进度）'''
    answers = session.assessment_answers or {}
    done = len(answers)
    return f'测评进度：已完成 {done}/6 题，继续回答第 {done + 1} 题\u301c'


# ============================================================
# 兼容旧接口：analyze_answer
# ============================================================

async def analyze_answer(question_id: str, answer: str, session_id: str = None) -> dict:
    '''兼容 main.py 旧接口，调用 _analyze_single_question'''
    return await _analyze_single_question(question_id, answer, session_id)
