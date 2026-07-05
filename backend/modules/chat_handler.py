'''职准星 - 对话接待与意图识别模块

对接 main.py 会话状态机，处理对话接待全流程：
  1. 用户输入安全过滤（长度限制 + 恶意内容过滤）
  2. 空消息 / 超短消息友好引导
  3. 重复开场缓存拦截（解决对话重复开场卡顿 Bug）
  4. LLM 意图识别（闲聊 / 启动测评 / 直接分析）
  5. GO_NEXT 信号提取 + 括号推理文字过滤
  6. 新会话自动发送引导话术
'''

import re
import logging
from typing import Optional

from ..core import config
from ..core.llm import llm_client, LLMError
from ..core.prompts import (
    CHAT_SYSTEM_PROMPT,
    CHAT_USER_PROMPT_TEMPLATE,
    CHAT_HISTORY_CONTEXT_TEMPLATE,
    CHAT_INTRO_DELIVERED_HINT,
    ASSESSMENT_QUESTIONS,
)
from ..models.session import Session

logger = logging.getLogger(__name__)


# ============================================================
# 核心入口
# ============================================================

async def handle_chat(session: Session, user_message: str) -> dict:
    """
    对话接待入口函数。

    接收会话上下文 + 用户输入，依次执行：
      1. 输入安全过滤
      2. 空消息 / 超短消息 兜底
      3. 直接分析意图检测（简历 + JD）
      4. 构建系统提示（含开场缓存拦截）
      5. 调用 LLMClient 生成回复
      6. 括号推理文字过滤
      7. GO_NEXT 意图提取

    Args:
        session:       当前会话（SQLModel）
        user_message:  用户输入文本

    Returns:
        {
            "reply":  str,   # 返回给用户的回复文本
            "action": "CHAT" | "START_ASSESSMENT" | "DIRECT_ANALYSIS"
        }
    """
    # ---- 1. 输入安全过滤 ----
    user_message = sanitize_input(user_message)

    # ---- 2. 空消息兜底 ----
    clean_msg = user_message.strip()
    if not clean_msg:
        return {
            'reply': '请输入你想了解的内容\u301c',
            'action': 'CHAT',
        }

    # 超短消息（1-2 个字且不是关键词）引导多说
    if len(clean_msg) <= 2 and clean_msg not in ('开始', '好的', '是', '嗯', '好'):
        return {
            'reply': '能多说一点吗？你的回答越详细，我的分析就越准确哦\u301c',
            'action': 'CHAT',
        }

    # ---- 3. 直接分析意图检测 ----
    direct = _detect_direct_analysis(clean_msg)
    if direct:
        return direct

    # ---- 4. 构建会话历史上下文 ----
    history = session.get_recent_history(config.CHAT_HISTORY_RECENT_N)
    history_text = ''
    for msg in history:
        role_label = '用户' if msg.get('role') == 'user' else '助手'
        history_text += f'{role_label}\uff1a{msg.get("content", "")}\n'

    # ---- 5. 开场缓存拦截 ----
    has_intro = session.has_intro_delivered()
    system_prompt = CHAT_SYSTEM_PROMPT
    if has_intro:
        system_prompt += CHAT_INTRO_DELIVERED_HINT
        logger.debug(f'会话 {session.id}: 开场已输出，跳过重复开场白')

    # ---- 6. 调用 LLM ----
    try:
        user_prompt = CHAT_HISTORY_CONTEXT_TEMPLATE.format(
            history_text=history_text,
            user_message=clean_msg,
        )
        result = await llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            scene='chat',
            session_id=session.id,
        )
        reply = result.get('output', '')
    except LLMError as e:
        logger.error(f'Chat LLM调用失败: {str(e)}')
        return {
            'reply': '网络开小差了，请重新发送\u301c',
            'action': 'CHAT',
        }

    # ---- 7. 过滤括号推理文字 ----
    reply = _strip_bracket_text(reply)

    # ---- 8. 意图提取 ----
    return _determine_action(reply)


# ============================================================
# 输入安全过滤
# ============================================================

def sanitize_input(text: str) -> str:
    """
    输入安全过滤与长度限制。

    过滤规则：
      - 空值返回空字符串
      - 超长（> config.MAX_MESSAGE_LENGTH）自动截断
      - 恶意外链替换为 [链接已过滤]
      - 首尾空白修剪
    """
    if not text:
        return ''

    # 长度限制
    if len(text) > config.MAX_MESSAGE_LENGTH:
        logger.info(f'输入超长（{len(text)}字），截断至{config.MAX_MESSAGE_LENGTH}')
        text = text[:config.MAX_MESSAGE_LENGTH]

    # 恶意内容过滤（外链、涉政、广告等）
    for pattern in config.HARMFUL_PATTERNS:
        text = re.sub(pattern, '[链接已过滤]', text)

    return text.strip()


# ============================================================
# 括号推理文字过滤
# ============================================================

def _strip_bracket_text(text: str) -> str:
    """
    移除 LLM 输出的括号包裹推理文字。

    规则：
      - 移除 (xxx) 圆括号内容，包括嵌套括号
      - 移除（xxx）中文括号内容
      - 保留括号内无文字的情况（如 emoji）

    解决 Coze 原扣子工作流中 LLM 输出后台推理文字的 Bug。
    """
    # 递归移除括号内容
    while True:
        new_text = re.sub(r'\([^()]*\)', '', text)
        new_text = re.sub(r'\uff08[^\uff08\uff09]*\uff09', '', new_text)
        if new_text == text:
            break
        text = new_text
    return text.strip()


# ============================================================
# 意图提取
# ============================================================

def _determine_action(llm_output: str) -> dict:
    """
    判断 LLM 输出是否包含 GO_NEXT 信号。

    规则（对应需求文档模块 A.7）：
      - 包含 "GO_NEXT" → 提取之前文本作为回复，action=START_ASSESSMENT
      - 不包含          → 直接作为回复，action=CHAT
    """
    if 'GO_NEXT' in llm_output:
        parts = llm_output.split('GO_NEXT')
        reply = parts[0].strip()
        if not reply:
            reply = '好的，我们开始吧\uff01'
        logger.info('意图识别: START_ASSESSMENT')
        return {'reply': reply, 'action': 'START_ASSESSMENT'}

    logger.debug('意图识别: CHAT（闲聊）')
    return {'reply': llm_output, 'action': 'CHAT'}


# ============================================================
# 直接分析检测
# ============================================================

def _detect_direct_analysis(user_message: str) -> Optional[dict]:
    """
    检测用户是否直接提供了简历 + JD 文本。

    检测逻辑：
      - 消息同时包含简历关键词 + JD 关键词
      - 或消息长度 > 300 且包含简历关键词
      - 单纯打招呼/闲聊不触发

    Returns:
      - dict  {reply, action: 'DIRECT_ANALYSIS'}  命中
      - None                                        未命中
    """
    resume_keywords = ['简历', '个人简历', '实习经历', '教育背景']
    jd_keywords = ['岗位职责', '任职要求', '岗位描述', 'JD', '职位描述']

    has_resume = any(kw in user_message for kw in resume_keywords)
    has_jd = any(kw in user_message for kw in jd_keywords)
    is_long = len(user_message) > 300

    if has_resume and (has_jd or is_long):
        logger.info('意图识别: DIRECT_ANALYSIS（直接分析）')
        return {
            'reply': '好的，我收到了你的简历和JD信息，正在为你准备分析...',
            'action': 'DIRECT_ANALYSIS',
        }

    return None


# ============================================================
# 新会话引导话术
# ============================================================

def get_welcome_message() -> str:
    '''新会话首次进入时展示的引导话术'''
    return '''你好呀\uff01欢迎来到职准星，专为在校应届生打造的AI求职教练\u3002

我可以帮你：
\ud83d\udccc **发现适合的岗位方向** \u2014 通过6道背景测评，推荐3个匹配岗位+配套JD
\ud83d\udccc **深度差距分析** \u2014 上传简历与目标JD，产出评分报告、优化方案、逆袭话术

你可以直接说「开始」进入测评，或者随便聊聊你的求职困惑\u301c'''
