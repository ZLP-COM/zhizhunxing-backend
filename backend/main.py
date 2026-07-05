"""职准星 - FastAPI 主应用入口"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，支持从任何工作目录运行
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import os
import json
import re
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, Session as DBSession, create_engine, select

from backend.core import config
from backend.core.llm import llm_client
from backend.models.session import Session
from backend.models.schemas import (
    ChatRequest, ChatResponse, UploadResponse,
    NewSessionResponse, ErrorResponse, TokenStatsResponse,
)
from backend.modules.pdf_parser import safe_parse_pdf, PDFParseError
from backend.modules.kb_manager import kb_manager, init_knowledge_base
from backend.modules.chat_handler import handle_chat
from backend.modules.assessment import analyze_answer, get_question_text
from backend.modules.reasoning import execute_reasoning, format_recommendation
from backend.modules.kb_manager import search_jd_for_recommendations
from backend.modules.kb_manager import search_product_manual
from backend.modules.gap_analysis import execute_gap_analysis
from backend.modules.report_gen import (
    build_report_markdown,
    generate_docx_report,
    build_report_summary,
)
from backend.core.prompts import ASSESSMENT_QUESTIONS

# 日志配置
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# SQLite 数据库
DATABASE_URL = "sqlite:///./sessions.db"
engine = create_engine(DATABASE_URL, echo=False)

# 限流记录 {ip: [timestamp, ...]}
_rate_limit_records: Dict[str, list] = {}

# 报告缓存 {cache_key: filename}
_report_cache: Dict[str, str] = {}

# 差距分析进度 {session_id: {"progress": 0-100, "stage": str, "done": bool}}
_gap_progress: Dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    logger.info("职准星后端启动中...")

    # 1. 创建数据库表
    SQLModel.metadata.create_all(engine)

    # 2. 初始化知识库
    try:
        await init_knowledge_base(kb_manager)
    except Exception as e:
        logger.warning(f"知识库初始化失败（将使用降级方案）: {str(e)}")

    # 3. 清理过期会话
    asyncio.create_task(_periodic_cleanup())

    logger.info("职准星后端启动完成")
    yield

    # 关闭
    logger.info("职准星后端关闭")


app = FastAPI(
    title="职准星 API",
    description="校招AI求职教练后端服务",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# 中间件：全局错误处理 + 限流 + 隐私日志
# ============================================================

@app.middleware("http")
async def global_middleware(request: Request, call_next):
    """全局中间件：限流 + 错误处理 + 隐私保护"""

    # 限流检查（非静态文件请求）
    client_ip = request.client.host if request.client else "unknown"
    if not request.url.path.startswith("/static"):
        if not _check_rate_limit(client_ip):
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试", "code": "RATE_LIMITED"},
            )

    # 处理请求
    try:
        response = await call_next(request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"未捕获的异常: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务器内部错误，请稍后重试",
                "code": "INTERNAL_ERROR",
            },
        )


def _check_rate_limit(ip: str) -> bool:
    """单IP限流检查：每分钟最多10次"""
    now = datetime.now()
    if ip not in _rate_limit_records:
        _rate_limit_records[ip] = []

    # 清理1分钟前的记录
    _rate_limit_records[ip] = [
        t for t in _rate_limit_records[ip]
        if now - t < timedelta(minutes=1)
    ]

    if len(_rate_limit_records[ip]) >= config.RATE_LIMIT_PER_MINUTE:
        logger.warning(f"IP {ip} 触发限流")
        return False

    _rate_limit_records[ip].append(now)
    return True


async def _periodic_cleanup():
    """定时清理过期会话和文件"""
    while True:
        try:
            await asyncio.sleep(3600)  # 每小时执行一次
            await _cleanup_expired_sessions()
        except Exception as e:
            logger.error(f"定时清理任务异常: {str(e)}")


async def _cleanup_expired_sessions():
    """清理过期会话及关联文件"""
    with DBSession(engine) as db_session:
        cutoff = datetime.now() - timedelta(hours=config.MAX_SESSION_HOURS)
        expired = db_session.exec(
            select(Session).where(Session.updated_at < cutoff)
        ).all()

        for session in expired:
            # 清理上传文件
            if session.resume_file_name:
                file_path = os.path.join(config.UPLOAD_DIR, session.resume_file_name)
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"清理上传文件: {file_path}")

            # 清理报告文件
            if session.report_filename:
                report_path = os.path.join(config.REPORT_DIR, session.report_filename)
                if os.path.exists(report_path):
                    os.remove(report_path)
                    logger.info(f"清理报告文件: {report_path}")

            db_session.delete(session)
            logger.info(f"清理过期会话: {session.id}")

        db_session.commit()
        if expired:
            logger.info(f"本次清理 {len(expired)} 个过期会话")


# ============================================================
# 辅助函数
# ============================================================

def get_db_session():
    """获取数据库会话"""
    with DBSession(engine) as session:
        yield session


def _get_session_or_404(session_id: str) -> Session:
    """获取会话，不存在则抛出404"""
    with DBSession(engine) as db:
        session = db.get(Session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在或已过期")
        # 检查会话是否超时
        if datetime.now() - session.updated_at > timedelta(hours=config.MAX_SESSION_HOURS):
            db.delete(session)
            db.commit()
            raise HTTPException(status_code=404, detail="会话已过期，请重新开始")
        return session


def _save_session(session: Session):
    """保存会话到数据库"""
    with DBSession(engine) as db:
        session.updated_at = datetime.now()
        db.merge(session)
        db.commit()


# ============================================================
# API 路由
# ============================================================

@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "service": "职准星",
        "version": "1.0.0",
        "kb_available": kb_manager.is_available(),
    }


@app.post("/api/session", response_model=NewSessionResponse)
async def create_session():
    """新建会话"""
    session = Session()
    with DBSession(engine) as db:
        db.add(session)
        db.commit()
        db.refresh(session)

    logger.info(f"新建会话: {session.id}")
    return NewSessionResponse(
        session_id=session.id,
        state=session.state,
        created_at=session.created_at.isoformat(),
        message="你好呀！欢迎来到职准星，专为在校应届生打造的AI求职教练。",
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """获取会话状态"""
    session = _get_session_or_404(session_id)
    return session.to_dict()


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """对话接口"""
    session = _get_session_or_404(request.session_id)
    user_message = request.message.strip()

    # 附加文件处理（拖拽上传的PDF）
    attachments = request.attachments
    resume_text_from_attach = None

    if attachments and len(attachments) > 0:
        for att in attachments:
            file_path = att.get("file_path") or att.get("path", "")
            if file_path and os.path.exists(file_path):
                try:
                    resume_text_from_attach = safe_parse_pdf(file_path)
                    session.final_resume = resume_text_from_attach
                    session.resume_source = "pdf"
                    logger.info(f"从附件解析简历: {file_path}")
                except PDFParseError as e:
                    return ChatResponse(
                        session_id=session.id,
                        reply=str(e),
                        state=session.state,
                        action="PDF_PARSE_FAILED",
                    )

    # 保存用户消息到历史
    session.add_chat_history("user", user_message)

    # ---- 状态机处理 ----
    current_state = session.state

    # 任何状态下检测"开始测评"意图，重置并重新进入测评
    if current_state not in ("IDLE", "ASSESSMENT_Q1", "ASSESSMENT_Q2", "ASSESSMENT_Q3",
                             "ASSESSMENT_Q4", "ASSESSMENT_Q5", "ASSESSMENT_Q6", "REASONING"):
        assess_triggers = ["开始测评", "重新测评", "再做一次", "开始测试", "测评"]
        if any(t in user_message for t in assess_triggers):
            # 重置会话为测评状态
            session.state = "IDLE"
            session.current_question = 0
            session.assessment_answers = {}
            session.assessment_results = {}
            session.reasoning_result = {}
            session.selected_jd_list = []
            session.formatted_recommendation = ""
            session.selected_direction = ""
            session.selected_jd = ""
            session.final_jd = ""
            session.final_resume = ""
            session.gap_analysis_result = {}
            session.optimized_resume_result = {}
            session.counterattack_result = {}
            session.report_filename = ""
            session.skipped_questions = []
            _save_session(session)

            # 重新进入 IDLE 状态处理流程
            result = await handle_chat(session, user_message)
            reply = result["reply"]
            action = result["action"]
            if action == "START_ASSESSMENT":
                with DBSession(engine) as db:
                    db_session = db.get(Session, session.id)
                    if db_session:
                        db_session.state = "ASSESSMENT_Q1"
                        db_session.current_question = 1
                        db.commit()
                session.state = "ASSESSMENT_Q1"
                session.current_question = 1
                q1_text = get_question_text(1)
                reply = f"{reply}\n\n{q1_text}"
            session.add_chat_history("system", reply)
            _save_session(session)
            return ChatResponse(
                session_id=session.id, reply=reply, state=session.state, action=action,
            )

    try:
        if current_state == "IDLE":
            # 闲聊/意图识别
            result = await handle_chat(session, user_message)
            reply = result["reply"]
            action = result["action"]

            if action == "START_ASSESSMENT":
                # 进入测评 - 直接写数据库，避免 detached 对象问题
                with DBSession(engine) as db:
                    db_session = db.get(Session, session.id)
                    if db_session:
                        db_session.state = "ASSESSMENT_Q1"
                        db_session.current_question = 1
                        db.commit()
                session.state = "ASSESSMENT_Q1"
                session.current_question = 1
                q1_text = get_question_text(1)
                reply = f"{reply}\n\n{q1_text}"

            elif action == "DIRECT_ANALYSIS":
                # 直接分析模式
                session.state = "WAITING_JD_CUSTOM"
                reply = "好的，请先粘贴目标岗位的JD文本，然后我再分析你的简历。"

            session.add_chat_history("system", reply)
            _save_session(session)

            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
                action=action,
            )

        elif current_state.startswith("ASSESSMENT_Q"):
            # 测评问题回答
            q_num = int(current_state[-1])
            q_id = f"Q{q_num}"

            # 检查跳过
            if user_message.strip() in ["跳过", "不想回答", "不想说", "pass", "略过"]:
                session.skipped_questions = (session.skipped_questions or []) + [q_id]
                session.assessment_answers = (session.assessment_answers or {})
                session.assessment_answers[q_id] = ""
                analysis_result = {"dimension": None, "confidence": "低", "skipped": True}
            else:
                # 保存回答
                session.assessment_answers = (session.assessment_answers or {})
                session.assessment_answers[q_id] = user_message

                # LLM分析
                analysis_result = await analyze_answer(q_id, user_message, session.id)

                # 处理需要追问的情况
                if analysis_result.get("reply_needed"):
                    reply = analysis_result.get("reply", "能再说说吗？")
                    session.add_chat_history("system", reply)
                    _save_session(session)
                    return ChatResponse(
                        session_id=session.id,
                        reply=reply,
                        state=session.state,
                    )

            # 保存分析结果
            session.assessment_results = (session.assessment_results or {})
            session.assessment_results[q_id] = analysis_result

            # 进入下一个问题
            if q_num < 6:
                next_state = f"ASSESSMENT_Q{q_num + 1}"
                session.state = next_state
                session.current_question = q_num + 1
                next_q_text = get_question_text(q_num + 1)
                reply = f"收到，继续下一个问题～\n\n{next_q_text}"
            else:
                # 6个问题已答完 → 自动执行综合推理（Fix A：不用用户再问）
                session.state = "REASONING"
                session.current_question = 0
                session.add_chat_history("system", "测评完成！正在为你分析结果，请稍等...")
                _save_session(session)
                logger.info(f"会话 {session.id}: Q6完成，开始自动推理")

                # 直接执行推理
                results = session.assessment_results or {}
                try:
                    reasoning_result = await execute_reasoning(
                        q1_result=results.get("Q1", {}),
                        q2_result=results.get("Q2", {}),
                        q3_result=results.get("Q3", {}),
                        q4_result=results.get("Q4", {}),
                        q5_result=results.get("Q5", {}),
                        q6_result=results.get("Q6", {}),
                        session_id=session.id,
                    )
                    session.reasoning_result = reasoning_result
                    logger.info(f"会话 {session.id}: 推理完成，holland={reasoning_result.get('holland_code','?')}")

                    jd_results = await search_jd_for_recommendations(reasoning_result)
                    session.selected_jd_list = jd_results
                    logger.info(f"会话 {session.id}: JD检索完成，共{len(jd_results)}个方向")

                    formatted = await format_recommendation(reasoning_result, jd_results, session.id)
                    session.formatted_recommendation = formatted
                    session.state = "SHOWING_RECOMMENDATIONS"
                    reply = formatted
                    logger.info(f"会话 {session.id}: 格式化推荐完成，reply长度={len(reply)}")
                except Exception as e:
                    logger.error(f"会话 {session.id}: 综合推理异常: {str(e)}", exc_info=True)
                    reply = "分析过程遇到了一点小问题，请重新开始测评。"
                    session.state = "IDLE"

            session.add_chat_history("system", reply)
            _save_session(session)

            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
            )

        elif current_state == "REASONING":
            # 综合推理中（异步执行）
            # 先返回一个"处理中"状态，实际推理在后台执行
            results = session.assessment_results or {}

            try:
                reasoning_result = await execute_reasoning(
                    q1_result=results.get("Q1", {}),
                    q2_result=results.get("Q2", {}),
                    q3_result=results.get("Q3", {}),
                    q4_result=results.get("Q4", {}),
                    q5_result=results.get("Q5", {}),
                    q6_result=results.get("Q6", {}),
                    session_id=session.id,
                )
                session.reasoning_result = reasoning_result

                # 知识库检索JD
                jd_results = await search_jd_for_recommendations(reasoning_result)
                session.selected_jd_list = jd_results

                # 格式化推荐
                formatted = await format_recommendation(reasoning_result, jd_results, session.id)
                session.formatted_recommendation = formatted

                # 进入展示推荐状态
                session.state = "SHOWING_RECOMMENDATIONS"
                reply = formatted

            except Exception as e:
                logger.error(f"综合推理异常: {str(e)}")
                reply = "分析过程遇到了一点小问题，请重新开始测评。"
                session.state = "IDLE"

            session.add_chat_history("system", reply)
            _save_session(session)

            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
            )

        elif current_state == "SHOWING_RECOMMENDATIONS":
            # 用户选择岗位方向
            recs = session.reasoning_result.get("recommendations", [])
            selected = _parse_user_choice(user_message, recs)

            if selected is None:
                reply = "没有识别到你选择的方向，请回复序号（1/2/3）或方向名称。\n\n" + session.formatted_recommendation
                session.add_chat_history("system", reply)
                _save_session(session)
                return ChatResponse(
                    session_id=session.id,
                    reply=reply,
                    state=session.state,
                )

            # 保存选定方向
            session.selected_direction = selected.get("job_direction", "")
            session.selected_direction_index = selected.get("rank", 1) - 1

            # 获取该方向的JD全文
            jd_list_for_selected = []
            for jd_res in session.selected_jd_list:
                if jd_res.get("job_direction") == selected.get("job_direction"):
                    jd_list_for_selected = jd_res.get("jd_list", [])
                    break

            if jd_list_for_selected:
                # 取第一个JD
                first_jd = jd_list_for_selected[0]
                session.selected_jd = first_jd.get("content", "")

                # 展示JD + 二选一
                jd_display = f"""📋 你选择的岗位方向：{selected.get('job_direction', '')}

匹配度：{selected.get('match_score', '?')}%
匹配理由：{selected.get('match_reason', '')}

📄 配套JD：

公司：{first_jd.get('company', '未知')}
岗位：{first_jd.get('title', '未知')}

{first_jd.get('content', '(JD内容)')[:2000]}

---
你可以：
A. 直接使用这份JD进行分析（只需发送简历）
B. 粘贴你自己找到的JD（需同时发送JD和简历）

请回复 A 或 B"""
            else:
                # 即使没有 JD 全文，也设置占位 JD 用于后续分析
                placeholder_jd = f"岗位方向：{selected.get('job_direction', '')}。{selected.get('match_reason', '')}"
                session.selected_jd = placeholder_jd
                jd_display = f"""📋 你选择的岗位方向：{selected.get('job_direction', '')}

匹配度：{selected.get('match_score', '?')}%
匹配理由：{selected.get('match_reason', '')}

---
你可以：
A. 使用系统推荐的JD进行分析（只需发送简历）
B. 粘贴你自己找到的JD（需同时发送JD和简历）

请回复 A 或 B"""

            session.state = "JD_CHOICE"
            reply = jd_display
            session.add_chat_history("system", reply)
            _save_session(session)

            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
            )

        elif current_state == "JD_CHOICE":
            # 二选一交互
            choice = user_message.strip().upper()

            if choice in ["A", "A.", "用这个", "使用系统", "就这个"]:
                # 选项A：使用系统JD
                session.jd_source = "system"
                session.final_jd = session.selected_jd
                session.state = "WAITING_RESUME_SYSTEM"
                reply = "好的，使用系统JD！请发送你的简历（支持上传PDF或直接粘贴文本）"

            elif choice in ["B", "B.", "换一个", "自己找", "我用其他"]:
                # 选项B：使用外部JD
                session.jd_source = "custom"
                session.state = "WAITING_JD_CUSTOM"
                reply = "好的，请粘贴你找到的JD文本："

            else:
                reply = "请回复 A（使用系统JD）或 B（粘贴外部JD）"
                session.add_chat_history("system", reply)
                _save_session(session)
                return ChatResponse(
                    session_id=session.id,
                    reply=reply,
                    state=session.state,
                )

            session.add_chat_history("system", reply)
            _save_session(session)

            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
            )

        elif current_state == "WAITING_JD_CUSTOM":
            # 等待用户粘贴外部JD
            if len(user_message) < 50:
                reply = "JD内容太少了，请确认是否完整粘贴了JD文本？"
                session.add_chat_history("system", reply)
                _save_session(session)
                return ChatResponse(
                    session_id=session.id,
                    reply=reply,
                    state=session.state,
                )

            session.custom_jd = user_message[:config.MAX_JD_LENGTH]
            session.final_jd = session.custom_jd
            session.state = "WAITING_RESUME_CUSTOM"
            reply = "已收到你的JD！现在请发送你的简历（支持上传PDF或直接粘贴文本）"
            session.add_chat_history("system", reply)
            _save_session(session)

            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
            )

        elif current_state in ["WAITING_RESUME_SYSTEM", "WAITING_RESUME_CUSTOM"]:
            # 等待简历：检查是PDF上传还是文本粘贴
            # 如果有附件中的简历文本，直接使用
            if resume_text_from_attach:
                session.final_resume = resume_text_from_attach
                session.resume_source = "pdf"
            else:
                # 文本粘贴
                if len(user_message) < 100:
                    reply = "简历内容太少了，请确认是否完整粘贴了简历文本？"
                    session.add_chat_history("system", reply)
                    _save_session(session)
                    return ChatResponse(
                        session_id=session.id,
                        reply=reply,
                        state=session.state,
                    )
                session.final_resume = user_message[:config.MAX_RESUME_LENGTH]
                session.resume_source = "text"

            # 保存简历，启动后台分析任务
            session.state = "GAP_ANALYSIS"
            reply = "已收到简历！正在进行分析，我会以对话形式通知你结果..."
            session.add_chat_history("system", reply)
            _save_session(session)

            # 启动后台任务
            asyncio.create_task(_background_gap_analysis(session.id))

            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
            )

        elif current_state == "GAP_ANALYSIS":
            # 检查后台分析是否完成
            gap_data = session.gap_analysis_result or {}
            if gap_data.get("match_score"):
                # 已完成，返回报告
                report_url = f"/api/report/{session.report_filename}" if session.report_filename else None
                summary = build_report_summary(gap_data)

                if report_url:
                    # 有链接：摘要 + 下载链接
                    reply = summary + f"\n📎 [下载完整报告]({report_url})"
                else:
                    # 无链接：全量文本输出（评分 + 优化 + 逆袭话术全部写出来）
                    opt_data = session.optimized_resume_result or {}
                    cta_data = session.counterattack_result or {}

                    # 逆袭话术
                    cta_text = ""
                    scripts = cta_data.get("counterattack_scripts", [])
                    if scripts:
                        cta_text = "\n\n### 🎯 面试逆袭话术\n\n"
                        for s in scripts[:3]:
                            cta_text += "**" + s.get("gap", "短板") + "**\n"
                            if s.get("interview_question"):
                                cta_text += "面试官可能问：" + s["interview_question"] + "\n"
                            if s.get("script_with_experience"):
                                cta_text += "✅ 有经验版：" + s["script_with_experience"] + "\n"
                            if s.get("script_no_experience"):
                                cta_text += "💡 无经验版：" + s["script_no_experience"] + "\n"
                            if s.get("script_one_liner"):
                                cta_text += "⚡ 一句话版：" + s["script_one_liner"] + "\n"
                            cta_text += "\n"

                    # 简历优化
                    opt_text = ""
                    changes = opt_data.get("changes", [])
                    if changes:
                        opt_text = "\n\n### 📝 简历优化建议\n\n"
                        for c in changes[:5]:
                            opt_text += "- **" + c.get("section", "") + "**：" + c.get("reason", "") + "\n"
                            if c.get("original"):
                                opt_text += "  原句：" + c["original"] + "\n"
                            if c.get("optimized"):
                                opt_text += "  优化：" + c["optimized"] + "\n"

                    # 预测面试题
                    interview_text = ""
                    predicted = cta_data.get("predicted_interview_questions", [])
                    if predicted:
                        interview_text = "\n\n### 🔮 预测面试题\n\n"
                        for q in predicted[:3]:
                            interview_text += "**" + q.get("question", "") + "**\n"
                            if q.get("answer_strategy"):
                                interview_text += "答题策略：" + q["answer_strategy"] + "\n"
                            interview_text += "\n"

                    reply = summary + opt_text + cta_text + interview_text

                session.state = "REPORT_READY"
                _save_session(session)
                return ChatResponse(
                    session_id=session.id, reply=reply, state=session.state,
                    action="REPORT_READY", report_url=report_url,
                )
            else:
                # 还在分析中
                reply = "分析仍在进行中（约需要 30-60 秒），请稍后再问～"
                session.add_chat_history("system", reply)
                _save_session(session)
                return ChatResponse(
                    session_id=session.id, reply=reply, state=session.state,
                )

        elif current_state == "REPORT_READY":
            # 报告就绪后的对话
            result = await handle_chat(session, user_message)
            reply = result["reply"]
            session.add_chat_history("system", reply)
            _save_session(session)
            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
            )

        else:
            # 未知状态，重置
            session.state = "IDLE"
            reply = "好像出了点小问题，我们重新开始吧！"
            session.add_chat_history("system", reply)
            _save_session(session)
            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state="IDLE",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"对话处理异常: {str(e)}", exc_info=True)
        return ChatResponse(
            session_id=session.id,
            reply="系统出了点小问题，请稍后重试～",
            state=session.state or "IDLE",
        )


@app.post("/api/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    """文件上传接口（PDF简历）"""
    # 文件类型检查
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="目前仅支持PDF格式")

    # 保存文件
    os.makedirs(config.UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(config.UPLOAD_DIR, f"{session_id}_{file.filename}")

    content = await file.read()

    # 文件大小检查
    if len(content) > config.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件过大（超过10MB限制）",
        )

    with open(file_path, "wb") as f:
        f.write(content)

    # 解析PDF
    try:
        resume_text = safe_parse_pdf(file_path)

        # 保存到会话
        with DBSession(engine) as db:
            session = db.get(Session, session_id)
            if session:
                session.final_resume = resume_text
                session.resume_source = "pdf"
                session.resume_file_name = f"{session_id}_{file.filename}"
                session.updated_at = datetime.now()
                db.merge(session)
                db.commit()

        return UploadResponse(
            success=True,
            resume_text=resume_text,
            file_name=file.filename,
            file_size=len(content),
        )

    except PDFParseError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"上传处理异常: {str(e)}")
        raise HTTPException(status_code=500, detail="文件处理失败")


@app.get("/api/report/{filename}")
async def download_report(filename: str):
    """报告下载接口"""
    # 安全校验：防止路径穿越
    if ".." in filename or "/" in filename:
        raise HTTPException(status_code=400, detail="非法的文件名")

    file_path = os.path.join(config.REPORT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="报告不存在或已过期")

    return FileResponse(
        file_path,
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=filename,
    )


@app.post("/api/session/{session_id}/reset")
async def reset_session(session_id: str):
    """重置会话"""
    with DBSession(engine) as db:
        session = db.get(Session, session_id)
        if not session:
            raise HTTPException(status_code=404, detail="会话不存在")

        # 重置所有字段
        session.state = "IDLE"
        session.current_question = 0
        session.assessment_answers = {}
        session.assessment_results = {}
        session.reasoning_result = {}
        session.formatted_recommendation = ""
        session.selected_direction = ""
        session.selected_jd = ""
        session.selected_jd_list = []
        session.jd_source = ""
        session.custom_jd = ""
        session.final_jd = ""
        session.final_resume = ""
        session.resume_source = ""
        session.gap_analysis_result = {}
        session.optimized_resume_result = {}
        session.counterattack_result = {}
        session.report_filename = ""
        session.skipped_questions = []
        session.updated_at = datetime.now()

        db.merge(session)
        db.commit()

    return {"message": "会话已重置", "session_id": session_id, "state": "IDLE"}


@app.get("/api/tokens/{session_id}")
async def get_token_stats(session_id: str):
    """获取Token消耗统计（调试接口）"""
    return TokenStatsResponse(
        session_id=session_id,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_calls=0,
    )


# ============================================================
# 辅助函数
# ============================================================

def judge_is_function_question(user_text: str) -> bool:
    '''
    判断用户提问是否为产品功能类问题（需检索产品说明书答疑）。

    内置正则关键词库，匹配功能相关词汇返回 True，
    普通闲聊/测评启动意图返回 False。

    Args:
        user_text: 用户输入文本

    Returns:
        True = 功能类问题，需检索产品说明书
        False = 普通闲聊，走原有对话逻辑
    '''
    text = user_text.strip()
    if not text:
        return False

    # 功能类关键词库（按优先级排列）
    function_keywords = [
        # 核心功能
        r'PDF', r'上传', r'简历上传', r'文件上传',
        r'测评', r'六问', r'霍兰德', r'兴趣测评', r'背景调研',
        r'报告', r'下载', r'导出', r'docx', r'差距分析',
        r'报错', r'错误', r'失败', r'不生效',
        # 操作指南
        r'怎么操作', r'如何.*使用', r'怎么用', r'怎么.*弄',
        r'功能', r'按钮', r'点击', r'操作步骤',
        r'扫描', r'缓存', r'清理', r'清除',
        r'部署', r'安装', r'启动', r'配置',
        r'恢复进度', r'重来', r'重置', r'重新开始',
        # 限制与规范
        r'登录', r'注册', r'账号',
        r'文件大小', r'大小限制', r'上限',
        r'限流', r'频率限制',
        r'会话', r'超时', r'过期', r'有效期',
        r'向量库', r'知识库', r'ChromaDB',
        r'API', r'Key', r'密钥', r'模型',
        r'隐私', r'数据安全', r'脱敏', r'清理',
        # 问题咨询
        r'这是做什么的', r'能干什么', r'有什么用',
        r'怎么收费', r'免费', r'费用',
        r'支持.*格式', r'支持.*类型',
    ]

    for pattern in function_keywords:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    return False


# ============================================================
# 产品说明书答疑接口
# ============================================================

PRODUCT_MANUAL_SYSTEM_PROMPT = '''\
你是「职准星」的产品说明书智能助理。请根据下面提供的产品文档内容回答用户提问。

【回答规则】
1. 严格依据提供的产品文档原文回答，禁止编造未实现的功能；
2. 若文档中没有用户询问的内容，如实告知"文档中没有相关说明"，不要杜撰；
3. 回答简洁清晰，用中文回答，使用适当的分点和emoji使答案易读；
4. 如果用户问的是如何使用某个功能，请给出具体的操作步骤；
5. 如果文档中包含相关章节标题和内容，请引用章节名称。

【产品文档片段】
{manual_context}

【用户提问】
{user_message}

请根据上述文档片段回答用户问题。如果文档片段不包含答案，请如实告知。
'''


@app.post('/api/chat_manual')
async def chat_manual(request: ChatRequest):
    '''
    产品功能答疑接口。

    分流逻辑：
      1. 调用 judge_is_function_question() 判断提问类型
      2. 普通闲聊 → 调用原有 handle_chat()，返回标准对话
      3. 功能类问题 → 检索产品说明书 → LLM 生成回复

    向量库故障时自动降级为 README 关键词匹配。
    '''
    session = _get_session_or_404(request.session_id)
    user_message = request.message.strip()

    # 保存用户消息到历史
    session.add_chat_history('user', user_message)

    # ---- 判断提问类型 ----
    is_function_q = judge_is_function_question(user_message)

    if not is_function_q:
        # ---- 普通闲聊：走原有对话逻辑 ----
        logger.debug(f'会话 {session.id}: 普通闲聊，不走产品说明书')
        try:
            result = await handle_chat(session, user_message)
            reply = result['reply']
            session.add_chat_history('system', reply)
            _save_session(session)
            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
                action=result.get('action'),
            )
        except Exception as e:
            logger.error(f'闲聊处理异常: {str(e)}')
            return ChatResponse(
                session_id=session.id,
                reply='网络开小差了，请重新发送\u301c',
                state=session.state,
            )

    # ---- 功能类问题：检索产品说明书 ----
    logger.info(f'会话 {session.id}: 功能类问题，检索产品说明书')
    try:
        # 检索说明书
        manual_context = await search_product_manual(user_message, n_results=3)

        if not manual_context:
            reply = '抱歉，我在产品说明书中没有找到相关的内容，请换个问法试试～'
            session.add_chat_history('system', reply)
            _save_session(session)
            return ChatResponse(
                session_id=session.id,
                reply=reply,
                state=session.state,
            )

        # 构造 LLM 请求
        system_prompt = PRODUCT_MANUAL_SYSTEM_PROMPT.format(
            manual_context=manual_context,
            user_message=user_message,
        )

        llm_result = await llm_client.chat(
            system_prompt=system_prompt,
            user_prompt=f'用户问题：{user_message}\n\n请根据上述产品文档内容回答。',
            temperature=0.5,
            max_tokens=1500,
            session_id=session.id,
        )
        reply = llm_result.get('output', '')
        if not reply:
            reply = '抱歉，我暂时无法回答这个问题，请稍后重试～'

    except Exception as e:
        logger.error(f'产品说明书答疑异常: {str(e)}')
        # 降级：直接返回检索片段
        try:
            manual_context = await search_product_manual(user_message)
            if manual_context:
                reply = f'根据产品说明书，我找到以下相关信息：\n\n{manual_context[:1500]}'
            else:
                reply = '抱歉，我没有找到相关的答案，请换个问题试试～'
        except Exception:
            reply = '服务暂时不可用，请稍后重试～'

    session.add_chat_history('system', reply)
    _save_session(session)

    return ChatResponse(
        session_id=session.id,
        reply=reply,
        state=session.state,
    )


def _parse_user_choice(user_input: str, recommendations: list) -> dict:
    """解析用户选择的岗位方向"""
    user_input = user_input.strip()

    # 尝试序号匹配
    if user_input.isdigit():
        idx = int(user_input) - 1
        if 0 <= idx < len(recommendations):
            return recommendations[idx]

    # 尝试名称模糊匹配
    for rec in recommendations:
        direction = rec.get("job_direction", "")
        if direction in user_input or user_input in direction:
            return rec

    return None


# ---------------------------------------------------------------------------
# 后台差距分析任务
# ---------------------------------------------------------------------------

async def _background_gap_analysis(session_id: str):
    """后台执行差距分析，更新进度，结果写入数据库"""
    _gap_progress[session_id] = {"progress": 0, "stage": "准备中...", "done": False}
    try:
        with DBSession(engine) as db:
            session = db.get(Session, session_id)
            if not session:
                _gap_progress[session_id]["done"] = True
                return
            resume_text = session.final_resume
            jd_text = session.final_jd

        if not resume_text or not jd_text:
            _gap_progress[session_id] = {"progress": 100, "stage": "数据不足", "done": True}
            return

        _gap_progress[session_id] = {"progress": 5, "stage": "解析简历与JD...", "done": False}
        results = await execute_gap_analysis(
            resume_text=resume_text, jd_text=jd_text, session_id=session_id,
        )

        _gap_progress[session_id] = {"progress": 80, "stage": "生成报告...", "done": False}
        with DBSession(engine) as db:
            session = db.get(Session, session_id)
            if not session:
                _gap_progress[session_id]["done"] = True
                return
            session.gap_analysis_result = results.get("gap_data", {})
            session.optimized_resume_result = results.get("optimized_data", {})
            session.counterattack_result = results.get("counterattack_data", {})
            md = build_report_markdown(
                gap_data=session.gap_analysis_result,
                opt_data=session.optimized_resume_result,
                cta_data=session.counterattack_result,
            )
            fp = generate_docx_report(md)
            session.report_filename = os.path.basename(fp)
            session.report_generated_at = datetime.now()
            db.commit()

        _gap_progress[session_id] = {"progress": 100, "stage": "完成", "done": True}
        logger.info(f"后台差距分析完成: 会话={session_id}")
    except Exception as e:
        logger.error(f"后台差距分析异常: 会话={session_id} {str(e)}", exc_info=True)
        _gap_progress[session_id] = {"progress": 100, "stage": "失败", "done": True}


async def _execute_gap_analysis_now(session):
    """直接执行差距分析，返回含评分+简历优化+面试逆袭话术的完整回复"""
    resume_text = session.final_resume
    jd_text = session.final_jd
    if not resume_text or not jd_text:
        session.state = "IDLE"
        reply = "简历或JD数据不完整，请重新开始～"
        session.add_chat_history("system", reply)
        _save_session(session)
        return ChatResponse(session_id=session.id, reply=reply, state=session.state)
    try:
        cache_key = hashlib.md5((resume_text[:200] + jd_text[:200]).encode()).hexdigest()
        gap_res = None
        if cache_key in _report_cache:
            cf = _report_cache[cache_key]
            cp = os.path.join(config.REPORT_DIR, cf)
            if os.path.exists(cp):
                session.report_filename = cf
                gap_res = session.gap_analysis_result
            else:
                del _report_cache[cache_key]
        if not gap_res:
            results = await execute_gap_analysis(
                resume_text=resume_text, jd_text=jd_text, session_id=session.id
            )
            session.gap_analysis_result = results.get("gap_data", {})
            session.optimized_resume_result = results.get("optimized_data", {})
            session.counterattack_result = results.get("counterattack_data", {})
            md = build_report_markdown(
                gap_data=session.gap_analysis_result,
                opt_data=session.optimized_resume_result,
                cta_data=session.counterattack_result,
            )
            fp = generate_docx_report(md)
            session.report_filename = os.path.basename(fp)
            session.report_generated_at = datetime.now()
            _report_cache[cache_key] = session.report_filename
        gap_data = session.gap_analysis_result or {}
        opt_data = session.optimized_resume_result or {}
        cta_data = session.counterattack_result or {}
        summary = build_report_summary(gap_data)
        report_url = f"/api/report/{session.report_filename}"
        cta_text = ""
        scripts = cta_data.get("counterattack_scripts", [])
        if scripts:
            cta_text = "\n\n### 面试逆袭话术\n\n"
            for s in scripts[:3]:
                cta_text += "**" + s.get("gap", "短板") + "**\n"
                if s.get("interview_question"):
                    cta_text += "面试官可能问：" + s["interview_question"] + "\n"
                if s.get("script_with_experience"):
                    cta_text += "有经验版：" + s["script_with_experience"] + "\n"
                if s.get("script_no_experience"):
                    cta_text += "无经验版：" + s["script_no_experience"] + "\n"
                cta_text += "\n"
        opt_text = ""
        changes = opt_data.get("changes", [])
        if changes:
            opt_text = "\n\n### 简历优化建议\n\n"
            for c in changes[:5]:
                opt_text += "- **" + c.get("section", "") + "**：" + c.get("reason", "") + "\n"
        full_reply = summary + opt_text + cta_text + "\n\n---\n📎 [下载完整报告](" + report_url + ")"
        session.state = "REPORT_READY"
        _save_session(session)
        return ChatResponse(
            session_id=session.id, reply=full_reply, state=session.state,
            action="REPORT_READY", report_url=report_url,
        )
    except Exception as e:
        logger.error("差距分析异常: %s", str(e), exc_info=True)
        session.state = "IDLE"
        reply = "分析过程遇到了问题，请重新尝试～"
        session.add_chat_history("system", reply)
        _save_session(session)
        return ChatResponse(session_id=session.id, reply=reply, state=session.state)


@app.get("/api/gap_progress/{session_id}")
async def get_gap_progress(session_id: str):
    """获取差距分析进度"""
    prog = _gap_progress.get(session_id, {"progress": 0, "stage": "等待中", "done": False})
    return prog


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
