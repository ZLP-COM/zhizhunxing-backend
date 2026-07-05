"""职准星 - 差距分析直接执行模块（Fix 4+5）
自动追加到 main.py 末尾。"""

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
        full_reply = summary + opt_text + cta_text + "\n\n---\n下载完整报告：" + report_url
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=True)
