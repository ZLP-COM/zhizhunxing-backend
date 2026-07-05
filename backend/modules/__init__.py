# 职准星 - 业务逻辑模块统一导出
#
# 导出全部模块的核心函数，外部调用方式：
#   from backend.modules import handle_chat, run_assessment_step, ...

from .chat_handler import (
    handle_chat,          # 对话接待入口（意图识别 + 开场缓存拦截）
    sanitize_input,       # 输入安全过滤
    get_welcome_message,  # 新会话引导话术
)

from .assessment import (
    run_assessment_step,  # 六问测评单步执行（统一入口）
    get_question,         # 获取问题定义
    get_question_text,    # 获取问题文本
    get_all_answers,      # 获取全部已作答内容
    get_all_results,      # 获取全部分析结果
    is_assessment_complete,  # 检查测评是否完成
    build_assessment_summary, # 测评进度摘要
)

from .reasoning import (
    run_job_reason,       # 综合推理统一入口
    execute_reasoning,    # 兼容旧接口
    format_recommendation, # 格式化推荐报告
)

from .gap_analysis import (
    run_gap_analysis,     # 差距分析工作流统一入口
    execute_gap_analysis, # 兼容旧接口
)

from .pdf_parser import (
    parse_pdf_bytes,      # PDF 解析入口（接收二进制流）
    safe_parse_pdf,       # 兼容旧接口（接收文件路径）
    PDFParseError,        # PDF 解析异常
    is_scanned_pdf,       # 扫描件检测
    validate_pdf_extension, # 扩展名校验
)

from .kb_manager import (
    kb_manager,           # 全局知识库管理器单例
    get_kb_retrieval_result,  # 统一检索入口
    # ChromaDB 未安装，以下函数降级为本地文件匹配（搜索函数保留兼容导入）
    # search_jd_for_recommendations,  # 推理用 JD 检索
    # search_kb_for_gap_analysis,     # 差距分析用校招规则检索
    # search_product_manual,          # 产品说明书检索（功能答疑）
    init_knowledge_base,  # 知识库初始化
    KnowledgeBaseManager, # 知识库管理器类
)

from .report_gen import (
    generate_full_report,  # 报告生成统一入口
    build_report_markdown, # Markdown 报告拼接
    build_report_summary,  # 对话内摘要
    render_docx_bytes,     # docx 二进制流渲染
    clear_cache,           # 清空报告缓存
)

__all__ = [
    # chat_handler
    'handle_chat', 'sanitize_input', 'get_welcome_message',
    # assessment
    'run_assessment_step', 'get_question', 'get_question_text',
    'get_all_answers', 'get_all_results', 'is_assessment_complete',
    'build_assessment_summary',
    # reasoning
    'run_job_reason', 'execute_reasoning', 'format_recommendation',
    # gap_analysis
    'run_gap_analysis', 'execute_gap_analysis',
    # pdf_parser
    'parse_pdf_bytes', 'safe_parse_pdf', 'PDFParseError',
    'is_scanned_pdf', 'validate_pdf_extension',
    # kb_manager
    'kb_manager', 'get_kb_retrieval_result',
    'search_jd_for_recommendations', 'search_kb_for_gap_analysis',
    'init_knowledge_base', 'KnowledgeBaseManager',
    # report_gen
    'generate_full_report', 'build_report_markdown',
    'build_report_summary', 'render_docx_bytes', 'clear_cache',
]
