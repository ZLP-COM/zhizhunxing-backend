"""职准星 - 配置管理模块

全部配置常量集中管理，通过 .env 文件可覆盖默认值。
包含：LLM超时、单IP限流、会话TTL、文件上传限制、报告缓存、向量库路径、自动清理等。
"""

import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env 文件
load_dotenv()

logger = logging.getLogger(__name__)

# ============================================================
# 项目基础路径
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
"""项目根目录（backend/）"""

# ============================================================
# 1. LLM 配置（含超时与重试）
# ============================================================

DEEPSEEK_API_KEY = os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
"""DeepSeek / 豆包 API Key"""

DEEPSEEK_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
"""API 基础地址，可切换为豆包等兼容端点"""

LLM_MODEL = os.getenv("LLM_MODEL_NAME") or os.getenv("LLM_MODEL", "deepseek-chat")
"""默认 LLM 模型名"""

LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
"""单次 LLM 调用超时（秒），超过此时间触发降级重试"""

LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
"""LLM 调用失败后的最大重试次数"""

LLM_TEMPERATURE_CHAT = 0.7
"""闲聊模式温度（高 = 创造性）"""

LLM_TEMPERATURE_ANALYSIS = 0.3
"""分析节点温度（低 = 确定性）"""

LLM_TEMPERATURE_REASONING = 0.5
"""综合推理温度（中等）"""

LLM_TEMPERATURE_FORMAT = 0.7
"""格式化生成温度（高 = 表达丰富）"""

LLM_TEMPERATURE_OPTIMIZE = 0.4
"""简历优化温度"""

LLM_TEMPERATURE_COUNTERATTACK = 0.6
"""逆袭话术温度"""

LLM_MAX_TOKENS_CHAT = 2200
"""闲聊最大输出 Token"""

LLM_MAX_TOKENS_ANALYSIS = 500
"""单题分析最大输出 Token"""

LLM_MAX_TOKENS_REASONING = 2000
"""综合推理最大输出 Token"""

LLM_MAX_TOKENS_DIAGNOSIS = 2500
"""差距诊断最大输出 Token"""

LLM_MAX_TOKENS_OPTIMIZE = 3000
"""简历优化最大输出 Token"""

LLM_MAX_TOKENS_COUNTERATTACK = 3000
"""逆袭话术最大输出 Token"""

LLM_MAX_TOKENS_FORMAT = 2000
"""格式化推荐最大输出 Token"""

LLM_DISABLE_DEEP_THINK = True
"""关闭深度思考模式，降低推理延迟"""

# ============================================================
# 2. 向量数据库（ChromaDB）配置
# ============================================================

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(BASE_DIR / "chroma_db"))
"""ChromaDB 持久化路径"""

KB_CHUNK_SIZE = 1200
"""知识库文本分块最大字符数"""

KB_CHUNK_OVERLAP = 200
"""分块间重叠字符数"""

KB_MIN_SCORE = 0.3
"""向量检索最低相似度阈值（低于此值的结果丢弃）"""

KB_JD_MIN_SCORE = 0.5
"""JD 检索最低相似度阈值（高于通用阈值）"""

KB_FALLBACK_TEXT = ""
"""知识库不可用时降级使用的静态文本"""

KB_CONTEXT_MAX_LENGTH = 4000
"""知识库上下文注入 LLM 的最大字符数"""

# ============================================================
# 3. 限流与并发控制（单IP + 单会话 + 全局）
# ============================================================

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
"""单 IP 每分钟最多对话请求次数"""

MAX_CONCURRENT_SESSIONS = int(os.getenv("MAX_CONCURRENT_SESSIONS", "5"))
"""服务同时支持的最大并发会话数"""

SESSION_LLM_CALLS_PER_30S = int(os.getenv("SESSION_LLM_CALLS_PER_30S", "5"))
"""单会话 30 秒窗口内最多 LLM 调用次数，防止触发 API 风控"""

# ============================================================
# 4. 会话生命周期管理（TTL）
# ============================================================

MAX_SESSION_HOURS = int(os.getenv("MAX_SESSION_HOURS", "24"))
"""会话有效期（小时），超时后强制新建"""
SESSION_CLEANUP_INTERVAL_MINUTES = 10
"""过期会话清理任务执行间隔（分钟）"""

# ============================================================
# 5. 文件上传限制
# ============================================================

MAX_FILE_SIZE = 10 * 1024 * 1024
"""单文件上传上限（10MB）"""

MAX_INPUT_LENGTH = 10000
"""用户单次输入最大字符数"""

MAX_MESSAGE_LENGTH = 5000
"""单条对话消息最大字符数"""

MAX_RESUME_LENGTH = 10000
"""简历文本最大长度（超过截断）"""

MAX_JD_LENGTH = 10000
"""JD 文本最大长度（超过截断）"""

ALLOWED_UPLOAD_EXTENSIONS = {".pdf"}
"""允许上传的文件扩展名"""

# ============================================================
# 6. 报告缓存
# ============================================================

REPORT_CACHE_MINUTES = 10
"""同一套 JD + 简历 10 分钟内重复请求直接复用已生成 docx，不重复调用 LLM"""

# ============================================================
# 7. 文件自动清理（隐私保护）
# ============================================================

ENABLE_AUTO_CLEANUP = os.getenv("ENABLE_AUTO_CLEANUP", "true").lower() == "true"
"""24 小时文件自动清理开关（上传 PDF / 生成报告），默认开启"""

CLEANUP_FILE_MAX_AGE_HOURS = 24
"""文件最大保留时间（小时），超时自动删除"""

CLEANUP_INTERVAL_HOURS = 1
"""文件清理任务执行间隔（小时）"""

# ============================================================
# 8. 目录与文件路径
# ============================================================

UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(BASE_DIR / "uploads"))
"""用户上传 PDF 存储目录"""

REPORT_DIR = os.getenv("REPORT_DIR", str(BASE_DIR / "reports"))
"""生成 docx 报告存储目录"""

KNOWLEDGE_DIR = BASE_DIR / "knowledge" / "data"
"""知识库源文件根目录"""

JDS_DIR = KNOWLEDGE_DIR / "jds"
"""校招 JD 库源文件目录（30 份 .txt）"""

JUDGMENT_RULES_FILE = KNOWLEDGE_DIR / "judgment_rules.md"
"""校招评判标准文档路径"""

HOLLAND_GUIDE_FILE = KNOWLEDGE_DIR / "holland_guide.md"
"""霍兰德职业兴趣参考表路径"""

# ============================================================
# 9. 产品说明书（README.md 知识库）
# ============================================================

LOAD_PRODUCT_MANUAL = False
"""是否启动时加载产品说明书 README.md 至向量库 product_manual 集合，用于功能答疑"""

PRODUCT_MANUAL_PATH = BASE_DIR.parent / "README.md"
"""指向项目根目录的产品说明书路径"""

# ============================================================
# 10. 数据库
# ============================================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sessions.db")
"""SQLite 会话存储数据库连接串"""

# ============================================================
# 10. CORS
# ============================================================

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
"""允许的跨域来源，多个用逗号分隔"""

# ============================================================
# 11. 服务启动参数
# ============================================================

HOST = os.getenv("HOST", "0.0.0.0")
"""FastAPI 监听地址"""

PORT = int(os.getenv("PORT", "8000"))
"""FastAPI 监听端口"""

# ============================================================
# 12. 日志配置
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
"""日志级别: DEBUG / INFO / WARNING / ERROR"""

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
"""日志输出格式"""

# 会话历史保留轮数
CHAT_HISTORY_MAX_ROUNDS = 20
"""会话历史最大保留轮数"""

CHAT_HISTORY_RECENT_N = 5
"""意图识别时传入 LLM 的最近对话轮数"""

# ============================================================
# 13. 敏感信息脱敏规则
# ============================================================

SENSITIVE_PATTERNS = [
    r"1[3-9]\d{9}",           # 手机号
    r"\d{18}[\dXx]",           # 身份证号
    r"\d{6}\s*\d{8}\s*\d{4}", # 银行卡号（简化）
]
"""日志打印时需脱敏的正则模式列表"""

# ============================================================
# 14. 输入安全过滤
# ============================================================

HARMFUL_PATTERNS = [
    r"http[s]?://(?!.*(?:zhizhunxing|localhost|deepseek))[^\s]{5,}",
]
"""需要过滤的恶意内容正则模式"""

# ============================================================
# 初始化
# ============================================================

def init_dirs() -> None:
    """启动时创建必要的存储目录"""
    dirs = [UPLOAD_DIR, REPORT_DIR, str(BASE_DIR / "chroma_db")]
    for d in dirs:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError as e:
            logger.warning(f"目录创建失败 {d}: {e}")


def validate_config() -> list[str]:
    """验证关键配置，返回缺失项列表"""
    warnings: list[str] = []
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY.startswith("sk-") is False:
        warnings.append("DEEPSEEK_API_KEY 未配置或格式不正确")
    if LLM_TIMEOUT < 5:
        warnings.append("LLM_TIMEOUT 过小（<5秒），可能导致频繁超时")
    if RATE_LIMIT_PER_MINUTE < 1:
        warnings.append("RATE_LIMIT_PER_MINUTE 必须 >= 1")
    # 产品说明书文件存在性校验
    if LOAD_PRODUCT_MANUAL:
        if not PRODUCT_MANUAL_PATH.exists():
            warnings.append(
                f"产品说明书未找到: {PRODUCT_MANUAL_PATH}，"
                "功能答疑将使用降级关键词匹配"
            )
    return warnings


# 启动时执行
init_dirs()
