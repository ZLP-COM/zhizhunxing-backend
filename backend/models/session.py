"""职准星 - 数据模型 (SQLModel + SQLite)"""

from sqlmodel import SQLModel, Field, Column
from sqlalchemy import JSON, LargeBinary
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any


class Session(SQLModel, table=True):
    """会话模型 - SQLite存储"""
    __tablename__ = "sessions"

    id: str = Field(primary_key=True, default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    state: str = Field(default="IDLE")

    # 第一段数据
    assessment_answers: Dict[str, str] = Field(default={}, sa_column=Column(JSON))
    assessment_results: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    reasoning_result: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    formatted_recommendation: str = Field(default="")

    # 选定的方向
    selected_direction: str = Field(default="")
    selected_direction_index: int = Field(default=-1)
    selected_jd: str = Field(default="")         # 选定的系统JD全文
    selected_jd_list: List[Dict] = Field(default=[], sa_column=Column(JSON))  # 所有匹配的JD

    # 第二段数据
    jd_source: str = Field(default="")           # "system" 或 "custom"
    custom_jd: str = Field(default="")           # 用户粘贴的外部JD
    final_jd: str = Field(default="")
    final_resume: str = Field(default="")
    resume_source: str = Field(default="")       # "pdf" 或 "text"
    resume_file_name: str = Field(default="")

    # 差距分析结果
    gap_analysis_result: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    optimized_resume_result: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))
    counterattack_result: Dict[str, Any] = Field(default={}, sa_column=Column(JSON))

    # 报告
    report_filename: str = Field(default="")
    report_generated_at: Optional[datetime] = Field(default=None)

    # 对话历史（最近20轮）
    chat_history: List[Dict] = Field(default=[], sa_column=Column(JSON))

    # 当前测评进度
    current_question: int = Field(default=0)  # 0-6, 0表示未开始

    # 用户跳过标记
    skipped_questions: List[str] = Field(default=[], sa_column=Column(JSON))

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于API响应）"""
        return {
            "session_id": self.id,
            "state": self.state,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "current_question": self.current_question,
            "total_questions": 6,
            "selected_direction": self.selected_direction,
            "jd_source": self.jd_source,
            "has_report": bool(self.report_filename),
        }

    def add_chat_history(self, role: str, content: str):
        """添加对话历史记录"""
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        self.chat_history = (self.chat_history or []) + [entry]
        # 保持最近20轮
        if len(self.chat_history) > 20:
            self.chat_history = self.chat_history[-20:]
        self.updated_at = datetime.now()

    def get_recent_history(self, n: int = 5) -> List[Dict]:
        """获取最近n轮对话历史"""
        history = self.chat_history or []
        return history[-n:] if len(history) > n else history

    def has_intro_delivered(self) -> bool:
        """检查是否已经输出过开场介绍"""
        for msg in (self.chat_history or []):
            if msg.get("role") == "system" and "欢迎来到职准星" in msg.get("content", ""):
                return True
        return False
