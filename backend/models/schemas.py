"""职准星 - Pydantic 数据模型 (API 请求/响应)"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class ChatRequest(BaseModel):
    """对话请求"""
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="用户消息")
    attachments: Optional[List[Dict[str, Any]]] = Field(default=None, description="文件附件")


class ChatResponse(BaseModel):
    """对话响应"""
    session_id: str = Field(..., description="会话ID")
    reply: str = Field(..., description="系统回复")
    state: str = Field(..., description="当前会话状态")
    action: Optional[str] = Field(default=None, description="前端需要执行的动作")
    report_url: Optional[str] = Field(default=None, description="报告下载链接")


class UploadResponse(BaseModel):
    """文件上传响应"""
    success: bool = Field(..., description="是否成功")
    resume_text: str = Field(..., description="解析出的简历纯文本")
    file_name: str = Field(..., description="文件名")
    file_size: int = Field(..., description="文件大小（字节）")


class SessionResponse(BaseModel):
    """会话状态响应"""
    session_id: str = Field(..., description="会话ID")
    state: str = Field(..., description="会话状态")
    created_at: Optional[str] = Field(default=None, description="创建时间")
    updated_at: Optional[str] = Field(default=None, description="更新时间")
    current_question: int = Field(default=0, description="当前问题序号")
    total_questions: int = Field(default=6, description="总问题数")
    selected_direction: Optional[str] = Field(default=None, description="已选方向")
    jd_source: Optional[str] = Field(default=None, description="JD来源")
    has_report: bool = Field(default=False, description="是否有报告")


class NewSessionResponse(BaseModel):
    """新建会话响应"""
    session_id: str = Field(..., description="会话ID")
    state: str = Field(default="IDLE", description="初始状态")
    created_at: str = Field(..., description="创建时间")
    message: str = Field(default="", description="欢迎消息")


class ErrorResponse(BaseModel):
    """错误响应"""
    detail: str = Field(..., description="错误详情")
    code: str = Field(default="UNKNOWN_ERROR", description="错误代码")


class TokenStatsResponse(BaseModel):
    """Token 统计响应"""
    session_id: str = Field(..., description="会话ID")
    total_prompt_tokens: int = Field(default=0, description="总输入Token")
    total_completion_tokens: int = Field(default=0, description="总输出Token")
    total_calls: int = Field(default=0, description="总调用次数")
