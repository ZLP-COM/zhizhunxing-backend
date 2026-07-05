"""职准星 - LLM 调用封装模块

统一 LLM 请求封装层，支持 DeepSeek / 豆包 / mimo-v2.5 等兼容 OpenAI 接口的模型。
全部配置读取自 config 模块（间接读取 .env），无需硬编码。

核心特性：
  - 最多 2 次自动重试 + 指数退避
  - JSON 自动提取 + 多重容错 + 兜底静态文本
  - 按场景（闲聊/测评/推理/诊断/优化/话术）使用 config 预设温度
  - 401/超时/限流友好提示，不直接抛崩溃异常
  - Token 消耗统计（prompt / completion / 总调用次数）
  - 限流捕获：单会话 30s 内最多 5 次 LLM 调用
  - 兼容 mimo-v2.5 等第三方模型
  - 所有异常均以 LLMError 包装抛出，上层 catch 后走降级
"""

import os
import json
import re
import time
import asyncio
import logging
from typing import Optional, Any, Dict, List

from openai import AsyncOpenAI, APIError, APITimeoutError, AuthenticationError, RateLimitError
from . import config

logger = logging.getLogger(__name__)


# ============================================================
# 自定义异常
# ============================================================

class LLMError(Exception):
    """LLM 调用通用异常（调用方 catch 后走降级）"""
    pass


class LLMRateLimitError(LLMError):
    """触发了会话级限流"""
    pass


class LLMAuthError(LLMError):
    """API Key 鉴权失败（401）"""
    pass


class LLMTimeoutError(LLMError):
    """LLM 调用超时"""
    pass


# ============================================================
# 场景温度映射（读取 config 预设常量）
# ============================================================

SCENE_TEMPERATURE = {
    "chat":              config.LLM_TEMPERATURE_CHAT,              # 闲聊接待
    "assessment":        config.LLM_TEMPERATURE_ANALYSIS,         # 单题分析
    "reasoning":         config.LLM_TEMPERATURE_REASONING,        # 综合推理
    "format":            config.LLM_TEMPERATURE_FORMAT,           # 格式化推荐
    "diagnosis":         config.LLM_TEMPERATURE_ANALYSIS,         # 差距诊断
    "optimize":          config.LLM_TEMPERATURE_OPTIMIZE,         # 简历优化
    "counterattack":     config.LLM_TEMPERATURE_COUNTERATTACK,    # 逆袭话术
}

SCENE_MAX_TOKENS = {
    "chat":              config.LLM_MAX_TOKENS_CHAT,
    "assessment":        config.LLM_MAX_TOKENS_ANALYSIS,
    "reasoning":         config.LLM_MAX_TOKENS_REASONING,
    "format":            config.LLM_MAX_TOKENS_FORMAT,
    "diagnosis":         config.LLM_MAX_TOKENS_DIAGNOSIS,
    "optimize":          config.LLM_MAX_TOKENS_OPTIMIZE,
    "counterattack":     config.LLM_MAX_TOKENS_COUNTERATTACK,
}


# ============================================================
# Token 统计
# ============================================================

class TokenStats:
    """全局 Token 消耗统计（单例）"""

    def __init__(self):
        self._records: Dict[str, Dict[str, int]] = {}  # session_id -> {prompt, completion, calls}

    def record(self, session_id: str, prompt_tokens: int, completion_tokens: int) -> None:
        if session_id not in self._records:
            self._records[session_id] = {"prompt": 0, "completion": 0, "calls": 0}
        self._records[session_id]["prompt"] += prompt_tokens
        self._records[session_id]["completion"] += completion_tokens
        self._records[session_id]["calls"] += 1

    def get(self, session_id: str) -> Dict[str, int]:
        return self._records.get(session_id, {"prompt": 0, "completion": 0, "calls": 0})


token_stats = TokenStats()


# ============================================================
# 主客户端
# ============================================================

class LLMClient:
    """LLM 客户端封装，读取 config 全局配置"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # 优先读取 .env 中 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME，
        # 兼容旧 DEEPSEEK_* 变量名，兼容 mimo-v2.5 等第三方模型
        self.api_key = (
            api_key
            or os.getenv("LLM_API_KEY")
            or config.DEEPSEEK_API_KEY
        )
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or config.DEEPSEEK_BASE_URL
        )
        self.model = (
            model
            or os.getenv("LLM_MODEL_NAME")
            or config.LLM_MODEL
        )
        self._session_call_times: Dict[str, List[float]] = {}
        self._client: Optional[AsyncOpenAI] = None

    # ---- 懒加载 client ----
    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=config.LLM_TIMEOUT,
            )
        return self._client

    # ---- 核心调用入口 ----
    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[str] = None,
        session_id: Optional[str] = None,
        scene: Optional[str] = None,
    ) -> Any:
        """
        统一 LLM 调用入口。

        Args:
            system_prompt: 系统提示词
            user_prompt:   用户提示词
            temperature:   覆盖温度（不传则从 scene 推断，不传 scene 默认 0.3）
            max_tokens:    覆盖 max_tokens（不传则从 scene 推断）
            response_format: "json" | None
            session_id:    会话ID（用于限流 + Token 统计）
            scene:         场景名，用于自动选择温度 / max_tokens
                          chat | assessment | reasoning | format | diagnosis | optimize | counterattack

        Returns:
            response_format="json" → dict
            否则 → {"output": str}

        Raises:
            LLMError 系列（上层 catch 后降级）
        """
        # ---- 1. 限流检查 ----
        if session_id:
            self._check_rate_limit(session_id)

        # ---- 2. 参数推导 ----
        if temperature is None and scene:
            temperature = SCENE_TEMPERATURE.get(scene, 0.3)
        temperature = temperature if temperature is not None else 0.3

        if max_tokens is None and scene:
            max_tokens = SCENE_MAX_TOKENS.get(scene, 2000)
        max_tokens = max_tokens if max_tokens is not None else 2000

        # ---- 3. 构造请求 ----
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # 关闭深度思考（降低推理延迟）- 已禁用，DeepSeek API 不支持此参数
        # if config.LLM_DISABLE_DEEP_THINK:
        #     kwargs.setdefault("extra_body", {})
        #     kwargs["extra_body"].setdefault("enable_deep_think", False)

        # DeepSeek 不支持 response_format 参数，改为在 prompt 中指示 JSON 输出
        if response_format == "json":
            # 在 user prompt 末尾追加 JSON 格式指示，不依赖 API 参数
            kwargs["messages"][1]["content"] += (
                "\n\n请严格按照 JSON 格式输出，不包含任何其他文字、代码块标记或markdown。"
                "直接输出 JSON 对象本身。"
            )

        # ---- 4. 发送请求（最多 2 次重试）----
        last_exception: Optional[Exception] = None

        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content.strip()

                # ---- 5. Token 统计 ----
                usage = getattr(response, "usage", None)
                if usage:
                    pt = getattr(usage, "prompt_tokens", 0) or 0
                    ct = getattr(usage, "completion_tokens", 0) or 0
                    logger.info(
                        f"LLM调用 | 模型={self.model} | "
                        f"场景={scene or 'default'} | "
                        f"输入Token={pt} | 输出Token={ct} | "
                        f"会话={session_id or 'N/A'}"
                    )
                    if session_id:
                        token_stats.record(session_id, pt, ct)

                # ---- 6. 解析返回 ----
                if response_format == "json":
                    return self._parse_json_response(content)

                return {"output": content}

            except AuthenticationError as e:
                last_exception = LLMAuthError(
                    "API Key 鉴权失败（401），请检查 LLM_API_KEY / DEEPSEEK_API_KEY 配置"
                )
                # 鉴权失败不再重试
                break

            except APITimeoutError as e:
                last_exception = LLMTimeoutError(
                    f"AI服务响应超时（{config.LLM_TIMEOUT}s），请稍后重试"
                )
                if attempt < config.LLM_MAX_RETRIES:
                    wait = (attempt + 1) * 2
                    logger.warning(
                        f"LLM超时第{attempt + 1}次，{wait}s后重试: {str(e)}"
                    )
                    await asyncio.sleep(wait)
                    continue
                break

            except RateLimitError as e:
                last_exception = LLMError(
                    "API 调用频率过高，触发供应商限流，请稍后重试"
                )
                if attempt < config.LLM_MAX_RETRIES:
                    wait = (attempt + 1) * 5
                    logger.warning(
                        f"API限流第{attempt + 1}次，{wait}s后重试: {str(e)}"
                    )
                    await asyncio.sleep(wait)
                    continue
                break

            except APIError as e:
                status = getattr(e, "status_code", 0) or 0
                if 500 <= status < 600:
                    msg = f"AI服务暂时不可用（{status}），请稍后重试"
                elif status == 400 and response_format == "json":
                    # 回退：移除 response_format（已不在 kwargs，但保留逻辑兼容）
                    logger.warning(f"JSON 请求被拒(400)，回退无格式重试: {str(e)[:80]}")
                    response_format = None
                    attempt -= 1
                    continue
                else:
                    msg = f"API调用异常（{status}）: {str(e)[:100]}"
                last_exception = LLMError(msg)
                if attempt < config.LLM_MAX_RETRIES:
                    wait = (attempt + 1) * 3
                    await asyncio.sleep(wait)
                    continue
                break

            except Exception as e:
                last_exception = LLMError(f"LLM调用未知异常: {str(e)[:100]}")
                if attempt < config.LLM_MAX_RETRIES:
                    wait = (attempt + 1) * 2
                    logger.warning(
                        f"LLM未知异常第{attempt + 1}次，{wait}s后重试: {str(e)}"
                    )
                    await asyncio.sleep(wait)
                    continue
                break

        # 所有重试耗尽
        raise last_exception or LLMError("LLM调用失败（未知原因）")

    # ---- 安全 JSON 调用（失败自动降级兜底）----
    async def safe_chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        session_id: Optional[str] = None,
        scene: Optional[str] = None,
        fallback: Optional[dict] = None,
    ) -> dict:
        """
        安全的 JSON 模式调用。

        内部流程：
          1. 正常调用 chat(response_format="json")
          2. JSON 解析失败 / 返回 {"output":...} → 尝试从文本中提取 JSON
          3. 降低温度重试 1 次
          4. 仍失败 → 使用 fallback 兜底（不会抛异常）
          5. 非 JSON 类异常 → 使用 fallback 兜底
        """
        try:
            result = await self.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format="json",
                session_id=session_id,
                scene=scene,
            )
            # 当 response_format 被 DeepSeek 拒绝后回退到文本模式，
            # chat() 返回 {"output": content}，需要手动解析
            if isinstance(result, dict) and "output" in result:
                return self._parse_json_response(result["output"])
            return result

        except (json.JSONDecodeError, LLMError):
            # JSON 解析失败 → 降低温度重试一次
            logger.warning("JSON解析失败，降低温度重试")
            lower_temp = max((temperature or 0.3) - 0.2, 0.1)
            try:
                result = await self.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=lower_temp,
                    max_tokens=max_tokens,
                    response_format="json",
                    session_id=session_id,
                    scene=scene,
                )
                if isinstance(result, dict) and "output" in result:
                    return self._parse_json_response(result["output"])
                return result
            except (json.JSONDecodeError, LLMError):
                pass
            except (json.JSONDecodeError, LLMError):
                pass

        except Exception:
            pass

        # 兜底
        if fallback is not None:
            logger.warning(f"LLM JSON调用最终降级，使用兜底数据")
            return fallback

        # 没有 fallback 则抛出通用提示
        raise LLMError("AI服务暂时不可用，请稍后重试")

    # ---- JSON 多重容错解析 ----
    def _parse_json_response(self, content: str) -> dict:
        """
        解析 LLM 返回的 JSON 字符串，多重容错：

        1. 移除 ```json ... ``` 代码块标记
        2. 标准 json.loads
        3. 提取 ```json ... ``` 块内容
        4. 提取 {...} 对象
        5. 以上均失败 → 抛 json.JSONDecodeError
        """
        content = content.strip()

        # 移除 markdown 代码块标记
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
        content = content.strip()

        # 尝试标准解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 尝试提取 ```json ... ``` 块
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", content)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # 尝试提取 {...} 根对象
        match = re.search(r"\{[\s\S]*\}", content)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # 尝试提取 {...} 嵌套对象
        for m in re.finditer(r"\{[^{}]*\}", content):
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                continue

        raise json.JSONDecodeError(
            f"无法解析LLM输出为JSON（前200字符）: {content[:200]}", content, 0
        )

    # ---- 会话级限流 ----
    def _check_rate_limit(self, session_id: str) -> None:
        """
        单会话 30 秒内最多 SESSION_LLM_CALLS_PER_30S 次 LLM 调用。
        超限抛出 LLMRateLimitError（友好提示，不崩溃）。
        """
        now = time.time()
        if session_id not in self._session_call_times:
            self._session_call_times[session_id] = []

        # 清理 30 秒前的记录
        window = [t for t in self._session_call_times[session_id] if now - t < 30]
        self._session_call_times[session_id] = window

        max_calls = config.SESSION_LLM_CALLS_PER_30S

        if len(window) >= max_calls:
            wait = 30 - (now - window[0])
            if wait > 0:
                logger.warning(
                    f"会话 {session_id} 触发限流（{len(window)}次/30s），等待 {wait:.0f}s"
                )
                raise LLMRateLimitError(
                    f"请求太频繁了，请 {int(wait) + 1} 秒后再试～"
                )

        self._session_call_times[session_id].append(now)

    # ---- 重置限流计数（会话重置时调用）----
    def reset_rate_limit(self, session_id: str) -> None:
        """重置指定会话的限流记录（会话重置时调用）"""
        self._session_call_times.pop(session_id, None)

    # ---- 健康检查 ----
    async def health_check(self) -> dict:
        """
        轻量级 LLM 连通性检查。
        返回 {"available": bool, "model": str, "message": str}
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return {
                "available": True,
                "model": self.model,
                "message": "LLM 服务可用",
            }
        except Exception as e:
            return {
                "available": False,
                "model": self.model,
                "message": f"LLM 服务不可用: {str(e)[:80]}",
            }


# ============================================================
# 全局单例
# ============================================================

llm_client = LLMClient()
