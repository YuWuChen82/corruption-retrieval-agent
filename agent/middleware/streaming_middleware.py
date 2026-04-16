"""
流式输出中间件：包装流式 LLM 响应，逐 token/逐 chunk 产出给调用方。

支持两种模式：
  - sync (默认): 返回 Generator，调用方自己遍历
  - callback:    每收到一个 chunk 就调用 callback 函数（适合 WebSocket / SSE）

使用示例：
    # 同步模式
    for token in streaming_mw.stream(agent, "贪污最高的是谁"):
        print(token, end="", flush=True)

    # 回调模式
    streaming_mw.stream(agent, "贪污罪是什么", callback=print_token)
"""

from agent.middleware.base import BaseMiddleware
from model.factory import chat_model
from typing import Any, Callable, Generator
import sys


class StreamingMiddleware(BaseMiddleware):
    """
    流式输出中间件。

    包装 model.stream()，在输出每个 chunk 前/后可插入自定义处理。
    """

    name = "streaming_middleware"

    def __init__(self, encoding: str = "utf-8", flush: bool = True):
        self.encoding = encoding
        self.flush = flush
        self._enabled = True

    def enable(self) -> "StreamingMiddleware":
        self._enabled = True
        return self

    def disable(self) -> "StreamingMiddleware":
        self._enabled = False
        return self

    def stream(
        self,
        messages: list[dict],
        system_prompt: str = "",
        callback: Callable[[str], None] | None = None,
    ) -> Generator[str, None, None]:
        """
        执行流式推理，逐 yield token。

        Args:
            messages:      对话历史 [{role: str, content: str}, ...]
            system_prompt: 系统提示词（可选）
            callback:      每个 token 产出时的回调函数

        Yields:
            str: 逐 token 的文本内容
        """
        if not self._enabled:
            # 降级为非流式
            resp = chat_model.invoke(messages)
            yield resp.content
            return

        # 组装完整消息列表
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # 执行流式调用
        try:
            stream_resp = chat_model.stream(full_messages)
        except Exception:
            # 模型不支持流式，降级
            resp = chat_model.invoke(full_messages)
            yield resp.content
            return

        buffer = ""
        for event in stream_resp:
            # 兼容 LangChain 的 AIMessageChunk 和原始文本
            token = self._extract_token(event)
            if token:
                buffer += token
                for processed in self.transform_stream(token):
                    if processed:
                        if callback:
                            callback(processed)
                        yield processed

    def _extract_token(self, event: Any) -> str:
        """从不同格式的流式事件中提取文本 token"""
        try:
            # LangChain 流式 chunk 格式
            if hasattr(event, "content"):
                return event.content or ""
            if hasattr(event, "text"):
                return event.text or ""
            if isinstance(event, str):
                return event
            # dict 格式
            if isinstance(event, dict):
                return event.get("content", "") or event.get("text", "")
            return str(event)
        except Exception:
            return ""

    def transform_stream(self, chunk: str) -> Generator[str, None, None]:
        """
        流式 chunk 后处理。可被子类重写做实时打印、分词等。
        默认透传。
        """
        yield chunk

    def print_stream(self, generator: Generator[str, None, None]) -> str:
        """
        消费 generator 并打印到终端，返回完整字符串。
        便捷方法，用于 CLI 测试。
        """
        full_text = []
        for token in generator:
            print(token, end="", flush=self.flush)
            full_text.append(token)
        print()  # 换行
        return "".join(full_text)


# 全局单例
_streaming_mw_instance: StreamingMiddleware | None = None


def get_streaming_middleware() -> StreamingMiddleware:
    global _streaming_mw_instance
    if _streaming_mw_instance is None:
        _streaming_mw_instance = StreamingMiddleware()
    return _streaming_mw_instance
