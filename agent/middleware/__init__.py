"""
Agent 中间件包

导出：
    BaseMiddleware     — 中间件基类
    MiddlewareManager   — 中间件管理器
    ContextMiddleware   — 上下文注入中间件
    StreamingMiddleware — 流式输出中间件
    get_context_middleware / get_streaming_middleware — 单例获取函数
"""

from agent.middleware.base import BaseMiddleware, MiddlewareManager
from agent.middleware.context_middleware import (
    ContextMiddleware,
    get_context_middleware,
)
from agent.middleware.streaming_middleware import (
    StreamingMiddleware,
    get_streaming_middleware,
)

__all__ = [
    "BaseMiddleware",
    "MiddlewareManager",
    "ContextMiddleware",
    "StreamingMiddleware",
    "get_context_middleware",
    "get_streaming_middleware",
]
