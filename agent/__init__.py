"""
Agent 包：Agent 工厂、中间件、工具集的统一导出。
"""

from agent.agent_factory import Agent, AgentFactory, get_agent, ask, ask_stream
from agent.middleware import (
    BaseMiddleware,
    MiddlewareManager,
    ContextMiddleware,
    StreamingMiddleware,
    get_context_middleware,
    get_streaming_middleware,
)

__all__ = [
    # Agent
    "Agent",
    "AgentFactory",
    "get_agent",
    "ask",
    "ask_stream",
    # Middleware
    "BaseMiddleware",
    "MiddlewareManager",
    "ContextMiddleware",
    "StreamingMiddleware",
    "get_context_middleware",
    "get_streaming_middleware",
]
