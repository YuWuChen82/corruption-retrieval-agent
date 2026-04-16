"""
Agent 工厂：创建、配置、管理 Agent 实例。

职责：
  1. 从 registry 自动获取所有已注册工具（无需手动维护列表）
  2. 配置中间件（MiddlewareManager）
  3. 创建 Agent 实例（支持流式 / 非流式）
  4. 提供统一的 execute / execute_stream 接口

使用方式：
  agent = AgentFactory.create()

  # 非流式
  answer = agent.execute("贪污金额最高的是谁")

  # 流式
  for token in agent.execute_stream("贪污金额最高的是谁"):
      print(token, end="", flush=True)
"""

from typing import Generator

from agent.middleware import (
    MiddlewareManager,
    get_context_middleware,
    get_streaming_middleware,
)
from model.factory import chat_model
from agent.tools.router import rag_summarize_tool
# 触发所有 @auto_tool 装饰器注册（必须在获取 tools 之前）
from agent.tools import data_query_tool   # noqa: F401
from agent.tools import web_search_tool   # noqa: F401
from agent.tools.registry import TOOL_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# 工具注册表（Tool Registry）
# 所有通过 @auto_tool 注册的工具在此自动汇总，绑定到模型
# ─────────────────────────────────────────────────────────────────────────────

# 从 registry 获取所有已注册工具（含 CSV 工具、联网工具、RAG 工具）
ALL_AGENT_TOOLS = TOOL_REGISTRY.tools

# 工具名 → 函数对象 映射，供 execute 按名调用
TOOL_MAP = {t.name: t for t in ALL_AGENT_TOOLS}

# 绑定工具后的模型（不带 system prompt，system prompt 由 AgentFactory 注入）
_model_with_tools = chat_model.bind_tools(ALL_AGENT_TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# Agent 类
# ─────────────────────────────────────────────────────────────────────────────

class Agent:
    """
    Agent 实例：封装一次完整的对话请求。

    持有：
      middleware_manager — 中间件管理器
      session_history   — 当前会话的历史记录
    """

    def __init__(
        self,
        middleware_manager: MiddlewareManager,
        session_id: str | None = None,
        system_prompt: str = "",
    ):
        self.middleware_manager = middleware_manager
        self.session_id = session_id or "default"
        self.system_prompt = system_prompt
        self.session_history: list[dict] = []

    def _build_messages(self, user_query: str) -> list[dict]:
        """组装消息列表，加上 system prompt 和历史"""
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.session_history)
        messages.append({"role": "user", "content": user_query})
        return messages

    def execute(self, user_query: str) -> str:
        """
        非流式执行：等待完整回答后返回。

        流程：
          中间件输入处理 → 路由判断 → 工具调用 / RAG → 中间件输出处理 → 返回
        """
        # 中间件：输入处理
        processed_query = self.middleware_manager.apply_input(user_query)

        # 执行路由（路由内部已包含工具调用和 RAG）
        from agent.tools.router import route_and_answer as router_func
        raw_result = router_func(processed_query)

        # 中间件：输出处理
        final_result = self.middleware_manager.apply_output(raw_result)

        # 记录历史
        self.session_history.append({"role": "user", "content": user_query})
        self.session_history.append({"role": "assistant", "content": final_result})

        return final_result

    def execute_stream(self, user_query: str) -> tuple[Generator, list[dict]]:
        """
        流式执行：返回 (token_generator, full_messages)。

        调用方负责消费 generator（逐 token 输出），
        消费完毕后可从 full_messages 取出完整对话历史。
        """
        # 中间件：输入处理
        processed_query = self.middleware_manager.apply_input(user_query)

        # 组装消息
        messages = self._build_messages(processed_query)

        # 使用流式路由
        from agent.tools.router import route_and_answer_stream

        router_stream = route_and_answer_stream(processed_query)

        def _final_stream():
            for token in router_stream:
                yield from self.middleware_manager.apply_stream(token)

        return _final_stream(), messages

    def clear_history(self) -> "Agent":
        """清空会话历史"""
        self.session_history.clear()
        return self


# ─────────────────────────────────────────────────────────────────────────────
# AgentFactory
# ─────────────────────────────────────────────────────────────────────────────

class AgentFactory:
    """
    Agent 工厂类：创建配置好的 Agent 实例。

    使用示例：
      # 默认配置
      agent = AgentFactory.create()

      # 指定会话 ID
      agent = AgentFactory.create(session_id="user_001")

      # 开启报告生成模式（自动注入报告提示词）
      context_mw = get_context_middleware()
      context_mw.fill_context("report")
      agent = AgentFactory.create(middleware_manager=mw_manager)

      # 流式输出
      gen, msgs = agent.execute_stream("什么是贪污罪")
      for token in gen:
          print(token, end="", flush=True)
    """

    @staticmethod
    def _default_middleware_manager() -> MiddlewareManager:
        return (
            MiddlewareManager()
            .add(get_context_middleware())
            .add(get_streaming_middleware())
        )

    @staticmethod
    def create(
        session_id: str | None = None,
        system_prompt: str = "",
        middleware_manager: MiddlewareManager | None = None,
    ) -> Agent:
        mw = middleware_manager or AgentFactory._default_middleware_manager()

        ctx_mw = get_context_middleware()
        if ctx_mw.is_active():
            effective_system_prompt = ctx_mw.get_system_prompt()
        else:
            effective_system_prompt = system_prompt

        return Agent(
            middleware_manager=mw,
            session_id=session_id,
            system_prompt=effective_system_prompt,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 便捷函数（单 Agent 单会话场景）
# ─────────────────────────────────────────────────────────────────────────────

_default_agent: Agent | None = None


def get_agent(session_id: str | None = None) -> Agent:
    """获取全局单例 Agent（延迟初始化）"""
    global _default_agent
    if _default_agent is None:
        _default_agent = AgentFactory.create(session_id=session_id)
    return _default_agent


def ask(query: str) -> str:
    """快捷方法：直接问一个问题（非流式）"""
    return get_agent().execute(query)


def ask_stream(query: str) -> Generator[str, None, None]:
    """快捷方法：直接问一个问题（流式），返回 generator"""
    gen, _ = get_agent().execute_stream(query)
    return gen
