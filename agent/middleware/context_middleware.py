"""
上下文注入中间件：专门用于报告生成场景。

当 Agent 切换到报告生成模式时，中间件自动注入：
  1. 报告生成的系统提示词（覆盖默认 prompt）
  2. 额外的上下文信息（如时间范围、关注点等）
"""

from agent.middleware.base import BaseMiddleware
from utils.path_tool import get_abs_path
import os


class ContextMiddleware(BaseMiddleware):
    """
    上下文注入中间件。

    使用方式：
      调用 fill_context() 后，后续所有请求自动携带报告场景上下文。
      调用 clear_context() 可清除上下文，恢复默认状态。
    """

    name = "context_middleware"

    # 报告场景对应的提示词文件（key → 文件名，不含后缀）
    PROMPT_FILES = {
        "report": "report_prompt",      # prompts/report_prompt.txt
        "main":   "main_prompt",        # prompts/main_prompt.txt
    }

    def __init__(self):
        self._active_context: dict | None = None   # 当前激活的上下文
        self._context_stack: list[dict] = []       # 支持嵌套上下文

    def fill_context(
        self,
        context_name: str = "report",
        extra_vars: dict | None = None,
    ) -> "ContextMiddleware":
        """
        激活指定场景的上下文，自动注入对应的系统提示词。

        Args:
            context_name:  场景名，对应 PROMPT_FILES 中的 key
            extra_vars:    额外变量，可在 prompt 中使用 {key} 引用
        """
        prompt_file = self.PROMPT_FILES.get(context_name)
        system_prompt = ""

        if prompt_file:
            prompt_path = get_abs_path(f"prompts/{prompt_file}.txt")
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    system_prompt = f.read()

        self._active_context = {
            "name": context_name,
            "system_prompt": system_prompt,
            "extra_vars": extra_vars or {},
        }
        return self

    def clear_context(self) -> "ContextMiddleware":
        """清除当前上下文，恢复默认状态"""
        self._active_context = None
        return self

    def push_context(self, context: dict) -> "ContextMiddleware":
        """压栈保存当前上下文，支持嵌套"""
        if self._active_context:
            self._context_stack.append(self._active_context)
        self._active_context = context
        return self

    def pop_context(self) -> "ContextMiddleware":
        """弹出栈顶上下文"""
        if self._context_stack:
            self._active_context = self._context_stack.pop()
        else:
            self._active_context = None
        return self

    def get_system_prompt(self) -> str:
        """获取当前激活的系统提示词（供 AgentFactory 使用）"""
        if self._active_context:
            return self._active_context["system_prompt"]
        return ""

    def transform_input(self, query: str, context: dict | None = None) -> str:
        """
        输入转换：保持 query 不变，仅在 context 中携带上下文信息。
        实际注入由 get_system_prompt() 在 AgentFactory 中使用。
        """
        if context is not None and self._active_context:
            context["_middleware_active"] = True
            context["_middleware_name"] = self._active_context["name"]
            context["_middleware_vars"] = self._active_context["extra_vars"]
        return query

    def transform_output(self, output: str) -> str:
        """输出转换：报告场景可在此做格式化等后处理"""
        return output

    def is_active(self) -> bool:
        """当前是否有激活的上下文"""
        return self._active_context is not None


# 全局单例，供各模块共享
_context_middleware_instance: ContextMiddleware | None = None


def get_context_middleware() -> ContextMiddleware:
    global _context_middleware_instance
    if _context_middleware_instance is None:
        _context_middleware_instance = ContextMiddleware()
    return _context_middleware_instance
