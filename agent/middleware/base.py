"""
Agent 中间件基类，定义中间件的接口规范。

中间件分两种模式：
  - transform_input:  在请求发往前，对输入做预处理
  - transform_output: 在响应返回前，对输出做后处理

所有中间件实现继承 BaseMiddleware，重写对应的方法即可。
"""

from abc import ABC, abstractmethod
from typing import Any, Generator


class BaseMiddleware(ABC):
    """中间件基类"""

    name: str = "base_middleware"

    def transform_input(self, query: str, context: dict | None = None) -> str:
        """
        在请求发往前调用，可修改 query 或 context。
        默认直接返回原值。
        """
        return query

    def transform_output(self, output: str) -> str:
        """
        在响应返回前调用，可对结果做后处理。
        默认直接返回原值。
        """
        return output

    def transform_stream(self, token: str) -> Generator[str, None, None]:
        """
        在流式输出的每个 token 发出前调用，可做过滤/增强/分词等。
        返回一个 generator，必须 yield 处理后的 token。
        默认透传。
        """
        yield token


class MiddlewareManager:
    """
    中间件管理器：按顺序执行所有中间件的 transform 方法。
    """

    def __init__(self):
        self._middlewares: list[BaseMiddleware] = []

    def add(self, middleware: BaseMiddleware) -> "MiddlewareManager":
        """链式注册中间件"""
        self._middlewares.append(middleware)
        return self

    def apply_input(self, query: str, context: dict | None = None) -> str:
        """依次执行所有中间件的 transform_input"""
        result = query
        for mw in self._middlewares:
            result = mw.transform_input(result, context)
        return result

    def apply_output(self, output: str) -> str:
        """依次执行所有中间件的 transform_output（逆序）"""
        result = output
        for mw in reversed(self._middlewares):
            result = mw.transform_output(result)
        return result

    def apply_stream(self, token: str) -> Generator[str, None, None]:
        """
        将 token 依次流经所有中间件的 transform_stream。
        每个中间件 transform_stream 返回 generator，展平后 yield。
        """
        current: Generator[str, None, None] = (t for t in [token])
        for mw in reversed(self._middlewares):
            def wrapped(chunk_iter, mw_obj):
                for chunk in chunk_iter:
                    yield from mw_obj.transform_stream(chunk)
            current = wrapped(current, mw)
        yield from current
