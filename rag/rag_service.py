"""
RAG 摘要服务：接收用户提问，通过向量检索找到参考资料，
将问题与参考资料提交给模型，由模型总结回复。

职责单一：不关心工具、不关心路由、不关心调用方式。
"""

from typing import Generator
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable


class RagSummarizeService:
    def __init__(
        self,
        retriever: Runnable,
        prompt_template: PromptTemplate,
        chain: Runnable,
    ):
        """
        依赖由外部注入，保持职责单一。

        Args:
            retriever:      向量检索器，接收 str，返回 list[Document]
            prompt_template: 提示词模板，需接受 {input} 和 {context} 两个变量
            chain:           已组装好的 LCEL 流水线，接收 dict，返回 str
        """
        self._retriever = retriever
        self._template = prompt_template
        self._chain = chain

    def retrieve(self, query: str) -> list[Document]:
        """执行向量检索，返回相关文档列表"""
        return self._retriever.invoke(query)

    def summarize(self, query: str, context_docs: list[Document] | None = None) -> str:
        """
        核心方法：将参考资料拼入提示词，发给模型，返回模型回复。

        Args:
            query:        用户原始提问
            context_docs: 可选，若不传则自动检索
        """
        if context_docs is None:
            context_docs = self.retrieve(query)

        context = "\n".join(
            f"【参考资料{i}】: {doc.page_content} | 元数据：{doc.metadata}"
            for i, doc in enumerate(context_docs, 1)
        )

        return self._chain.invoke({
            "input": query,
            "context": context,
        })

    def summarize_stream(self, query: str) -> Generator[str, None, None]:
        """
        流式版本：将参考资料拼入提示词，LLM 流式生成。

        Yields:
            str: 逐 token 的文本内容
        """
        docs = self.retrieve(query)
        context = "\n".join(
            f"【参考资料{i}】: {doc.page_content} | 元数据：{doc.metadata}"
            for i, doc in enumerate(docs, 1)
        )

        try:
            stream_resp = self._chain.stream({
                "input": query,
                "context": context,
            })
            for chunk in stream_resp:
                # chain.stream() yields TextAccessor 或 dict，需提取文本
                token = ""
                if hasattr(chunk, "text"):
                    token = chunk.text or ""
                elif isinstance(chunk, dict):
                    token = chunk.get("content", "") or chunk.get("text", "")
                elif isinstance(chunk, str):
                    token = chunk
                if token:
                    yield token
        except Exception:
            # 流式失败，降级到完整字符串
            yield self.summarize(query)
