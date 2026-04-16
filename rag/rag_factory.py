"""
RAG 服务工厂：负责将各组件组装成可用的 RagSummarizeService。

这是唯一需要感知 VectorStore、Model、Prompt 等所有依赖的地方。
其他模块只通过 RagSummarizeService 的接口交互。
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from model.factory import chat_model
from rag.rag_service import RagSummarizeService


def create_rag_service() -> RagSummarizeService:
    """构建并返回一个完整的 RAG 摘要服务实例"""
    vector_store = VectorStoreService()
    retriever = vector_store.get_retriever()

    prompt_text = load_rag_prompts()
    prompt_template = PromptTemplate.from_template(prompt_text)

    chain = prompt_template | chat_model | StrOutputParser()

    return RagSummarizeService(
        retriever=retriever,
        prompt_template=prompt_template,
        chain=chain,
    )


# 单例，全局共享同一实例
_rag_service_instance: RagSummarizeService | None = None


def get_rag_service() -> RagSummarizeService:
    """获取全局 RAG 服务实例（延迟初始化）"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = create_rag_service()
    return _rag_service_instance
