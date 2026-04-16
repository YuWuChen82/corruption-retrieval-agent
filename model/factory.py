"""
模型工厂：提供全局单例 chat_model 和 embed_model。

职责单一：只负责初始化，不做其他逻辑。
"""

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


def _create_chat_model() -> ChatTongyi:
    return ChatTongyi(model=rag_conf["chat_model_name"])


def _create_embed_model() -> DashScopeEmbeddings:
    return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = _create_chat_model()
embed_model = _create_embed_model()
