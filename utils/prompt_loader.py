"""
提示词加载器：从配置文件指定的路径加载提示词模板。

只暴露被外部模块实际调用的函数。
"""

from utils.config_handler import prompts_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


def load_rag_prompts() -> str:
    """加载 RAG 摘要提示词（被 rag_factory 调用）"""
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts] yaml配置项中缺少 rag_summarize_prompt_path")
        raise e

    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts] 解析rag提示词出错：{str(e)}")
        raise e
