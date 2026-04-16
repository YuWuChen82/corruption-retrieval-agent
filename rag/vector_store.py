from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
from utils.file_handler import pdf_loader, txt_loader, csv_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.db_handler import is_mysql_available, corruption_get_all_for_sync
from utils.logger_handler import logger
import os
import hashlib


# MySQL → CSV 同步字段顺序
_SYNC_CSV_FIELDS = [
    "案件序号", "当事人姓名", "职务/身份", "涉案金额",
    "主要犯罪事实", "判决/处理结果", "通报/宣判时间", "备注",
]


def _sync_mysql_to_csv() -> str | None:
    """
    将 MySQL 全部记录同步覆盖到 data/贪污记录.csv。
    用于向量库加载时同步最新数据（Chroma 仍读 CSV）。
    返回被同步的 CSV 路径；MySQL 不可达时返回 None。
    """
    if not is_mysql_available():
        return None

    records = corruption_get_all_for_sync()
    if not records:
        logger.info("[vector_store] MySQL 无数据，跳过同步")
        return None

    csv_path = get_abs_path("data/贪污记录.csv")
    try:
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            import csv as _csv
            writer = _csv.DictWriter(f, fieldnames=_SYNC_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(records)
        # 清除 md5_hex_store 中该文件的旧记录，强制向量库下次重新加载
        _clear_csv_md5(csv_path)
        logger.info(f"[vector_store] MySQL→CSV 同步完成 ({len(records)} 条)，已清除 MD5 记录")
        return csv_path
    except Exception as e:
        logger.warning(f"[vector_store] MySQL→CSV 同步失败: {e}")
        return None


def _clear_csv_md5(csv_path: str):
    """清除 md5_hex_store 中指定文件的 MD5 记录（强制下次重新向量化）"""
    md5_file = get_abs_path(chroma_conf["md5_hex_store"])
    if not os.path.exists(md5_file):
        return
    try:
        with open(csv_path, "rb") as f:
            new_md5 = hashlib.md5(f.read()).hexdigest()
        with open(md5_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        lines = [l for l in lines if l.strip() != new_md5]
        with open(md5_file, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception as e:
        logger.warning(f"[vector_store] 清除 CSV MD5 记录失败: {e}")


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        """
        从数据文件夹内读取数据文件，转为向量存入向量库。
        每次加载前先将 MySQL 数据同步到 CSV（保持向量库与数据库一致）。
        要计算文件的MD5做去重
        :return: None
        """

        # ── Step 0：MySQL → CSV 同步 ──────────────────────────────────
        _sync_mysql_to_csv()

        def check_md5_hex(md5_for_check: str):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                # 创建文件
                open(get_abs_path(chroma_conf["md5_hex_store"]), "w", encoding="utf-8").close()
                return False            # md5 没处理过

            with open(get_abs_path(chroma_conf["md5_hex_store"]), "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line == md5_for_check:
                        return True     # md5 处理过

                return False            # md5 没处理过

        def save_md5_hex(md5_for_check: str):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path: str):
            if read_path.endswith("txt"):
                return txt_loader(read_path)

            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            if read_path.endswith("csv"):
                return csv_loader(read_path)
            return []

        allowed_files_path: tuple[...] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        for path in allowed_files_path:
            # 获取文件的MD5
            md5_hex = get_file_md5_hex(path)

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                # 将内容存入向量库
                self.vector_store.add_documents(split_document)

                # 记录这个已经处理好的文件的md5，避免下次重复加载
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue


if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("5593")
    for r in res:
        print(r.page_content)
        print("-"*20)


