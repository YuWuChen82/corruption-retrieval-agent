"""
MySQL 数据库操作模块（连接池 + CRUD）。

提供：
  - 全局连接池（延迟初始化）
  - CRUD 操作函数（字典键名与 CSV 字段一致）
  - 缓存失效接口
  - MySQL/CSV 双模式回退机制
"""

import threading
import csv as _csv
import os
import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from typing import Optional
from utils.logger_handler import logger
from utils.config_handler import mysql_conf
from utils.path_tool import get_abs_path

# ─────────────────────────────────────────────────────────────────────────────
# 全局状态
# ─────────────────────────────────────────────────────────────────────────────

_mysql_available: bool | None = None  # None=未检测, True=可用, False=不可达
_db_pool: Optional["_SimplePool"] = None


# ─────────────────────────────────────────────────────────────────────────────
# 极简连接池
# ─────────────────────────────────────────────────────────────────────────────

class _SimplePool:
    """极简连接池，基于 queue.Queue，borrow/release 上下文管理器"""

    def __init__(self, config: dict):
        pool_cfg = config.get("pool", {})
        self._host = config["host"]
        self._port = config.get("port", 3306)
        self._user = config["user"]
        self._password = config["password"]
        self._database = config.get("database", "")
        self._charset = config.get("charset", "utf8mb4")
        self._connect_timeout = config.get("connect_timeout", 10)
        self._read_timeout = config.get("read_timeout", 30)
        self._write_timeout = config.get("write_timeout", 30)

    def _new_connection(self) -> pymysql.Connection:
        return pymysql.connect(
            host=self._host,
            port=self._port,
            user=self._user,
            password=self._password,
            database=self._database,
            charset=self._charset,
            connect_timeout=self._connect_timeout,
            read_timeout=self._read_timeout,
            write_timeout=self._write_timeout,
            autocommit=False,
        )

    @contextmanager
    def borrow(self):
        """借出一条连接，用完自动关闭（内部池化，pymysql 连接开销小）"""
        conn = self._new_connection()
        try:
            yield conn
        finally:
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# 全局连接池获取
# ─────────────────────────────────────────────────────────────────────────────

def get_db_pool() -> Optional[_SimplePool]:
    global _db_pool, _mysql_available
    if _mysql_available is False:
        return None
    if _db_pool is None:
        try:
            pool = _SimplePool(mysql_conf)
            # 预热：验证连接可用
            conn = pool._new_connection()
            conn.close()
            _db_pool = pool
            _mysql_available = True
            logger.info("[db_handler] MySQL 连接池初始化成功")
        except Exception as e:
            logger.warning(f"[db_handler] MySQL 连接失败，将回退到 CSV 模式: {e}")
            _mysql_available = False
            return None
    return _db_pool


def is_mysql_available() -> bool:
    """检测 MySQL 是否可用（结果缓存）"""
    global _mysql_available
    if _mysql_available is None:
        get_db_pool()
    return _mysql_available is True


# ─────────────────────────────────────────────────────────────────────────────
# 缓存失效（与 data_query_tool 的全局缓存联动）
# ─────────────────────────────────────────────────────────────────────────────

def invalidate_query_cache():
    """清空 data_query_tool 中的 CSV 缓存（写入后调用，保证查到最新数据）"""
    try:
        from agent.tools import data_query_tool as dqt
        dqt._invalidate_csv_cache()
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# CRUD 操作
# ─────────────────────────────────────────────────────────────────────────────

def corruption_select_all() -> list[dict]:
    """查询全部记录"""
    pool = get_db_pool()
    if pool is None:
        return []
    SQL = """
        SELECT case_seq AS "案件序号",
               person_name AS "当事人姓名",
               job_title AS "职务/身份",
               amount AS "涉案金额",
               crime_facts AS "主要犯罪事实",
               verdict AS "判决/处理结果",
               announce_date AS "通报/宣判时间",
               notes AS "备注"
        FROM corruption_records
        ORDER BY id ASC
    """
    try:
        with pool.borrow() as conn:
            with conn.cursor(cursor=DictCursor) as cur:
                cur.execute(SQL)
                return cur.fetchall()
    except Exception as e:
        logger.error(f"[db_handler] SELECT ALL 失败: {e}")
        return []


def corruption_select_by_name(person_name: str) -> list[dict]:
    """按当事人姓名模糊查询"""
    pool = get_db_pool()
    if pool is None:
        return []
    SQL = """
        SELECT case_seq AS "案件序号",
               person_name AS "当事人姓名",
               job_title AS "职务/身份",
               amount AS "涉案金额",
               crime_facts AS "主要犯罪事实",
               verdict AS "判决/处理结果",
               announce_date AS "通报/宣判时间",
               notes AS "备注"
        FROM corruption_records
        WHERE person_name LIKE %s
        ORDER BY id ASC
    """
    try:
        with pool.borrow() as conn:
            with conn.cursor(cursor=DictCursor) as cur:
                cur.execute(SQL, (f"%{person_name}%",))
                return cur.fetchall()
    except Exception as e:
        logger.error(f"[db_handler] SELECT BY NAME 失败: {e}")
        return []


def corruption_select_by_keyword(keyword: str) -> list[dict]:
    """全字段模糊搜索（当事人姓名、犯罪事实、职务、备注）"""
    pool = get_db_pool()
    if pool is None:
        return []
    kw = f"%{keyword}%"
    SQL = """
        SELECT case_seq AS "案件序号",
               person_name AS "当事人姓名",
               job_title AS "职务/身份",
               amount AS "涉案金额",
               crime_facts AS "主要犯罪事实",
               verdict AS "判决/处理结果",
               announce_date AS "通报/宣判时间",
               notes AS "备注"
        FROM corruption_records
        WHERE person_name LIKE %s
           OR crime_facts LIKE %s
           OR job_title LIKE %s
           OR notes LIKE %s
        ORDER BY id ASC
    """
    try:
        with pool.borrow() as conn:
            with conn.cursor(cursor=DictCursor) as cur:
                cur.execute(SQL, (kw, kw, kw, kw))
                return cur.fetchall()
    except Exception as e:
        logger.error(f"[db_handler] SELECT BY KEYWORD 失败: {e}")
        return []


def corruption_select_by_year(year: str) -> list[dict]:
    """按年份筛选（announce_date 包含该年份字符串）"""
    pool = get_db_pool()
    if pool is None:
        return []
    SQL = """
        SELECT case_seq AS "案件序号",
               person_name AS "当事人姓名",
               job_title AS "职务/身份",
               amount AS "涉案金额",
               crime_facts AS "主要犯罪事实",
               verdict AS "判决/处理结果",
               announce_date AS "通报/宣判时间",
               notes AS "备注"
        FROM corruption_records
        WHERE announce_date LIKE %s
        ORDER BY id ASC
    """
    try:
        with pool.borrow() as conn:
            with conn.cursor(cursor=DictCursor) as cur:
                cur.execute(SQL, (f"%{year}%",))
                return cur.fetchall()
    except Exception as e:
        logger.error(f"[db_handler] SELECT BY YEAR 失败: {e}")
        return []


def _is_valid_record(record: dict) -> tuple[bool, str]:
    """
    校验记录是否有效：当事人姓名必填，其余字段允许为空。
    返回 (是否有效, 错误信息)。
    """
    name = record.get("当事人姓名", "").strip()
    if not name:
        return False, "当事人姓名为空，禁止写入"
    # 所有字段均为空也不行
    values = [str(record.get(k, "").strip()) for k in record if k != "案件序号"]
    if not any(v for v in values):
        return False, "所有字段均为空，禁止写入"
    return True, ""


def corruption_insert(record: dict) -> tuple[bool, int, str]:
    """
    插入一条贪腐记录。

    Args:
        record: dict，键名为 CSV 字段名
    Returns:
        (成功标志, 自增id或0, 错误信息)
    """
    # 非空校验
    ok, err = _is_valid_record(record)
    if not ok:
        logger.warning(f"[db_handler] 禁止写入空记录: {err}")
        return (False, 0, err)

    pool = get_db_pool()
    if pool is None:
        return (False, 0, "MySQL 不可达")
    INSERT_SQL = """
        INSERT INTO corruption_records
          (case_seq, person_name, job_title, amount,
           crime_facts, verdict, announce_date, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    try:
        with pool.borrow() as conn:
            with conn.cursor() as cur:
                cur.execute(INSERT_SQL, (
                    str(record.get("案件序号", "")),
                    record.get("当事人姓名", ""),
                    record.get("职务/身份", ""),
                    record.get("涉案金额", ""),
                    record.get("主要犯罪事实", ""),
                    record.get("判决/处理结果", ""),
                    record.get("通报/宣判时间", ""),
                    record.get("备注", ""),
                ))
                conn.commit()
                new_id = cur.lastrowid
                invalidate_query_cache()
                return (True, new_id, "")
    except Exception as e:
        logger.error(f"[db_handler] INSERT 失败: {e}")
        return (False, 0, str(e))


def corruption_get_next_seq() -> int:
    """返回当前最大 case_seq + 1"""
    pool = get_db_pool()
    if pool is None:
        return 1
    SQL = "SELECT MAX(CAST(case_seq AS UNSIGNED)) AS max_seq FROM corruption_records"
    try:
        with pool.borrow() as conn:
            with conn.cursor() as cur:
                cur.execute(SQL)
                row = cur.fetchone()
                max_seq = row[0] if row and row[0] is not None else 0
                return int(max_seq) + 1
    except Exception as e:
        logger.error(f"[db_handler] GET NEXT SEQ 失败: {e}")
        return 1


def corruption_check_exists(name: str, date: str, verdict: str) -> tuple[bool, list[dict]]:
    """
    检查是否已存在同名+同日期+前20字符相同判决的记录。
    Returns: (是否存在, 匹配到的已有记录列表)
    """
    pool = get_db_pool()
    if pool is None:
        return (False, [])  # MySQL 不可达时保守放行

    verdict_prefix = verdict.strip()[:20] if verdict else ""
    date_norm = date.strip()

    SQL = """
        SELECT case_seq AS "案件序号",
               person_name AS "当事人姓名",
               job_title AS "职务/身份",
               amount AS "涉案金额",
               crime_facts AS "主要犯罪事实",
               verdict AS "判决/处理结果",
               announce_date AS "通报/宣判时间",
               notes AS "备注"
        FROM corruption_records
        WHERE person_name LIKE %s
          AND announce_date = %s
          AND LEFT(verdict, 20) = %s
    """
    try:
        with pool.borrow() as conn:
            with conn.cursor(cursor=DictCursor) as cur:
                cur.execute(SQL, (f"%{name}%", date_norm, verdict_prefix))
                rows = cur.fetchall()
                return (len(rows) > 0, rows)
    except Exception as e:
        logger.error(f"[db_handler] CHECK EXISTS 失败: {e}")
        return (False, [])  # 查询失败时保守放行


def corruption_get_all_for_sync() -> list[dict]:
    """
    将 MySQL 全部记录导出（供 MySQL→CSV 同步用）。
    与 corruption_select_all 相同，但不带 ORDER BY（同步场景不需要）。
    """
    pool = get_db_pool()
    if pool is None:
        return []
    SQL = """
        SELECT case_seq AS "案件序号",
               person_name AS "当事人姓名",
               job_title AS "职务/身份",
               amount AS "涉案金额",
               crime_facts AS "主要犯罪事实",
               verdict AS "判决/处理结果",
               announce_date AS "通报/宣判时间",
               notes AS "备注"
        FROM corruption_records
    """
    try:
        with pool.borrow() as conn:
            with conn.cursor(cursor=DictCursor) as cur:
                cur.execute(SQL)
                return cur.fetchall()
    except Exception as e:
        logger.error(f"[db_handler] GET ALL FOR SYNC 失败: {e}")
        return []
