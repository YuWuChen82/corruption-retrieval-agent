"""
一次性迁移脚本：将 data/贪污记录.csv 导入 MySQL。

用法：
  python scripts/migrate_csv_to_mysql.py [--dry-run] [--verify]

前置条件：
  1. config/mysql.yaml 配置正确
"""

import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_handler import get_db_pool
from utils.path_tool import get_abs_path


# CSV 字段顺序（与 MySQL INSERT 顺序对应）
_CSV_FIELDNAMES = [
    "案件序号", "当事人姓名", "职务/身份", "涉案金额",
    "主要犯罪事实", "判决/处理结果", "通报/宣判时间", "备注",
]

_CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS corruption_records (
        id              BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
        case_seq       VARCHAR(20)   NOT NULL,
        person_name     VARCHAR(100)  NOT NULL,
        job_title       VARCHAR(255)  DEFAULT '',
        amount          VARCHAR(100)  DEFAULT '',
        crime_facts     TEXT,
        verdict         VARCHAR(500)  DEFAULT '',
        announce_date   VARCHAR(50)   DEFAULT '',
        notes           VARCHAR(255)  DEFAULT '',
        created_at      DATETIME       DEFAULT CURRENT_TIMESTAMP,
        updated_at      DATETIME       DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_person_name(person_name),
        INDEX idx_announce_date(announce_date),
        INDEX idx_case_seq(case_seq)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
"""


def _ensure_database():
    """连接 MySQL（不指定 database），确保 corruption_db 存在"""
    from utils.config_handler import mysql_conf
    import pymysql
    try:
        conn = pymysql.connect(
            host=mysql_conf["host"],
            port=mysql_conf.get("port", 3306),
            user=mysql_conf["user"],
            password=mysql_conf["password"],
            charset=mysql_conf.get("charset", "utf8mb4"),
        )
        with conn.cursor() as cur:
            cur.execute(
                "CREATE DATABASE IF NOT EXISTS corruption_db "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            conn.commit()
        conn.close()
        print("[迁移] 数据库 corruption_db 就绪")
    except Exception as e:
        print(f"[迁移] ERROR: 无法连接 MySQL 或创建数据库: {e}")
        raise


def _create_table(pool):
    """创建表（幂等：先 DROP 再 CREATE）"""
    with pool.borrow() as conn:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS corruption_records")
            cur.execute(_CREATE_TABLE_SQL)
            conn.commit()
    print("[迁移] 表 corruption_records 创建成功")


def migrate(csv_path: str, dry_run: bool = False):
    """读取 CSV 并写入 MySQL"""
    print(f"[迁移] 读取 CSV: {csv_path}")
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[迁移] CSV 共 {len(rows)} 条记录")

    if dry_run:
        print("[迁移] DRY RUN 模式：跳过实际写入")
        for i, r in enumerate(rows, 1):
            print(f"  [{i}] {r.get('当事人姓名', '')} | {r.get('涉案金额', '')}")
        return len(rows)

    # 1. 确保数据库存在
    _ensure_database()

    # 2. 获取连接池
    pool = get_db_pool()
    if pool is None:
        print("[迁移] ERROR: MySQL 连接失败，请检查 config/mysql.yaml 配置")
        sys.exit(1)

    # 3. 创建表
    _create_table(pool)

    # 4. 逐行写入
    INSERT_SQL = """
        INSERT INTO corruption_records
          (case_seq, person_name, job_title, amount,
           crime_facts, verdict, announce_date, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    ok_count = 0
    err_count = 0
    err_examples = []
    with pool.borrow() as conn:
        with conn.cursor() as cur:
            for r in rows:
                try:
                    cur.execute(INSERT_SQL, (
                        r.get("案件序号", ""),
                        r.get("当事人姓名", ""),
                        r.get("职务/身份", ""),
                        r.get("涉案金额", ""),
                        r.get("主要犯罪事实", ""),
                        r.get("判决/处理结果", ""),
                        r.get("通报/宣判时间", ""),
                        r.get("备注", ""),
                    ))
                    ok_count += 1
                except Exception as e:
                    err_count += 1
                    if len(err_examples) < 3:
                        err_examples.append(f"  [{r.get('当事人姓名', '')}]: {e}")
            conn.commit()

    print(f"[迁移] 完成：成功 {ok_count} 条，失败 {err_count} 条")
    if err_examples:
        print("[迁移] 失败示例:")
        for ex in err_examples:
            print(ex)
    return ok_count


def verify(csv_path: str, pool):
    """迁移后校验 CSV 行数与 MySQL 行数是否一致"""
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        csv_count = sum(1 for _ in csv.DictReader(f))

    with pool.borrow() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM corruption_records")
            mysql_count = cur.fetchone()[0]

    print(f"[校验] CSV 行数:  {csv_count}")
    print(f"[校验] MySQL 行数: {mysql_count}")
    if csv_count == mysql_count:
        print("[校验] PASS: 行数一致")
    else:
        print(f"[校验] FAIL: 行数不一致 (差异 {abs(csv_count - mysql_count)})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CSV → MySQL 迁移脚本")
    parser.add_argument("--dry-run", action="store_true", help="仅打印，不写入")
    parser.add_argument("--verify", action="store_true", help="迁移后校验")
    args = parser.parse_args()

    csv_path = get_abs_path("data/贪污记录.csv")
    if not os.path.exists(csv_path):
        print(f"[迁移] ERROR: CSV 文件不存在: {csv_path}")
        sys.exit(1)

    migrate(csv_path, dry_run=args.dry_run)

    if args.verify and not args.dry_run:
        pool = get_db_pool()
        if pool:
            verify(csv_path, pool)
        else:
            print("[校验] SKIP: MySQL 不可用")
