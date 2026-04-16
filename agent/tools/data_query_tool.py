"""
数据查询工具：精确查询贪污记录 CSV，提供排序、筛选、统计能力。
输出 Markdown 表格，在终端和 Web 界面均可良好呈现。

所有工具通过 @auto_tool 装饰器注册，无需手动添加到任何列表。
"""

import csv
import re
import os
import threading
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.db_handler import (
    is_mysql_available,
    corruption_select_all,
    corruption_select_by_name,
    corruption_select_by_keyword,
    corruption_select_by_year,
    corruption_insert,
    corruption_get_next_seq,
    invalidate_query_cache,
)
from agent.tools.registry import auto_tool, TOOL_REGISTRY


# ─────────────────────────────────────────────────────────────────────────────
# 内部辅助函数
# ─────────────────────────────────────────────────────────────────────────────

_csv_data_cache = None


def _load_data() -> list[dict]:
    """
    统一数据加载入口：MySQL 优先，CSV 降级。
    MySQL 可用时走数据库；不可达时回退到带缓存的 CSV 读取。
    """
    if is_mysql_available():
        rows = corruption_select_all()
        if rows:  # MySQL 返回了有效数据
            return rows
        logger.warning("[data_query_tool] MySQL 查询返回空，尝试 CSV")
    # CSV 降级兜底
    return _load_csv_data()


def _write_record(record: dict) -> bool:
    """
    统一写入入口：MySQL 优先，CSV 降级。
    写入后自动失效缓存，保证下次查询拿到最新数据。
    """
    if is_mysql_available():
        ok, new_id, err = corruption_insert(record)
        if ok:
            logger.info(f"[data_query_tool] MySQL 写入成功 (id={new_id})")
            return True
        logger.warning(f"[data_query_tool] MySQL 写入失败，降级到 CSV: {err}")
    # CSV 降级
    return _append_to_csv(record)


def _load_csv_data() -> list[dict]:
    """读取CSV文件，返回字典列表（带缓存）"""
    global _csv_data_cache
    if _csv_data_cache is not None:
        return _csv_data_cache

    csv_path = get_abs_path("data/贪污记录.csv")
    records = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(row)
        _csv_data_cache = records
    except Exception as e:
        logger.error(f"读取CSV失败: {e}")
        _csv_data_cache = []
    return _csv_data_cache


def _get_next_seq() -> int:
    """获取下一个案件序号（MySQL 优先，CSV 兜底）"""
    if is_mysql_available():
        seq = corruption_get_next_seq()
        if seq > 0:
            return seq
        logger.warning("[data_query_tool] MySQL 获取序号失败，尝试 CSV")
    # CSV 兜底
    records = _load_csv_data()
    if not records:
        return 1
    max_seq = 0
    for r in records:
        try:
            seq = int(r.get("案件序号", 0))
            if seq > max_seq:
                max_seq = seq
        except (ValueError, TypeError):
            pass
    return max_seq + 1


def _parse_amount(amount_str: str) -> float:
    """从涉案金额字符串中提取最大金额（单位统一转为万元）"""
    if not isinstance(amount_str, str):
        return -1.0
    amount_str = amount_str.strip()
    if amount_str in ("未公开", "巨额", ""):
        return -1.0
    candidates = []
    for match in re.finditer(r"([\d.]+)亿", amount_str):
        candidates.append(float(match.group(1)) * 10000)
    for match in re.finditer(r"([\d.]+)万", amount_str):
        candidates.append(float(match.group(1)))
    return max(candidates) if candidates else -1.0


def _fmt_amount(amount_str: str) -> str:
    """将金额字符串转为"X亿元"或"X万元"格式，便于表格展示"""
    num = _parse_amount(amount_str)
    if num < 0:
        return amount_str
    if num >= 10000:
        return f"{num / 10000:.2f}亿元" if num % 10000 == 0 else f"{num / 10000:.1f}亿元"
    return f"{num:.0f}万元"


def _build_markdown_table(records: list[dict], columns: list[str] | None = None) -> str:
    default_cols = ["案件序号", "当事人姓名", "职务/身份", "涉案金额", "判决/处理结果", "通报/宣判时间"]
    cols = columns or default_cols
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for r in records:
        cells = []
        for col in cols:
            val = r.get(col, "").strip()
            if col == "涉案金额":
                val = _fmt_amount(val)
            val = val.replace("|", "｜")
            cells.append(val)
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)


def _sort_records(records: list[dict], order_by: str) -> list[dict]:
    """按指定规则排序记录"""
    for r in records:
        r["_金额数值"] = _parse_amount(r.get("涉案金额", ""))
    if order_by == "amount_desc":
        return sorted(records, key=lambda x: x["_金额数值"], reverse=True)
    if order_by == "amount_asc":
        return sorted(records, key=lambda x: x["_金额数值"])
    if order_by == "date_desc":
        return sorted(records, key=lambda x: x.get("通报/宣判时间", ""), reverse=True)
    if order_by == "date_asc":
        return sorted(records, key=lambda x: x.get("通报/宣判时间", ""))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 工具定义（通过 @auto_tool 注册一次，全局自动生效）
# ─────────────────────────────────────────────────────────────────────────────


@auto_tool(
    description="按金额或时间排序贪污记录。order_by 可选：amount_desc（金额从高到低）、amount_asc（金额从低到高）、date_desc（时间从新到旧）、date_asc（时间从旧到新）。top_n 指定返回条数。返回格式化的 Markdown 表格。",
    patterns=[
        r"最高", r"最低", r"最多", r"最少",
        r"第[一二三四五六七八九十百0-9\d]+",
        r"排名前", r"排序", r"金额最高", r"涉案最多", r"涉案最少",
        r"最[多高少大长前后]",
    ],
    extract_params="rank",
)
def rank_corruption_records(order_by: str = "amount_desc", top_n: int = 10) -> str:
    """按指定排序方式返回贪污记录，常用于"金额最高""最新案件"等比较类问题。"""
    records = _load_data()
    if not records:
        return "**数据文件为空或读取失败**"

    ranked = _sort_records(records, order_by)
    ranked = [r for r in ranked if r["_金额数值"] > 0]
    result = ranked[:top_n]

    if not result:
        return "**无符合条件的记录**"

    label_map = {
        "amount_desc": "金额从高到低",
        "amount_asc": "金额从低到高",
        "date_desc": "时间从新到旧",
        "date_asc": "时间从旧到新",
    }
    label = label_map.get(order_by, order_by)

    table = _build_markdown_table(result)
    return (
        f"### {label}（共 {len(result)} 条，数据库共 {len(records)} 条）\n\n"
        f"{table}\n\n"
        f"_以上结果按 {label} 排序_"
    )


@auto_tool(
    description='根据当事人姓名精确查找贪污记录，返回该人员的所有案件信息（职务、涉案金额、判决结果、通报时间等）。用于"XXX的判决""XXX涉案"等问题。',
    patterns=[
        # 人名 + 明确的法律关键词（2-8个中文字符，排除纯数字年份）
        r"[\u4e00-\u9fa5]{2,8}的判决",
        r"[\u4e00-\u9fa5]{2,8}涉案",
        r"[\u4e00-\u9fa5]{2,8}的情况",
        r"[\u4e00-\u9fa5]{2,8}的(案件|犯罪|贪腐|贪污|腐败)",
        r"关于[\u4e00-\u9fa5]{2,8}的",
        r"[\u4e00-\u9fa5]{2,8}是什么罪",
    ],
    extract_params="person_name",
)
def query_corruption_by_name(person_name: str) -> str:
    """按人名精确查找案件，常用于"XXX的判决是什么""XXX涉案金额"等问题。"""
    if not person_name or not person_name.strip():
        return "**未提供当事人姓名，无法查询**"

    person_name = person_name.strip()

    # MySQL 优先，直接在数据库层做 LIKE 过滤（减少数据传输）
    if is_mysql_available():
        matched = corruption_select_by_name(person_name)
    else:
        records = _load_csv_data()
        matched = [r for r in records if person_name in r.get("当事人姓名", "")]

    if not matched:
        return f"**未找到当事人「{person_name}」的相关记录**"

    table = _build_markdown_table(matched)
    return (
        f"### 查询结果：{person_name}（共 {len(matched)} 条）\n\n"
        f"{table}\n\n"
        f"_以上为数据库中「{person_name}」的全部相关记录_"
    )


@auto_tool(
    description="根据关键词全文搜索贪污记录，关键词可匹配当事人姓名、犯罪事实、涉案金额、职务等字段。返回格式化的 Markdown 表格。",
    patterns=[
        r"\d{4}-\d{2}",
        r"地区", r"省|市|县|区|国企|央企",
        r"县委|市委|省政府|县委|市长",
        r"案子|案例",
        r"查找.*",
    ],
    extract_params="keyword",
)
def search_corruption_records(keyword: str) -> str:
    """按关键词全文搜索，常用于"2026年的案件""某地区的案例"等模糊查询。"""
    if not keyword or not keyword.strip():
        return "**未提供搜索关键词，无法查询**"

    keyword = keyword.strip()

    # MySQL 优先，4字段 LIKE 过滤
    if is_mysql_available():
        matched = corruption_select_by_keyword(keyword)
    else:
        records = _load_csv_data()
        matched = [
            r for r in records
            if keyword in r.get("当事人姓名", "")
            or keyword in r.get("主要犯罪事实", "")
            or keyword in r.get("涉案金额", "")
            or keyword in r.get("职务/身份", "")
            or keyword in r.get("备注", "")
        ]

    if not matched:
        return f"**未找到关键词「{keyword}」的相关记录**"

    table = _build_markdown_table(matched)
    return (
        f"### 搜索结果：{keyword}（共 {len(matched)} 条）\n\n"
        f"{table}\n\n"
        f"_以上为数据库中包含「{keyword}」的记录_"
    )


@auto_tool(
    description='返回贪污记录库的全部记录，或按年份过滤。year 参数指定年份（如"2026"）。返回格式化的 Markdown 表格，按涉案金额从高到低排序。',
    patterns=[
        r"全部案件", r"所有案件",
        r"有哪些案件", r"哪些案件",
        r"有哪些重要", r"有哪些重大", r"哪些重要", r"哪些重大",
        r"全部记录", r"所有记录",
        r"列出.*记录", r"展示.*记录",
        r"\d{4}年的案件", r"\d{4}年有哪些", r"\d{4}年的.*案件",
    ],
    extract_params="year",
)
def get_all_corruption_records(year: str = "") -> str:
    """返回全部（或指定年份的）贪污记录，常用于"有哪些案件""某年有哪些案件"等查询。"""
    # MySQL 优先，按年份过滤
    if is_mysql_available():
        all_records = corruption_select_all()
    else:
        all_records = _load_csv_data()

    records = all_records
    if year:
        records = [r for r in records if year in r.get("通报/宣判时间", "")]

    ranked = _sort_records(records, "amount_desc")
    table = _build_markdown_table(ranked)
    year_hint = f"（{year}年）" if year else ""
    return (
        f"### 全部贪污记录{year_hint}（共 {len(ranked)} 条，数据库共 {len(records)} 条）\n\n"
        f"{table}\n\n"
        f"_按涉案金额从高到低排序_"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 自动补全工具（新增）
# ─────────────────────────────────────────────────────────────────────────────

# 写锁：防止并发写入导致 CSV 损坏（CSV 降级路径用）
_csv_write_lock = threading.Lock()


def _invalidate_csv_cache():
    """清空 CSV 数据缓存，下次查询时重新加载"""
    global _csv_data_cache
    _csv_data_cache = None


def _append_to_csv(record: dict) -> bool:
    """
    将一条记录追加到 CSV 末尾（线程安全）。
    record 包含：当事人姓名、职务/身份、涉案金额、
                主要犯罪事实、判决/处理结果、通报/处理时间、备注
    返回 True 表示成功，False 表示失败。
    """
    # 非空校验
    name = record.get("当事人姓名", "").strip()
    if not name:
        logger.warning("[data_query_tool] 当事人为空，禁止写入 CSV")
        return False
    values = [str(record.get(k, "").strip()) for k in record if k != "案件序号"]
    if not any(v for v in values):
        logger.warning("[data_query_tool] 所有字段均为空，禁止写入 CSV")
        return False

    csv_path = get_abs_path("data/贪污记录.csv")
    fieldnames = ["案件序号", "当事人姓名", "职务/身份", "涉案金额",
                  "主要犯罪事实", "判决/处理结果", "通报/宣判时间", "备注"]

    with _csv_write_lock:
        try:
            file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
            with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(record)
            _invalidate_csv_cache()
            return True
        except Exception as e:
            logger.error(f"追加CSV记录失败: {e}")
            return False


# ── LLM 辅助：从搜索结果中提取贪腐信息 ───────────────────────────────────────

def _extract_case_info_from_text(person_name: str, text: str) -> dict | None:
    """
    从网页文本 / 搜索摘要中提取贪腐案件信息。
    返回 dict 或 None（未检测到有效贪腐信息）。
    """
    import re as _re

    info = {
        "当事人姓名": person_name,
        "职务/身份": "",
        "涉案金额": "",
        "主要犯罪事实": "",
        "判决/处理结果": "",
        "通报/宣判时间": "",
        "备注": "联网自动补全",
    }

    has_content = False

    # 职务（常见的表述）
    job_patterns = [
        r"(?:原|前|现任|当.{0,4})?([^\s，,。.]{2,15})(?:省|市|县|区|委|书记|副书记|长|主任|主席|总裁|总经理|董事长|副总理|部长|副部长|局长|处长|校长|院长|厅长|队长)",
        r"(?:曾任|历任|担任)([^，。,\n]{5,30})?(?:书记|主席|总裁|总经理|董事长|长)",
    ]
    for p in job_patterns:
        m = _re.search(p, text)
        if m:
            info["职务/身份"] = m.group(0).strip()[:50]
            has_content = True
            break

    # 涉案金额（多种格式）
    amount_candidates = []
    for p in [
        r"([\d.]+)亿余?元",
        r"([\d.]+)万[余]?元",
        r"贪污([\d.]+)[亿万]?",
        r"受贿([\d.]+)[亿万]?",
        r"涉案金额[为:]?[^\d]*?([\d.,]+)[亿万]?",
        r"([\d,]+)万元",
    ]:
        for m in _re.finditer(p, text):
            val = m.group(1).replace(",", "")
            try:
                if "亿" in m.group(0) or ("万" in m.group(0) and float(val) > 100):
                    amount_candidates.append(float(val) * 10000)
                else:
                    amount_candidates.append(float(val))
            except ValueError:
                pass
    if amount_candidates:
        max_amt = max(amount_candidates)
        if max_amt >= 10000:
            info["涉案金额"] = f"{max_amt / 10000:.2f}亿元"
        else:
            info["涉案金额"] = f"{max_amt:.0f}万元"
        has_content = True

    # 主要犯罪事实（截取相关段落）
    crime_keywords = ["贪污", "受贿", "挪用", "套取", "侵占", "非法占有", "行贿", "滥用职权"]
    for kw in crime_keywords:
        idx = text.find(kw)
        if idx >= 0:
            snippet = text[max(0, idx - 10):idx + 40].strip()
            snippet = _re.sub(r"\s+", " ", snippet)
            info["主要犯罪事实"] = snippet[:80]
            has_content = True
            break

    # 判决结果
    verdict_keywords = ["判", "判处", "获刑", "一审", "二审", "决定执行"]
    for kw in verdict_keywords:
        idx = text.find(kw)
        if idx >= 0:
            snippet = text[max(0, idx - 5):idx + 50].strip()
            snippet = _re.sub(r"\s+", " ", snippet)
            info["判决/处理结果"] = snippet[:80]
            has_content = True
            break

    # 时间
    date_patterns = [
        r"(\d{4})[年-](\d{1,2})[月-](\d{1,2})[日]?",
        r"(\d{4})年(\d{1,2})月",
    ]
    for p in date_patterns:
        m = _re.search(p, text)
        if m:
            info["通报/宣判时间"] = m.group(0)
            has_content = True
            break

    if not has_content:
        return None
    return info


@auto_tool(
    description=(
        "自动补全贪腐数据库。当用户查询某个人的案件，但该人不在本地数据库时调用此工具。"
        "自动联网搜索该人的贪腐新闻，提取关键信息（职务、金额、判决、日期），"
        "追加到本地 CSV 数据库，并返回补全结果。"
    ),
    patterns=[
        r"自动补全", r"添加到数据库", r"加到数据库",
        r"补全.*数据库", r"加入.*数据库",
    ],
    extract_params="person_name",
)
def auto_enrich_and_save(person_name: str) -> str:
    """
    当人名不在数据库时，自动联网搜索并追加到 CSV。
    """
    person_name = person_name.strip()
    if not person_name:
        return "**未提供人名，无法补全**"

    # Step 1：确认数据库中不存在此人
    records = _load_csv_data()
    existing = [r for r in records if person_name in r.get("当事人姓名", "")]
    if existing:
        table = _build_markdown_table(existing)
        return (
            f"### 「{person_name}」已在数据库中（共 {len(existing)} 条）\n\n"
            f"{table}"
        )

    # Step 2：联网搜索贪腐新闻
    search_term = f"{person_name} 贪腐 受贿 贪污 判决"
    search_result = None
    try:
        from agent.tools.web_search_tool import _search_ddg
        _, raw_results = _search_ddg(search_term, num_results=3)
    except Exception:
        raw_results = []

    if not raw_results:
        return (
            f"**未找到「{person_name}」的贪腐记录**\n\n"
            f"联网搜索「{search_term}」未返回任何结果，"
            f"可能此人无公开贪腐记录，或搜索词不匹配。"
        )

    # 取第一条搜索结果的摘要作为信息源
    top_result = raw_results[0]
    snippet = top_result.get("body", "") or top_result.get("title", "")
    title = top_result.get("title", "")

    # Step 3：提取贪腐信息
    combined_text = f"{title} {snippet}"
    info = _extract_case_info_from_text(person_name, combined_text)

    if info is None:
        return (
            f"**联网找到「{person_name}」相关信息，但未检测到明确的贪腐案件记录**\n\n"
            f"搜索标题：{title}\n"
            f"摘要：{snippet[:200]}\n\n"
            f"提示：如果确认此人有贪腐记录，可以提供更多线索（如职务、所在地区）重新查询。"
        )

    # Step 4：校验人名是否出现在原始文本中（避免搜索到无关内容就写入）
    # 人名必须出现在标题或摘要里，否则说明搜索结果与该人无关
    name_in_title = person_name in title
    name_in_snippet = person_name in snippet
    if not name_in_title and not name_in_snippet:
        return (
            f"**联网搜索未找到「{person_name}」的相关贪腐记录**\n\n"
            f"搜索标题：{title}\n"
            f"摘要：{snippet[:200]}\n\n"
            f"提示：搜索结果与「{person_name}」无关，可能是同名人员。请提供更多线索（如职务、所在地区）重新查询。"
        )

    # Step 6：追加到 CSV
    info["案件序号"] = str(_get_next_seq())
    success = _append_to_csv(info)

    if not success:
        return (
            f"**联网找到「{person_name}」的贪腐记录，但写入数据库失败**\n\n"
            f"提取的信息：\n"
            f"- 职务：{info['职务/身份']}\n"
            f"- 涉案金额：{info['涉案金额']}\n"
            f"- 判决：{info['判决/处理结果']}\n"
            f"- 时间：{info['通报/宣判时间']}\n\n"
            f"请检查 data/贪污记录.csv 文件权限。"
        )

    # Step 7：返回补全结果
    return (
        f"### 「{person_name}」已补全到数据库 ✅\n\n"
        f"**已自动追加到 `data/贪污记录.csv`（序号 {info['案件序号']}）**\n\n"
        f"| 字段 | 内容 |\n"
        f"| --- | --- |\n"
        f"| 职务/身份 | {info['职务/身份'] or '未提取到'} |\n"
        f"| 涉案金额 | {info['涉案金额'] or '未提取到'} |\n"
        f"| 主要犯罪事实 | {info['主要犯罪事实'] or '未提取到'} |\n"
        f"| 判决/处理结果 | {info['判决/处理结果'] or '未提取到'} |\n"
        f"| 通报/宣判时间 | {info['通报/宣判时间'] or '未提取到'} |\n\n"
        f"_数据来源：联网搜索自动提取，如有误请手动修正_"
    )


# 暴露给 router 调用（无需 @auto_tool 注册的辅助函数）
def check_person_in_db(person_name: str) -> tuple[bool, list]:
    """检查人名是否在数据库中，返回 (是否在库中, 匹配记录列表)"""
    if is_mysql_available():
        matched = corruption_select_by_name(person_name.strip())
        return (len(matched) > 0, matched)
    records = _load_csv_data()
    matched = [r for r in records if person_name.strip() in r.get("当事人姓名", "")]
    return (len(matched) > 0, matched)

