"""
联网搜索工具：通过 ddgs (DuckDuckGo) 搜索实现，无需 API key。

所有工具通过 @auto_tool 装饰器注册，无需手动添加到任何列表。
"""

import re
import csv
import os
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.db_handler import (
    is_mysql_available,
    corruption_check_exists,
    corruption_insert,
    corruption_get_next_seq,
    invalidate_query_cache,
)
from agent.tools.registry import auto_tool, TOOL_REGISTRY

try:
    from ddgs import DDGS as _DDGS
    _DDGS_AVAILABLE = True
except ImportError:
    _DDGS_AVAILABLE = False
    logger.warning("ddgs 未安装，联网搜索将不可用。请运行：pip install ddgs")

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logger.warning("requests 未安装，网页读取将不可用。请运行：pip install requests")


def _extract_url(query: str) -> str | None:
    """从查询中提取第一个 URL"""
    match = re.search(r"https?://[^\s<>\"']+", query)
    return match.group(0) if match else None


def _extract_search_keywords(query: str) -> str:
    """
    从用户的自然问句中提取精准的搜索关键词。

    处理策略：
    1. 保留人名/案件名（最核心的搜索实体）
    2. 保留"最新进展/最新动态"等搜索意图词
    3. 去掉句末疑问词（如何、怎样、吗、呢）和句首引导词（帮我搜、请问等）
    4. 去掉"案件/问题/情况"等泛化词
    """
    q = query.strip()
    # 去掉句首引导词
    q = re.sub(r"^(帮我搜?|帮我查?|请问|我想知道|想问一下|能否|可以|能不能|帮我查一下|去|上网)?", "", q, flags=re.IGNORECASE)
    # 去掉句末疑问词
    q = re.sub(r"(如何|怎样|了吗|么|呢|吗|呀|啊|\?|？|,|，|.|\…)$", "", q)
    # 去掉"案(件)最新进展"等冗余描述，保留主语
    # "谭瑞松案最新进展如何" → "谭瑞松 最新进展"
    q = re.sub(r"(案|案件|问题|情况|详情|具体|个人)?(最新进展|最新动态|最新消息|进展如何|动态如何|情况如何|详情|审判结果|判决结果)$",
               r" \2", q, flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _search_ddg(query: str, num_results: int = 5) -> tuple[str, list[dict]]:
    """
    用 DuckDuckGo 搜索，返回 (实际搜索词, 结构化结果列表)。

    搜索词经过关键词提取，去除问句干扰，提升结果相关性。
    """
    if not _DDGS_AVAILABLE:
        logger.error("[web_search] ddgs 库未安装")
        return query, []

    # 提取精准关键词，避免整句干扰搜索
    search_term = _extract_search_keywords(query)
    logger.info(f"[web_search] 原始问题：{query} → 搜索词：{search_term}")

    try:
        results = []
        with _DDGS() as ddg:
            for r in ddg.text(search_term, max_results=num_results):
                results.append({
                    "title": r.get("title", "").strip(),
                    "url":   r.get("href", "").strip(),
                    "body":  r.get("body", "").strip(),
                })
        return search_term, results
    except Exception as e:
        logger.error(f"[web_search] 搜索失败: {e}")
        return search_term, []


# ─────────────────────────────────────────────────────────────────────────────
# 搜索结果安全过滤
# ─────────────────────────────────────────────────────────────────────────────

# 黑名单域名（精确匹配或包含匹配）
_BLOCK_DOMAINS = frozenset({
    # 外网 / 视频站
    "youtube.com", "youtu.be", "twitter.com", "x.com",
    "facebook.com", "instagram.com", "tiktok.com",
    "weibo.com", "weibo.cn",
    # 色情 / 赌博关键词（域名包含即命中）
    "porn", "xvideos", "xhamster", "redtube",
    "sex", "erotic", "adult", "nsfw",
    "casino", "bet", "gambling",
    # 垃圾 / 聚合站
    "29pen.com", ".xyz", ".top",
})

# 黄赌毒内容关键词（正文或标题命中即拒绝）
_BAD_PATTERNS = [
    # 色情
    r"av女优", r"av.*stars", r"stars\d+", r"番号",
    r"东京热", r"一本道", r"裸聊", r"援交", r"约炮",
    r"成人.*视频", r"上门.*服务", r"性爱", r"黄色网站",
    # 赌博
    r"博彩平台", r"网上赌场", r"开户.*送.*金", r"威尼斯人",
    r"葡京娱乐", r"金沙.*赌场", r"投注.*优惠",
]


def _filter_search_results(results: list[dict]) -> list[dict]:
    """
    过滤搜索结果，去掉不可接受的结果（黑名单域名 + 黄赌毒内容）。
    若过滤后少于 2 条，改用宽松模式（仅过滤明确域名）。
    """
    def _is_ok(r: dict) -> bool:
        url = r.get("url", "").lower()
        # 域名黑名单
        for blocked in _BLOCK_DOMAINS:
            if blocked in url:
                return False
        # 内容关键词
        text = (r.get("title", "") + r.get("body", "")).lower()
        for pat in _BAD_PATTERNS:
            if re.search(pat, text):
                return False
        return True

    filtered = [r for r in results if _is_ok(r)]

    # 过滤过严时放宽：仅拒绝对应域名
    if len(filtered) < 2:
        strict = {"youtube", "youtu.be", "bilibili", "b23.tv", "twitter",
                   "x.com", "facebook", "instagram", "tiktok", "29pen.com",
                   "porn", "xvideos", "xhamster", "casino", "bet", "gambling"}
        filtered = [
            r for r in results
            if not any(b in r.get("url", "").lower() for b in strict)
        ]

    return filtered


def _fetch_snippet(url: str) -> str:
    """
    获取 URL 的简短摘要（用于搜索结果展示）。
    若 URL 不可访问或 body 过短，返回空字符串。
    """
    if not _REQUESTS_AVAILABLE:
        return ""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = _requests.get(url, headers=headers, timeout=8, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or "utf-8"
        if resp.status_code != 200:
            return ""
        text = _strip_html(resp.text)
        # 取前300字符作为摘要
        snippet = text[:300].strip()
        return snippet if snippet else ""
    except Exception:
        return ""


def _format_search_results(results: list[dict]) -> str:
    """将搜索结果格式化为 Markdown 列表。body 为空时尝试读取网页正文作为摘要。"""
    if not results:
        return "**未找到相关结果**"
    lines = []
    for i, r in enumerate(results, 1):
        body = r.get("body", "").strip()
        url = r.get("url", "").strip()
        # body 不足30字符时，尝试读取网页正文作为摘要
        if len(body) < 30 and url:
            body = _fetch_snippet(url) or body
        if not body:
            body = "暂无摘要"
        else:
            body = body[:200].strip()
        lines.append(
            f"{i}. **{r['title']}**\n"
            f"   - 链接：[点击查看]({url})\n"
            f"   - 摘要：{body}..."
        )
    return "\n\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 工具定义（通过 @auto_tool 注册一次，全局自动生效）
# ─────────────────────────────────────────────────────────────────────────────


# ── URL 总结工具（优先级最高，必须在 web_fetch 之前定义）─────────────────────

_SUMMARIZE_URL_SYSTEM_PROMPT = """你是一个专业的网页内容分析助手。用户会提供一个URL和对应的网页正文内容，你需要：

1. 简要说明这篇文章/页面是关于什么的（1-2句话）
2. 提取并总结页面的核心内容要点（按逻辑分段，用列表呈现）
3. 如果有数据、结论、步骤等实用信息，重点提取

要求：
- 用中文回答
- 语言简洁有条理
- 忠实于原文，不添加原文没有的信息
- 重点突出，不要复述无意义的导航栏/版权声明等干扰内容"""


def _strip_html(raw_html: str) -> str:
    """去除 HTML 标签，返回纯文本"""
    import re
    text = re.sub(r"<script[^>]*>.*?</script>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@auto_tool(
    description=(
        "读取网页内容并进行智能总结。当用户要求「总结」「概括」「提炼」某个URL对应的网页内容时调用此工具。"
        "先获取网页正文，再由 LLM 生成结构化总结。"
    ),
    patterns=[
        r"总结.*https?://",
        r"概括.*https?://",
        r"提炼.*https?://",
        r"解读.*https?://",
        r"summarize.*https?://",
        r"帮我总结.*http",
        r"帮我概括.*http",
    ],
    extract_params="web_fetch",
)
def web_fetch_and_summarize(query: str) -> str:
    """
    读取网页内容并生成结构化总结。
    先 fetch 网页，再由 LLM 总结。
    """
    url = _extract_url(query)
    if not url:
        return "未能在问题中找到有效的 URL，请提供完整网页链接。"

    if not _REQUESTS_AVAILABLE:
        return "requests 库未安装，无法读取网页。请运行：pip install requests"

    # Step 1: 获取网页正文
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = _requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or "utf-8"
        if resp.status_code != 200:
            return f"网页返回状态码 {resp.status_code}，读取失败。"
        raw_text = _strip_html(resp.text)
    except Exception as e:
        return f"读取网页时出错：{str(e)}"

    # 截取前 8000 字符（LLM 上下文限制）
    content = raw_text[:8000]
    truncated = "（内容过长已截断）" if len(raw_text) > 8000 else ""

    # Step 2: 用 LLM 总结
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from model.factory import chat_model

        messages = [
            SystemMessage(content=_SUMMARIZE_URL_SYSTEM_PROMPT),
            HumanMessage(content=f"URL：{url}\n\n网页正文内容：\n{content}\n\n{truncated}"),
        ]
        summary_resp = chat_model.invoke(messages)
        summary = summary_resp.content if hasattr(summary_resp, "content") else str(summary_resp)
        return (
            f"## 📄 页面内容总结\n\n"
            f"**来源**：{url}\n\n"
            f"{summary}\n\n"
            f"_以上内容由 AI 根据网页原文自动总结生成_"
        )
    except Exception as e:
        # LLM 失败，降级到返回原始文本
        return (
            f"【网页内容】URL：{url}\n\n"
            f"{content}{truncated}\n\n"
            f"（LLM 总结失败，以下为原始文本。共 {len(content)} 字符）"
        )


@auto_tool(
    description="读取指定网页内容。当用户提供具体 URL（http:// 或 https:// 开头）时调用此工具。自动提取网页正文并返回纯文本（最多3000字符）。",
    patterns=[
        r"https?://",
        r"读取.*网页", r"打开.*链接", r"访问.*网址",
        r"这个页面", r"该网页",
    ],
    extract_params="web_fetch",
)
def web_fetch(query: str) -> str:
    """读取指定网页内容工具。当用户提供了具体 URL 时调用此工具。"""
    url = _extract_url(query)
    if not url:
        return "未能在问题中找到有效的 URL，请提供完整网页链接（如 https://...）"

    if not _REQUESTS_AVAILABLE:
        return "requests 库未安装，无法读取网页。请运行：pip install requests"

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
        resp = _requests.get(url, headers=headers, timeout=15, allow_redirects=True)
        resp.encoding = resp.apparent_encoding or "utf-8"

        if resp.status_code != 200:
            return f"网页返回状态码 {resp.status_code}，读取失败。"

        content = resp.text
        text = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        preview = text[:3000]
        truncated = "（...内容过长已截断）" if len(text) > 3000 else ""

        return (
            f"【网页内容】URL：{url}\n\n"
            f"{preview}{truncated}\n\n"
            f"（以上内容来自网页读取，共 {len(text)} 字符）"
        )
    except Exception as e:
        return f"读取网页时出错：{str(e)}"


@auto_tool(
    description="联网搜索工具。当用户问题涉及最新进展、实时新闻、法律条文、司法解释、不在本地数据库中的案件信息时调用此工具。使用 DuckDuckGo 搜索，自动读取网页正文作为摘要（解决部分链接无摘要的问题）。",
    patterns=[
        r"最新消息", r"最新进展", r"最新通报", r"最新反腐", r"最新动态",
        r"现在.*怎样", r"目前.*情况", r"实时",
        r"新闻.*搜", r"报道.*搜", r"网上.*搜", r"网上查",
        r"帮我搜.*新闻", r"帮我搜.*进展", r"帮我查.*新闻",
        r"最近.*新闻", r"最近.*动态", r"今天.*新闻",
        r"最近.*通报", r"近期.*案件",
        r"搜索.*新闻", r"查找.*进展",
        r"上网搜", r"搜一下", r"搜搜", r"搜它",
        r"上网查", r"网上搜", r"去搜",
        r"搜.*最新", r"查.*最新",
        r"帮我搜索(最新|最近|反腐)?",
    ],
    extract_params="web_search",
)
def web_search(query: str) -> str:
    """联网搜索工具。查询实时贪腐新闻、法律条文等不在本地数据中的信息。"""
    search_term, results = _search_ddg(query, num_results=8)
    original_count = len(results)
    results = _filter_search_results(results)
    filtered_count = original_count - len(results)
    formatted = _format_search_results(results)
    extra_note = f"（已过滤 {filtered_count} 条不相关内容）" if filtered_count else ""
    return (
        f"【联网搜索结果】原始问题：{query}\n"
        f"实际搜索词：**{search_term}**{extra_note}\n"
        f"共找到 {len(results)} 条结果：\n\n"
        f"{formatted}\n\n"
        f"（以上结果来自 DuckDuckGo 搜索，实时获取）"
    )


@auto_tool(
    description="贪污新闻推荐工具。根据本地案件数据，推荐值得关注的重要案件，包括涉案金额最高Top5、最新通报案件、职务层级分布。无需输入参数。",
    patterns=[
        r"推荐.*案件", r"推荐.*新闻", r"给我推荐.*案件",
        r"值得关注",
        r"典型.*案件", r"重点.*案件", r"热门.*案件",
    ],
    extract_params="none",
)
def recommend_news() -> str:
    """贪污新闻推荐工具。根据本地案件数据，推荐值得关注的重要案件。"""
    from agent.tools.data_query_tool import _load_data, _parse_amount, _build_markdown_table

    records = _load_data()
    if not records:
        return "**数据文件为空，无法推荐**"

    for r in records:
        r["_金额数值"] = _parse_amount(r.get("涉案金额", ""))

    high_value = sorted(
        [r for r in records if r["_金额数值"] > 0],
        key=lambda x: x["_金额数值"],
        reverse=True,
    )
    recent = sorted(
        [r for r in records if r.get("通报/宣判时间", "")],
        key=lambda x: x.get("通报/宣判时间", ""),
        reverse=True,
    )

    lines = ["## 值得关注的重要贪腐案件推荐\n"]

    lines.append("### 一、涉案金额最高（重大典型）\n")
    lines.append(_build_markdown_table(high_value[:5]))

    lines.append("\n### 二、最新通报案件\n")
    lines.append(_build_markdown_table(recent[:5]))

    lines.append("\n### 三、职务层级分布\n")
    provincial = [r for r in records if "省" in r.get("职务/身份", "")]
    bureau = [r for r in records if "厅" in r.get("职务/身份", "") or "局" in r.get("职务/身份", "")]
    county = [r for r in records if "县" in r.get("职务/身份", "") or "市" in r.get("职务/身份", "")]
    lines.append(
        f"| 职务层级 | 人数 | 代表人物 |\n"
        f"| --- | --- | --- |\n"
        f"| 省部级 | {len(provincial)} | {', '.join(r.get('当事人姓名','') for r in provincial[:3])} |\n"
        f"| 厅局级 | {len(bureau)} | {', '.join(r.get('当事人姓名','') for r in bureau[:3])} |\n"
        f"| 县处级 | {len(county)} | {', '.join(r.get('当事人姓名','') for r in county[:3])} |"
    )

    lines.append(
        "\n\n> **推荐说明**：以上案件均为数据库中的公开记录，"
        "标注为典型案例或近期通报案件。如需了解特定案件详情，请直接提问。"
    )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 自动联网补全 + 按需写入 CSV
# 流程：Step1 联网搜索 → Step2 LLM 判断是否贪腐案件（置信度） → Step3 按置信度决定是否写入
# ─────────────────────────────────────────────────────────────────────────────

# Step2：判断搜索结果是否与贪腐相关的 prompt
_RELEVANCE_CHECK_PROMPT = """判断以下搜索结果是否与贪腐/受贿/职务犯罪案件相关。

搜索关键词：{person} 贪腐 案件 判决 受贿

搜索结果：
{results_text}

请严格按以下 JSON 格式输出（只输出 JSON，不要任何解释）：
{{
  "is_corruption_case": true或false,
  "confidence": 0到100的整数（置信度，越高越确定是贪腐案件）,
  "reason": "简要说明判断理由（1句话）",
  "confirmed_name": "搜索结果中确认的当事人姓名（无法确认则为空字符串）"
}}

判断标准：
- is_corruption_case=true：搜索结果指向明确的贪腐/受贿/职务犯罪案件（如"XXX被开除党籍""XXX受贿案宣判"）
- is_corruption_case=false：搜索结果主要是同名普通人、新闻人物但不涉及贪腐、或者信息不相关
- confidence ≥ 80：结果明确指向贪腐案件，直接写入数据库
- 50 ≤ confidence < 80：结果部分相关，需要展示给用户确认
- confidence < 50：结果不相关，不写入，只返回搜索摘要
"""


# Step3：结构化字段提取 prompt
_EXTRACT_PROMPT = """你是一个贪腐案件数据提取专家。根据搜索结果，提取当事人信息并按指定格式返回 JSON。

严格按以下字段提取（只填有把握的内容，留空表示无法提取）：
- person_name：当事人姓名（字符串）
- job_title：职务/身份（字符串）
- amount：涉案金额（单位统一为"X万元"或"X亿余元"格式，如无法确定填""）
- crime_facts：主要犯罪事实（字符串，简短概括）
- verdict：判决/处理结果（字符串，简短）
- date：通报/宣判时间（YYYY-MM-DD 格式，填""表示无法确定）
- note：备注（如"联网补全"，填""表示无）

输出格式（只输出 JSON，不要任何解释）：
{{"person_name":"","job_title":"","amount":"","crime_facts":"","verdict":"","date":"","note":""}}
"""


def _build_search_context(results: list[dict]) -> str:
    """将搜索结果拼接为 LLM 可读的上下文"""
    return "\n".join(
        f"【标题】{r.get('title', '未知')}  【链接】{r.get('url', '')}  【摘要】{r.get('body', '')}"
        for r in results[:5]
    )


# ── Step1+2 合并：联网搜索 + LLM 判断相关性 ────────────────────────────────

def _search_and_validate(person: str) -> dict:
    """
    联网搜索 + LLM 判断相关性。
    返回 {"status": "ok"|"no_results"|"not_corruption", "confidence": int,
          "results": [...], "confirmed_name": str, "reason": str}
    """
    search_q = f"{person} 贪腐 案件 判决 受贿"
    logger.info(f"[auto_enrich] 联网搜索：{person}")

    search_term, raw_results = _search_ddg(search_q, num_results=5)
    results = _filter_search_results(raw_results)

    if not results:
        return {"status": "no_results", "confidence": 0,
                "results": [], "confirmed_name": "", "reason": "搜索无结果"}

    results_text = _build_search_context(results)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from model.factory import chat_model
        import json as _json

        resp = chat_model.invoke([
            SystemMessage(content=_RELEVANCE_CHECK_PROMPT.format(
                person=person, results_text=results_text,
            )),
            HumanMessage(content=f"请判断搜索结果是否与贪腐案件相关："),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        raw = raw.strip()
        raw = re.sub(r"```(?:json)?[^\n]*\n?", "", raw).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("LLM 未返回有效 JSON")
        judgment = _json.loads(m.group())
    except Exception as e:
        logger.error(f"[auto_enrich] 相关性判断失败：{e}，使用宽松默认")
        judgment = {
            "is_corruption_case": True,
            "confidence": 50,
            "reason": "LLM判断异常，默认中等置信度",
            "confirmed_name": results[0].get("title", "").split("——")[0].split("|")[0].strip(),
        }

    is_corruption = judgment.get("is_corruption_case", False)
    confidence = int(judgment.get("confidence", 0))
    confirmed_name = judgment.get("confirmed_name", person)
    reason = judgment.get("reason", "")

    if not is_corruption:
        return {
            "status": "not_corruption",
            "confidence": confidence,
            "results": results,
            "confirmed_name": confirmed_name,
            "reason": reason,
        }

    return {
        "status": "ok",
        "confidence": confidence,
        "results": results,
        "confirmed_name": confirmed_name,
        "reason": reason,
    }


# ── Step3：提取结构化数据（不写入，只返回行数据） ────────────────────────────

def _extract_structured(person: str, results: list[dict]) -> dict:
    """
    对搜索结果进行 LLM 结构化提取。
    返回 {"ok": bool, "row": dict, "error": str}
    """
    results_text = _build_search_context(results)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from model.factory import chat_model
        import json as _json

        resp = chat_model.invoke([
            SystemMessage(content=_EXTRACT_PROMPT),
            HumanMessage(content=f"搜索结果：\n{results_text}\n\n请提取 {person} 的贪腐案件信息："),
        ])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        raw = raw.strip()
        raw = re.sub(r"```(?:json)?[^\n]*\n?", "", raw).strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            raise ValueError("LLM 未返回有效 JSON")
        data = _json.loads(m.group())
    except Exception as e:
        logger.error(f"[auto_enrich] 结构化提取失败：{e}")
        return {"ok": False, "row": {}, "error": str(e)}

    return {"ok": True, "row": data, "error": ""}


# ── CSV 写入（单独函数，可独立调用） ─────────────────────────────────────────

_CSV_FIELDS = [
    "案件序号", "当事人姓名", "职务/身份", "涉案金额",
    "主要犯罪事实", "判决/处理结果", "通报/宣判时间", "备注",
]


def _check_record_exists(name: str, date: str, verdict: str) -> tuple[bool, list]:
    """
    检查数据库中是否已存在完全相同的记录（同名 + 相同日期 + 相似判决）。
    优先 MySQL；不可达时回退到 CSV 原逻辑。
    """
    # MySQL 优先路径
    if is_mysql_available():
        return corruption_check_exists(name, date, verdict)

    # CSV 降级（原逻辑保持不变）
    csv_path = get_abs_path("data/贪污记录.csv")
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return False, []
    except Exception:
        return False, []

    name_lower = name.lower()
    date_norm = date.strip()
    verdict_norm = verdict.strip()[:20]

    matches = []
    for r in rows:
        db_name = r.get("当事人姓名", "").strip()
        if name_lower not in db_name.lower():
            continue
        db_date = r.get("通报/宣判时间", "").strip()
        db_verdict = r.get("判决/处理结果", "").strip()[:20]
        if db_date == date_norm and db_verdict == verdict_norm:
            matches.append(r)
    return bool(matches), matches


def _write_csv_row(data: dict, confirmed_name: str) -> dict:
    """
    将一条结构化记录写入数据库。
    优先 MySQL；失败时降级 CSV。
    返回 {"ok": bool, "seq": int, "error": str}
    """
    # 获取下一条序号（优先 MySQL）
    next_seq = 1
    if is_mysql_available():
        next_seq = corruption_get_next_seq()
        if next_seq <= 0:
            next_seq = 1
    else:
        csv_path = get_abs_path("data/贪污记录.csv")
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                nums = [
                    int(r.get("案件序号", 0))
                    for r in reader
                    if r.get("案件序号", "").isdigit()
                ]
                next_seq = max(nums) + 1 if nums else 1
        except Exception:
            pass

    # 非空校验
    person_val = data.get("person_name", confirmed_name).strip()
    if not person_val:
        return {"ok": False, "seq": 0, "error": "当事人姓名为空，禁止写入"}
    values = [str(data.get(k, "").strip()) for k in ["person_name", "job_title", "amount", "crime_facts", "verdict", "date", "note"] if data.get(k)]
    if not any(v for v in values):
        return {"ok": False, "seq": 0, "error": "所有字段均为空，禁止写入"}

    new_row = {
        "案件序号": str(next_seq),
        "当事人姓名": data.get("person_name", confirmed_name),
        "职务/身份": data.get("job_title", ""),
        "涉案金额": data.get("amount", ""),
        "主要犯罪事实": data.get("crime_facts", ""),
        "判决/处理结果": data.get("verdict", ""),
        "通报/宣判时间": data.get("date", ""),
        "备注": data.get("note", "联网补全"),
    }

    # MySQL 优先写入
    if is_mysql_available():
        ok, new_id, err = corruption_insert(new_row)
        if ok:
            invalidate_query_cache()
            return {"ok": True, "seq": next_seq, "error": ""}
        logger.warning(f"[web_search_tool] MySQL 写入失败，降级到 CSV: {err}")

    # CSV 降级写入
    csv_path = get_abs_path("data/贪污记录.csv")
    try:
        file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
        with open(csv_path, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(new_row)
        invalidate_query_cache()
        return {"ok": True, "seq": next_seq, "error": ""}
    except Exception as e:
        logger.error(f"[web_search_tool] CSV 写入失败: {e}")
        return {"ok": False, "seq": next_seq, "error": str(e)}


# ── 主入口工具 ──────────────────────────────────────────────────────────────

@auto_tool(
    description=(
        "联网搜索人员信息并按需追加到本地数据库。"
        "Step1 联网搜索；Step2 LLM 判断是否为贪腐案件（置信度）；"
        "Step3 置信度≥80自动写入，50-79展示确认，<50不写入。"
        "只有确认为贪腐人员才会被加入 data/贪污记录.csv。"
    ),
    patterns=[
        r"联网补全",
        r"联网.*追加",
        r"自动补全",
        r"上网.*更新.*数据库",
    ],
    extract_params="person_name",
)
def auto_enrich_and_save(person_name: str) -> str:
    """
    联网补全人员信息：搜索 → LLM 判断相关性 → 按置信度决定是否写入 CSV。

    置信度规则：
      ≥ 80：高置信贪腐案件，自动写入数据库
      50-79：中置信，展示搜索摘要请用户确认后再写入
      < 50：非贪腐相关，不写入，只返回搜索摘要
    """
    if not person_name or not person_name.strip():
        return "未提供人名，无法联网补全。"

    person = person_name.strip()

    # ── Step 1+2：联网搜索 + LLM 判断相关性 ─────────────────────────
    check = _search_and_validate(person)

    if check["status"] == "no_results":
        return f"联网搜索「{person}」未找到相关信息。"

    if check["status"] == "not_corruption":
        # 搜索结果不是贪腐案件，只展示搜索摘要，不写入
        formatted = _format_search_results(check["results"])
        return (
            f"### ⚠️ 「{person}」未识别为贪腐案件（置信度：{check['confidence']}%）\n\n"
            f"判断理由：{check['reason']}\n\n"
            f"联网搜索结果如下（未写入数据库）：\n\n{formatted}"
        )

    # ── Step 3a：置信度 ≥ 80 → 直接提取并写入 ───────────────────────
    if check["confidence"] >= 80:
        extract = _extract_structured(person, check["results"])
        if not extract["ok"]:
            return (
                f"联网搜索成功（置信度 {check['confidence']}%），"
                f"但信息提取失败：{extract['error']}\n\n"
                f"搜索摘要：{_format_search_results(check['results'])}"
            )
        # Step 3a.1：检查是否已有相同记录
        new_row = extract["row"]
        dup, dup_records = _check_record_exists(
            check["confirmed_name"],
            new_row.get("date", ""),
            new_row.get("verdict", ""),
        )
        if dup:
            dup_dates = " / ".join(r.get("通报/宣判时间", "") for r in dup_records)
            return (
                f"### ⏭️ 「{check['confirmed_name']}」已有相同记录，跳过重复写入\n\n"
                f"| 字段 | 内容 |\n"
                f"| --- | --- |\n"
                f"| 当事人姓名 | {check['confirmed_name']} |\n"
                f"| 职务/身份 | {new_row.get('job_title', '—')} |\n"
                f"| 涉案金额 | {new_row.get('amount', '—')} |\n"
                f"| 判决/处理结果 | {new_row.get('verdict', '—')} |\n"
                f"| 通报/宣判时间 | {new_row.get('date', '—')} |\n\n"
                f"数据库中已有相同日期和判决的记录：{dup_dates}，"
                f"未重复写入。如需追加该人的其他案件，请提供不同时间或不同判决的线索。\n\n"
                f"_置信度：{check['confidence']}% | "
                f"来源：{check['results'][0].get('title', '未知')} | "
                f"共 {len(check['results'])} 条结果_"
            )

        # Step 3a.2：无重复，写入新记录
        write = _write_csv_row(new_row, check["confirmed_name"])
        if not write["ok"]:
            return (
                f"联网搜索 + 信息提取成功，但数据库写入失败（{write['error']}）。\n\n"
                f"提取到的信息：{new_row}\n\n"
                f"搜索来源：{check['results'][0].get('title', '未知')} | "
                f"共 {len(check['results'])} 条结果"
            )
        row = new_row
        return (
            f"### ✅ 「{check['confirmed_name']}」已联网补全并写入数据库\n\n"
            f"| 字段 | 内容 |\n"
            f"| --- | --- |\n"
            f"| 案件序号 | {write['seq']} |\n"
            f"| 当事人姓名 | {row.get('person_name', person)} |\n"
            f"| 职务/身份 | {row.get('job_title', '—')} |\n"
            f"| 涉案金额 | {row.get('amount', '—')} |\n"
            f"| 判决/处理结果 | {row.get('verdict', '—')} |\n"
            f"| 通报/宣判时间 | {row.get('date', '—')} |\n"
            f"| 备注 | {row.get('note', '联网补全')} |\n\n"
            f"_置信度：{check['confidence']}% | "
            f"来源：{check['results'][0].get('title', '未知')} | "
            f"共 {len(check['results'])} 条结果_"
        )

    # ── Step 3b：置信度 50-79 → 展示摘要请用户确认 ─────────────────
    # 注意：此工具是自动调用的，确认动作由上层 Agent 判断
    # 这里将搜索结果中关键内容提取展示，提示可再次调用确认写入
    extract = _extract_structured(person, check["results"])
    row_preview = extract["row"] if extract["ok"] else {}
    formatted = _format_search_results(check["results"])

    confirm_text = ""
    if extract["ok"] and row_preview:
        confirm_text = (
            f"\n\n**待写入数据预览**：\n"
            f"| 字段 | 内容 |\n"
            f"| --- | --- |\n"
            f"| 当事人姓名 | {row_preview.get('person_name', person)} |\n"
            f"| 职务/身份 | {row_preview.get('job_title', '—')} |\n"
            f"| 涉案金额 | {row_preview.get('amount', '—')} |\n"
            f"| 判决/处理结果 | {row_preview.get('verdict', '—')} |\n"
            f"| 通报/宣判时间 | {row_preview.get('date', '—')} |"
        )

    return (
        f"### ⚠️ 「{person}」为中等置信度贪腐案件（置信度：{check['confidence']}%）\n\n"
        f"判断理由：{check['reason']}。"
        f"请确认是否将此记录追加到数据库（可回复「确认写入」）。\n\n"
        f"联网搜索结果：\n{formatted}"
        f"{confirm_text}"
    )

