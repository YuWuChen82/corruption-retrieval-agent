# -*- coding: utf-8 -*-
"""
LLM Router - 多工具规划 + LLM 综合方案。

流程：
  1. llm_plan_tools()   — LLM 分析问题，规划需要调用哪些工具（可多个）
  2. execute_tools()     — 并发执行所有工具调用
  3. llm_synthesize()     — LLM 综合所有结果，生成最终回答
  4. 流式输出思考过程（每个工具的执行状态 + 最终答案）
"""

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator
from langchain_core.messages import HumanMessage, SystemMessage
from model.factory import chat_model


# ─────────────────────────────────────────────────────────────────────────────
# 工具描述构建
# ─────────────────────────────────────────────────────────────────────────────

def build_tool_descriptions(registry):
    """从 TOOL_REGISTRY 动态构建工具描述，供 LLM 理解每个工具的用途"""
    tools = []
    for meta in registry._tools:
        params = []
        try:
            import inspect
            sig = inspect.signature(meta.func)
            for pname, param in sig.parameters.items():
                if pname in ("query", "self"):
                    continue
                params.append({
                    "name": pname,
                    "type": "string",
                    "required": True,
                    "description": "param " + pname,
                })
        except Exception:
            pass
        tools.append({
            "name": meta.name,
            "description": meta.description,
            "params": params,
        })
    return tools


# ─────────────────────────────────────────────────────────────────────────────
# 工具名别名映射（router.py 维护，这里引用）
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_ALIAS = {
    "rag_summarize_tool": "rag_summarize",
    "query_corruption_by_name_tool": "query_corruption_by_name",
}


# ─────────────────────────────────────────────────────────────────────────────
# 工具执行函数（由 router.py 注入）
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_FUNC_MAP = {}   # name -> func


def register_tool(name, func):
    _TOOL_FUNC_MAP[name] = func


def _resolve_tool_name(name: str) -> str:
    """别名解析"""
    return _TOOL_ALIAS.get(name, name)


def invoke_single_tool(tool_name: str, params: dict) -> dict:
    """执行单个工具，返回 {"name": str, "result": str, "error": str|None}"""
    resolved = _resolve_tool_name(tool_name)
    func = _TOOL_FUNC_MAP.get(resolved)
    if not func:
        return {
            "name": tool_name,
            "result": "",
            "error": "tool not found: " + resolved,
        }
    try:
        result = func.invoke(params)
        return {"name": tool_name, "result": str(result), "error": None}
    except Exception as e:
        return {"name": tool_name, "result": "", "error": str(e)}


def execute_tools(tool_calls: list) -> list:
    """并发执行多个工具调用，返回结果列表"""
    results = []
    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
        futures = {
            pool.submit(invoke_single_tool, call["name"], call["params"]): call
            for call in tool_calls
        }
        for future in as_completed(futures):
            results.append(future.result())
    return results


# ─────────────────────────────────────────────────────────────────────────────
# LLM 多工具规划
# ─────────────────────────────────────────────────────────────────────────────

def _build_history_context(chat_history: list | None, max_turns: int = 4) -> str:
    """从对话历史中提取最近 max_turns 轮，格式化为简短的上下文文本"""
    if not chat_history:
        return ""
    pairs = []
    for i in range(len(chat_history) - 1, -1, -1):
        msg = chat_history[i]
        if msg.get("role") == "user":
            if i + 1 < len(chat_history) and chat_history[i + 1].get("role") == "assistant":
                user_q = str(msg.get("content", ""))[:120]
                assistant_a = str(chat_history[i + 1].get("content", ""))[:200]
                pairs.append((user_q, assistant_a))
        if len(pairs) >= max_turns:
            break
    if not pairs:
        return ""
    lines = ["【对话历史】"]
    for user_q, assistant_a in reversed(pairs):
        lines.append(f"用户：{user_q}")
        lines.append(f"助手：{assistant_a}")
    return "\n".join(lines)


_PLANNER_PROMPT_TEMPLATE = (
    "You are a Chinese anti-corruption assistant. Analyze the user query and plan which tools to call.\n\n"
    "CRITICAL RULES — FOLLOW CAREFULLY:\n"
    "1. EXTRACT ACTUAL PERSON NAMES: When user asks about a specific person (e.g. '谭瑞松的案件' or '谭瑞松是谁'), "
    "params.person_name MUST be the person's actual name like '谭瑞松'. "
    "DO NOT use '详细介绍', '详细情况', '案件信息' etc. as person_name values.\n"
    "2. CONTEXT REFERENCES: If the query contains pronouns like '他们' '他' '她' '详细介绍' '这个人' '上文' etc., "
    "you MUST look at the conversation history to find who is being referred to, "
    "then call query_corruption_by_name for EACH person.\n"
    "3. AUTO ENRICH (IMPORTANT): When user asks for '详细介绍' about a person that is NOT in the current query text "
    "(e.g. '详细介绍他们俩' after previously mentioning names), "
    "you MUST call auto_enrich_and_save with that person's name. "
    "This tool searches the web AND saves the result to the CSV database automatically.\n"
    "4. PERSON + LATEST/NEWS: When asking about a person's latest status/news, "
    "call BOTH query_corruption_by_name AND web_search.\n"
    "5. RANKINGS: Use rank_corruption_records for 'highest amount', 'top N', 'most serious'.\n"
    "6. LIST/ALL CASES: Use get_all_corruption_records for 'what cases exist', 'all cases', 'year N cases'.\n"
    "7. CONCEPT/DEFINITION: Use rag_summarize for definitions, policy understanding.\n"
    "8. You can call 1-4 tools at once. More tools = more comprehensive answer.\n"
    "9. NEVER call query_corruption_by_name with person_name='详细介绍' or similar — "
    "that is NOT a person name. If the query has no real person name, use rank_corruption_records instead.\n\n"
    "{history_context}"
    "Tools:\n{tool_text}\n\n"
    "Output JSON array of tool calls (each with name, params, reason):\n"
    '[{{"name": "tool_name", "params": {{"param": "value"}}, "reason": "why call this"}}]'
    "\n\nQuery: {query}"
)


def llm_plan_tools(user_query: str, tool_descs: list, chat_history: list | None = None) -> list:
    """
    LLM 规划：分析问题，决定调用哪些工具。
    返回 [{"Name": str, "params": dict, "reason": str}, ...]
    """
    tool_lines = []
    for t in tool_descs:
        ps = ", ".join(p["name"] + ": " + p["description"] for p in t["params"]) if t["params"] else "no params"
        tool_lines.append("- " + t["name"] + " | " + t["description"] + " | params: " + ps)
    tool_text = "\n".join(tool_lines)

    history_ctx = _build_history_context(chat_history)
    history_block = history_ctx + "\n\n" if history_ctx else ""
    prompt = _PLANNER_PROMPT_TEMPLATE.format(
        tool_text=tool_text,
        query=user_query,
        history_context=history_block,
    )

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Plan tools for: " + user_query),
    ]
    try:
        response = chat_model.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return [
            {"name": "rag_summarize", "params": {"query": user_query}, "reason": "LLM fail: " + str(e)[:20]}
        ]

    # 解析 JSON 数组
    try:
        raw = raw.strip()
        # 去 markdown
        raw = re.sub(r"```(?:json)?[^\n]*\n?", "", raw).strip()
        first = raw.find("[")
        last = raw.rfind("]")
        if first != -1 and last > first:
            raw = raw[first:last + 1]
        plan = json.loads(raw)
        # 归一化参数类型
        for call in plan:
            params = call.get("params", {})
            if "top_n" in params:
                try:
                    params["top_n"] = int(params["top_n"])
                except Exception:
                    params["top_n"] = 10
            if "order_by" in params:
                ob = str(params["order_by"])
                order_map = {
                    "金额从高到低": "amount_desc", "金额从低到高": "amount_asc",
                    "时间从新到旧": "date_desc", "时间从旧到新": "date_asc",
                }
                params["order_by"] = order_map.get(ob, ob)
            if "year" in params:
                yr = str(params["year"])
                m = re.search(r"\d{4}", yr)
                params["year"] = m.group(0) if m else ""
        return plan
    except Exception:
        return [
            {"name": "rag_summarize", "params": {"query": user_query}, "reason": "JSON parse fail"}
        ]


# ─────────────────────────────────────────────────────────────────────────────
# LLM 综合结果
# ─────────────────────────────────────────────────────────────────────────────

def llm_synthesize(user_query: str, tool_results: list, stream: bool = False, chat_history: list | None = None):
    """
    LLM 综合多个工具的结果，生成最终回答。
    stream=True  → 返回 generator，逐块 yield
    stream=False → 返回完整字符串
    chat_history 用于提供对话记忆上下文。
    """
    if not tool_results:
        return "" if not stream else iter([""])

    # 构建工具结果文本
    results_text = []
    for r in tool_results:
        name = r.get("name", "?")
        if r.get("error"):
            content = "[错误] " + r["error"]
        else:
            content = r.get("result", "")
            # 截断过长结果（LLM 上下文限制）
            if len(content) > 3000:
                content = content[:3000] + "\n...(内容过长已截断)"
        results_text.append(
            f"【{name}】\n{content}\n"
        )

    # 把对话历史加入上下文
    history_ctx = _build_history_context(chat_history)
    history_block = f"\n\n{history_ctx}\n" if history_ctx else ""

    synthesis_prompt = (
        "你是贪腐记录检索助手。请根据用户问题，综合多个工具的返回结果，生成完整准确的回答。{history}\n\n"
        "用户问题：{query}\n\n"
        "工具返回结果：\n{results}\n\n"
        "回答要求（严格遵守）：\n"
        "1. 综合利用所有工具的返回结果，不遗漏任何有价值的信息\n"
        "2. 呈现方式：列表优于段落，表格优于列表（比较多条记录时用 Markdown 表格）\n"
        "3. 关键数字（金额、人数）加粗\n"
        "4. 若部分工具返回空或报错，明确说明，只用有效结果作答\n"
        "5. 用中文回答\n\n"
        "禁止行为：\n"
        "- 编造工具返回结果中不存在的信息\n"
        "- 添加参考资料未提及的细节\n"
        "- 使用工具返回结果中未出现的来源引用\n\n"
        "综合回答："
    ).format(
        history=history_block,
        query=user_query,
        results="\n".join(results_text),
    )

    system_prompt = (
        "你是一个专业、严谨的贪腐记录领域助手。"
        "根据给定信息和对话上下文回答，不编造内容。"
        "{history_context}"
    ).format(history_context=history_ctx)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=synthesis_prompt),
    ]

    try:
        if stream:
            for event in chat_model.stream(messages):
                if hasattr(event, "content") and event.content:
                    yield event.content
        else:
            response = chat_model.invoke(messages)
            result = response.content if hasattr(response, "content") else str(response)
            yield result   # 让 list() 能收集到
    except Exception as e:
        # 综合失败，降级返回第一个有效结果
        for r in tool_results:
            if r.get("result") and not r.get("error"):
                yield r["result"]
                return
        yield "综合结果失败：" + str(e)


# ─────────────────────────────────────────────────────────────────────────────
# 单工具路由（保留兼容，当多工具规划失败时降级）
# ─────────────────────────────────────────────────────────────────────────────

_SINGLE_TOOL_PROMPT = (
    "You are a Chinese anti-corruption assistant. Select the best tool and extract params.\n\n"
    "CRITICAL RULES:\n"
    "1. CONTEXT REFERENCES: If the query contains pronouns like '他们' '他' '她' '详细介绍' '这个人' '上文' etc., "
    "look at the conversation history to find who is being referred to, "
    "then use query_corruption_by_name with the actual person name(s).\n"
    "2. PERSON NAME: When a Chinese person's name appears, ALWAYS use query_corruption_by_name.\n"
    "'XXX案最新进展' or 'XXX最新' means 'latest news about person XXX' -> query_corruption_by_name.\n"
    "3. Do NOT use web_search for person name queries even if '最新' is present.\n\n"
    "{history_context}"
    "Tools:\n{tool_text}\n\n"
    'Output JSON: {"tool": "tool_name", "reason": "reason", "params": {"param": "value"}}'
)


def llm_route_single(user_query: str, tool_descs: list, chat_history: list | None = None) -> dict:
    """单工具路由（保留兼容）"""
    tool_lines = []
    for t in tool_descs:
        ps = ", ".join(p["name"] + ": " + p["description"] for p in t["params"]) if t["params"] else "no params"
        tool_lines.append("- " + t["name"] + " | " + t["description"] + " | params: " + ps)
    tool_text = "\n".join(tool_lines)

    history_ctx = _build_history_context(chat_history)
    history_block = history_ctx + "\n\n" if history_ctx else ""
    prompt = _SINGLE_TOOL_PROMPT.format(tool_text=tool_text, history_context=history_block)

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Query: " + user_query),
    ]
    try:
        response = chat_model.invoke(messages)
        raw = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return {"tool": "rag_summarize", "reason": "LLM fail:" + str(e)[:20], "params": {"query": user_query}}

    try:
        raw = raw.strip()
        raw = re.sub(r"```(?:json)?[^\n]*\n?", "", raw).strip()
        first = raw.find("{")
        last = raw.rfind("}")
        if first != -1 and last > first:
            raw = raw[first:last + 1]
        result = json.loads(raw)
        tool = result.get("tool", "").strip()
        params = result.get("params", {})
        if "top_n" in params:
            try:
                params["top_n"] = int(params["top_n"])
            except Exception:
                params["top_n"] = 10
        if "order_by" in params:
            ob = str(params["order_by"])
            order_map = {
                "金额从高到低": "amount_desc", "金额从低到高": "amount_asc",
                "时间从新到旧": "date_desc", "时间从旧到新": "date_asc",
            }
            params["order_by"] = order_map.get(ob, ob)
        if "year" in params:
            yr = str(params["year"])
            m = re.search(r"\d{4}", yr)
            params["year"] = m.group(0) if m else ""
        return {"tool": tool, "reason": result.get("reason", ""), "params": params}
    except Exception:
        return {"tool": "rag_summarize", "reason": "JSON fail", "params": {"query": user_query}}
