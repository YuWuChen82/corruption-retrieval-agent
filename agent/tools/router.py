"""
智能路由 + 思考过程追踪器。

核心流程（多工具方案）：
  1. LLM 规划 — 分析问题，决定调用哪些工具（可多个）
  2. 并发执行 — ThreadPoolExecutor 并发调用所有工具
  3. LLM 综合 — 将所有工具结果拼接，由 LLM 生成最终回答
  4. 流式输出 — 思考过程实时展示，最终答案逐字输出

降级策略：
  - 多工具规划失败 → LLM 单工具路由
  - 单工具路由失败 → 正则规则匹配
  - 正则也未命中 → RAG 语义总结

提供带步骤记录的结果结构 RouteResult，供 UI 分步渲染。
"""

import os, sys as _sys
import re as _re

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)
del _project_root, _sys

from dataclasses import dataclass, field
from typing import Generator


# ─────────────────────────────────────────────────────────────────────────────
# 触发 @auto_tool 装饰器注册
# ─────────────────────────────────────────────────────────────────────────────

from agent.tools.registry import TOOL_REGISTRY
from agent.tools import data_query_tool   # noqa: F401
from agent.tools import web_search_tool   # noqa: F401

from agent.tools.llm_router import (
    build_tool_descriptions,
    llm_plan_tools,
    llm_synthesize,
    llm_route_single,
    execute_tools,
    invoke_single_tool,
    _TOOL_FUNC_MAP as _LLM_TOOL_MAP,
    _TOOL_ALIAS,
)

# ── 将 TOOL_REGISTRY 中已注册的工具注入到 LLM Router 的执行映射 ────────────
from agent.tools import llm_router
llm_router._TOOL_FUNC_MAP.update(TOOL_REGISTRY._func_map)

from rag.rag_factory import get_rag_service


# ─────────────────────────────────────────────────────────────────────────────
# 人名合法性校验（拒绝 LLM 幻觉人名）
# ─────────────────────────────────────────────────────────────────────────────

# LLM 常误提取的"人名"（实际是描述词，不是人名）
_BAD_PERSON_NAMES = frozenset({
    "详细介绍", "详细情况", "详细介绍下", "详细情况说明",
    "案件信息", "案件详情", "相关人员", "有关人员",
    "主要人物", "重要人物", "涉案人员", "相关情况",
    "更多详情", "更多情况", "更多内容",
    "以上", "以下", "前面", "后面", "这个人", "那个人",
    "怎么回事", "情况如何", "具体详情", "详细说明",
})

# 人名后缀特征词（带这些后缀的不是人名）
_NAME_BAD_SUFFIXES = (
    "是谁", "是什么", "怎么样", "如何", "的", "情况", "案件", "信息",
    "详情", "详细介绍", "详细情况", "人", "资料", "背景",
)


def _is_valid_person_name(candidate: str, full_query: str = "") -> bool:
    """
    判断 candidate 是否是合法人名，拒绝 LLM 幻觉出来的描述性"人名"。
    """
    if not candidate:
        return False
    # 完全黑名单
    if candidate in _BAD_PERSON_NAMES:
        return False
    # 后缀黑名单
    if any(candidate.endswith(s) for s in _NAME_BAD_SUFFIXES):
        return False
    # 长度限制：至少2字
    if len(candidate) < 2:
        return False
    # 包含数字/英文字母
    if _re.search(r"[a-zA-Z0-9]", candidate):
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# 从对话历史中提取人名
# ─────────────────────────────────────────────────────────────────────────────

def _extract_names_from_history(chat_history: list, current_query: str) -> list[str]:
    """
    从 chat_history 的上一轮助手回答中提取人名（2-4个中文字符）。
    用于"自动更新数据库"等泛化指令场景。
    """
    _KNOWN_TITLE = {
        "书记", "省长", "市长", "县长", "局长", "处长", "部长", "主任",
        "董事长", "总经理", "总裁", "副校长", "校长", "院长", "主席",
        "副主席", "副总理", "厅长", "队长", "会长", "副部长",
    }

    def _looks_like_name(candidate: str) -> bool:
        if len(candidate) < 2 or len(candidate) > 4:
            return False
        if candidate in _KNOWN_TITLE:
            return False
        if _re.match(r"[\u4e00-\u9fa5]{2,4}$", candidate) and not any(c.isdigit() for c in candidate):
            return True
        return False

    # 从最后一轮 assistant 回答中提取人名
    prev_text = ""
    if chat_history:
        for msg in reversed(chat_history):
            if msg.get("role") == "assistant" and msg.get("content"):
                prev_text = msg["content"]
                break

    if not prev_text:
        return []

    # 优先匹配表格中的人名（"韩非"、"张萌"等）
    name_candidates = set()
    # 表格行：2-4个中文字符（排除表头）
    table_pattern = r"(?<=\|)\s*([\u4e00-\u9fa5]{2,4})\s*(?=\|)"
    for m in _re.finditer(table_pattern, prev_text):
        cand = m.group(1).strip()
        if _looks_like_name(cand) and cand not in _KNOWN_TITLE:
            name_candidates.add(cand)

    # 从全文中提取 2-4 个中文字符的人名
    word_pattern = r"([\u4e00-\u9fa5]{2,4})"
    seen_positions = set()
    for m in _re.finditer(word_pattern, prev_text):
        start = m.start()
        end = m.end()
        if start in seen_positions or end in seen_positions:
            continue
        cand = m.group(1).strip()
        if _looks_like_name(cand) and cand not in _KNOWN_TITLE:
            # 排除当前问题中已出现的词（避免重复）
            if cand not in current_query:
                name_candidates.add(cand)
        seen_positions.update(range(start, end))

    # 过滤：至少2字符，且不在职务词表中
    return [n for n in name_candidates if len(n) >= 2]


# ─────────────────────────────────────────────────────────────────────────────
# 填充工具参数（LLM 漏提时的正则回退）
# ─────────────────────────────────────────────────────────────────────────────

def _fill_missing_params(tool_name: str, tool_args: dict, query: str) -> dict:
    """当 LLM 漏提参数时，用正则回退提取；同时验证 LLM 提供的人名是否合法。"""
    # 如果 LLM 提供了 person_name，验证其合法性
    if tool_name in ("query_corruption_by_name", "query_corruption_by_name_tool"):
        person = tool_args.get("person_name", "")
        if person and not _is_valid_person_name(person, query):
            # LLM 提供的人名不合法（可能是"详细介绍"等描述），忽略并走正则回退
            tool_args = {}

    # 参数名映射（LLM 常把 keyword 命名为 query）
    _PARAM_ALIAS = {
        "search_corruption_records": {"query": "keyword"},
    }
    if tool_name in _PARAM_ALIAS:
        aliases = _PARAM_ALIAS[tool_name]
        tool_args = {aliases.get(k, k): v for k, v in tool_args.items()}

    if tool_args:
        return tool_args

    # query_corruption_by_name: 提取人名
    if tool_name in ("query_corruption_by_name", "query_corruption_by_name_tool"):
        patterns = [
            r"([\u4e00-\u9fa5]{2,8})的案件",
            r"([\u4e00-\u9fa5]{2,8})的情况",
            r"([\u4e00-\u9fa5]{2,8})是什么罪",
            r"([\u4e00-\u9fa5]{2,8})的判决",
            r"([\u4e00-\u9fa5]{2,8})涉案",
            r"关于([\u4e00-\u9fa5]{2,8})",
            r"([\u4e00-\u9fa5]{2,8})案",
            r"([\u4e00-\u9fa5]{2,8})进展",
            r"查一下([\u4e00-\u9fa5]{2,6})",
            r"介绍一下([\u4e00-\u9fa5]{2,6})",
            r"查([\u4e00-\u9fa5]{2,6})",
        ]
        for p in patterns:
            m = _re.search(p, query)
            if m:
                candidate = m.group(1)
                if _is_valid_person_name(candidate, query):
                    return {"person_name": candidate}

        tail_patterns = [
            r"(?:介绍|查|看|找|请问|帮我)[的\s]*([\u4e00-\u9fa5]{2,4})",
            r"^(?:介绍一下|查一下|请问|帮我)[的\s]*([\u4e00-\u9fa5]{2,4})$",
            r"([\u4e00-\u9fa5]{2,4})$",
        ]
        _NOT_NAME = {"一下", "情况", "一下的", "一下情况", "这个人", "那个人", "此案", "此情况"}
        _NOT_SUFFIX = ("是谁", "是什么", "怎么样", "如何")

        for p in tail_patterns:
            m = _re.search(p, query.strip())
            if m:
                cand = m.group(1).strip()
                if cand not in _NOT_NAME and not any(cand.endswith(s) for s in _NOT_SUFFIX):
                    if _is_valid_person_name(cand, query):
                        return {"person_name": cand}

        head = _re.match(r"^([\u4e00-\u9fa5]{2,4})(?:的|是|案|情况|涉案|判决|进展|是什么|怎么样了)", query.strip())
        if head and _is_valid_person_name(head.group(1), query):
            return {"person_name": head.group(1)}

        ms = list(_re.finditer(r"[\u4e00-\u9fa5]{2,4}", query))
        valid = [m.group() for m in ms if _is_valid_person_name(m.group(), query)]
        if valid:
            return {"person_name": valid[-1]}
        return {"person_name": query.strip()}

    if tool_name in ("rag_summarize", "rag_summarize_tool"):
        return {"query": query}
    if tool_name == "web_search":
        return {"query": query}
    if tool_name == "web_fetch":
        m = _re.search(r"https?://[^\s<>\"'））]+", query)
        return {"query": m.group(0) if m else query}
    if tool_name == "web_fetch_and_summarize":
        return {"query": query}
    if tool_name == "search_corruption_records":
        kw = _re.sub(r"[的吗呀？?，。！!]", "", query).strip()
        return {"keyword": kw}

    # auto_enrich_and_save: 和人名查询共用同一套提取逻辑
    if tool_name == "auto_enrich_and_save":
        # 先尝试从 query 中直接提取人名（用于直接触发场景）
        patterns = [
            r"([\u4e00-\u9fa5]{2,8})联网",
            r"([\u4e00-\u9fa5]{2,8})更新",
            r"([\u4e00-\u9fa5]{2,8})追加",
        ]
        for p in patterns:
            m = _re.search(p, query)
            if m and _is_valid_person_name(m.group(1), query):
                return {"person_name": m.group(1)}
        # 无法提取人名时，尝试从 query 末尾取中文片段
        ms = list(_re.finditer(r"[\u4e00-\u9fa5]{2,4}", query))
        valid = [m.group() for m in ms if _is_valid_person_name(m.group(), query)]
        if valid:
            return {"person_name": valid[-1]}
        return {"person_name": query.strip()}

    return tool_args


# ─────────────────────────────────────────────────────────────────────────────
# 中间件（报告模式）
# ─────────────────────────────────────────────────────────────────────────────

def _get_ctx_mw():
    from agent.middleware.context_middleware import get_context_middleware
    return get_context_middleware()


# ─────────────────────────────────────────────────────────────────────────────
# 思考步骤数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolCallStep:
    """单个工具调用步骤"""
    icon: str = "🔧"
    title: str = ""
    tool_name: str = ""
    params: dict = field(default_factory=dict)
    reason: str = ""
    result: str = ""
    error: str = ""
    done: bool = False

    def to_step(self) -> "Step":
        if self.error:
            content = f"调用 **{self.tool_name}** 时出错：{self.error}"
        else:
            preview = self.result[:120].replace("\n", " ") if self.result else "(无返回)"
            content = f"**{self.tool_name}** → {preview}..."
        return Step(
            icon="✅" if self.done and not self.error else ("❌" if self.error else "⏳"),
            title=self.title,
            content=content,
            done=self.done,
        )


@dataclass
class Step:
    icon: str
    title: str
    content: str
    done: bool = False


import threading
from queue import Queue


@dataclass
class RouteResult:
    """路由结果：包含多步思考过程 + 最终回答"""
    steps: list[Step | ToolCallStep] = field(default_factory=list)
    answer: str = ""
    plan_raw: str = ""

    def mark_all_done(self):
        for s in self.steps:
            if hasattr(s, "done"):
                s.done = True
            if isinstance(s, Step):
                s.done = True

    def render_trace(self) -> str:
        self.mark_all_done()
        lines = []
        for s in self.steps:
            icon = "✅" if s.done else "⏳"
            if isinstance(s, ToolCallStep):
                lines.append(f"{icon} **{s.title}**\n{s.to_step().content}")
            else:
                lines.append(f"{icon} **{s.title}**\n{s.content}")
        return "\n\n".join(lines)

    def stream_answer(self) -> Generator[str, None, None]:
        for i in range(0, len(self.answer), 8):
            yield self.answer[i:i + 8]


class StreamingRouteResult:
    """
    流式路由结果：
    - 思考步骤立即可用（result.steps）
    - 回答在后台线程计算，通过 stream_answer() 流式获取
    """
    def __init__(self):
        self.steps: list[Step | ToolCallStep] = []
        self.answer: str = ""
        self._answer_queue: Queue = Queue()
        self._done: threading.Event = threading.Event()
        self._thread: threading.Thread | None = None

    def _synthesize_in_thread(self, query: str, tool_results: list):
        """后台线程：LLM 综合，chunk 进队列"""
        try:
            chunks = []
            for chunk in llm_synthesize(query, tool_results, stream=True):
                chunks.append(chunk)
                self._answer_queue.put(chunk)
            full = "".join(chunks)
            self._answer_queue.put(None)   # 结束信号
        except Exception as e:
            # 降级：返回第一个有效工具结果
            for r in tool_results:
                if r.get("result") and not r.get("error"):
                    self._answer_queue.put(r["result"])
                    self._answer_queue.put(None)
                    return
            self._answer_queue.put("综合失败：" + str(e))
            self._answer_queue.put(None)
        finally:
            self._done.set()

    def start_synthesis(self, query: str, tool_results: list):
        """启动后台综合线程"""
        self._thread = threading.Thread(target=self._synthesize_in_thread,
                                         args=(query, tool_results), daemon=True)
        self._thread.start()

    def stream_answer(self) -> Generator[str, None, None]:
        """流式获取 answer chunk（来自队列）"""
        if self._done.is_set():
            # 综合已完成，直接从 answer 切片
            for i in range(0, len(self.answer), 8):
                yield self.answer[i:i + 8]
            return
        # 从队列消费
        while True:
            chunk = self._answer_queue.get()
            if chunk is None:
                break
            yield chunk

    def is_done(self) -> bool:
        return self._done.is_set()


# ─────────────────────────────────────────────────────────────────────────────
# RAG 工具
# ─────────────────────────────────────────────────────────────────────────────

from langchain_core.tools import tool as _tool


def _get_report_system_prompt() -> str:
    global _REPORT_SYSTEM_PROMPT_CACHE
    if _REPORT_SYSTEM_PROMPT_CACHE is None:
        import os as _os
        _root = _os.path.dirname(_os.path.dirname(_os.path.dirname(os.path.abspath(__file__))))
        _path = _os.path.join(_root, "prompts", "report_prompt.txt")
        try:
            with open(_path, "r", encoding="utf-8") as f:
                _REPORT_SYSTEM_PROMPT_CACHE = f.read()
        except Exception:
            _REPORT_SYSTEM_PROMPT_CACHE = ""
    return _REPORT_SYSTEM_PROMPT_CACHE


_REPORT_SYSTEM_PROMPT_CACHE = None


@_tool(description='RAG summarization tool. For concept/definition/background questions like "what is corruption crime". If references are empty, auto falls back to web search.')
def rag_summarize_tool(query: str) -> str:
    rag = get_rag_service()
    ctx_mw = _get_ctx_mw()
    report_active = ctx_mw.is_active() and ctx_mw._active_context.get("name") == "report"
    if not report_active:
        return rag.summarize(query)
    try:
        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.runnables import RunnablePassthrough
        from model.factory import chat_model
        rp = _get_report_system_prompt()
        if not rp:
            return rag.summarize(query)
        tpl = PromptTemplate.from_template(rp + "\n\n用户问题：{input}\n\n参考资料：{context}")
        chain = (
            RunnablePassthrough.assign(_input=lambda x: x.get("input", ""), _context=lambda x: x.get("context", ""))
            | tpl | chat_model | StrOutputParser()
        )
        docs = rag.retrieve(query)
        ctx = "\n".join(f"【{i}】: {d.page_content}" for i, d in enumerate(docs, 1))
        return chain.invoke({"input": query, "context": ctx})
    except Exception:
        return rag.summarize(query)


# 注册到 LLM tool map
_LLM_TOOL_MAP["rag_summarize"] = rag_summarize_tool


# ─────────────────────────────────────────────────────────────────────────────
# 闲聊回复
# ─────────────────────────────────────────────────────────────────────────────

_GENERAL_CHAT = (
    "您好！我是**贪腐记录检索助手**，专门帮助您查询贪污受贿等腐败案件的记录。\n\n"
    "我可以帮您：\n"
    '- 按人名查询案件（如"谭瑞松的判决"）\n'
    '- 按金额/时间排序（如"涉案金额最高的前10名"）\n'
    "- 搜索特定年份或地区的案件\n"
    "- 推荐重要/典型案件\n"
    "- 联网搜索最新反腐动态\n\n"
    "请告诉我您想查询什么？"
)


# ─────────────────────────────────────────────────────────────────────────────
# 核心路由入口
# ─────────────────────────────────────────────────────────────────────────────

def route_and_answer_with_trace(
    user_query: str,
    chat_history: list | None = None,
) -> RouteResult:
    """
    多工具规划 + 并发执行 + LLM 综合。
    返回 RouteResult（含多步思考过程 + 最终回答）。
    """
    query = user_query.strip()
    result = RouteResult()

    if not query:
        result.answer = "问题为空，请重新输入。"
        return result

    # ── Step 1: 问题分析 ────────────────────────────────────────────────
    result.steps.append(Step(
        icon="🔍", title="问题分析",
        content=f"接收到问题：**{query}**",
    ))

    # ── Step 2: 多工具规划（LLM）────────────────────────────────────────
    tool_descs = build_tool_descriptions(TOOL_REGISTRY)

    try:
        plan = llm_plan_tools(query, tool_descs, chat_history=chat_history)
    except Exception:
        plan = None

    if plan is None:
        # 降级到单工具路由
        try:
            single = llm_route_single(query, tool_descs, chat_history=chat_history)
            plan = [{
                "name": single["tool"],
                "params": _fill_missing_params(single["tool"], single["params"], query),
                "reason": single.get("reason", "单工具路由"),
            }]
            plan_reason = single.get("reason", "")
        except Exception:
            plan = []
            plan_reason = "规划失败"
    else:
        # 填充参数（LLM 漏提时用正则回退）
        for call in plan:
            call["params"] = _fill_missing_params(call["name"], call.get("params") or {}, query)
        plan_reason = " + ".join(c.get("reason", "") for c in plan[:3])

    # general_chat
    if plan and plan[0]["name"] == "general_chat":
        result.steps.append(Step(icon="💡", title="路由决策",
            content=f"识别为闲聊（{plan_reason}）", done=True))
        result.steps.append(Step(icon="✅", title="闲聊直接回复", content="", done=True))
        result.answer = _GENERAL_CHAT
        result.mark_all_done()
        return result

    # ── Step 3: 显示规划决策 ───────────────────────────────────────────
    tool_names = " / ".join(c["name"] for c in plan) if plan else "（无）"
    result.steps.append(Step(
        icon="💡", title="工具规划",
        content=f"规划调用 {len(plan)} 个工具：{tool_names}（{plan_reason}）",
    ))

    if not plan:
        # 兜底：RAG
        plan = [{"name": "rag_summarize", "params": {"query": query}, "reason": "兜底"}]

    # ── Step 4: 并发执行所有工具 ────────────────────────────────────────
    tool_call_steps: list[ToolCallStep] = []
    for i, call in enumerate(plan):
        step = ToolCallStep(
            icon="⏳",
            title=f"调用工具 {i + 1}/{len(plan)}",
            tool_name=call["name"],
            params=call["params"],
            reason=call.get("reason", ""),
            result="",
            error="",
            done=False,
        )
        result.steps.append(step)
        tool_call_steps.append(step)

    # 并发执行
    exec_results = execute_tools([
        {"name": c["name"], "params": c["params"]} for c in plan
    ])

    # 填充执行结果到步骤
    for i, exec_r in enumerate(exec_results):
        step = tool_call_steps[i]
        step.result = exec_r.get("result", "")
        step.error = exec_r.get("error", "") or ""
        step.done = True
        # 更新步骤的显示内容
        if step.error:
            step.icon = "❌"
        else:
            step.icon = "✅"
            preview = step.result[:150].replace("\n", " ") if step.result else "(无返回)"
            step.content = f"**{step.tool_name}** → {preview}..."

    # ── 兜底：当 query_corruption_by_name 返回空时，自动补全到 CSV ────────
    # 从已执行结果中，找出"人名查询返回空"的记录
    _EMPTY_NAME_PATTERNS = ("未找到当事人", "未找到", "empty", "not found")
    extra_calls = []          # [{"name": ..., "params": {...}}, ...]
    extra_steps = []          # [ToolCallStep, ...]

    for i, step in enumerate(tool_call_steps):
        tool_name = step.tool_name
        res = step.result or ""
        if tool_name in ("query_corruption_by_name", "query_corruption_by_name_tool"):
            is_empty = any(pat in res for pat in _EMPTY_NAME_PATTERNS)
            if is_empty:
                # 从原始 plan 提取人名（params 中应有）
                person = plan[i].get("params", {}).get("person_name", "")
                if person and person not in _EMPTY_NAME_PATTERNS:
                    extra_calls.append({
                        "name": "auto_enrich_and_save",
                        "params": {"person_name": person},
                    })
                    s = ToolCallStep(
                        icon="⏳",
                        title="自动补全",
                        tool_name="auto_enrich_and_save",
                        params={"person_name": person},
                        reason="数据库无此人记录，自动联网搜索并追加到CSV",
                        result="",
                        error="",
                        done=False,
                    )
                    extra_steps.append(s)

    # 统一执行额外工具调用（避免后续重复调用）
    extra_results = execute_tools(extra_calls) if extra_calls else []

    # 如果有额外调用，追加到思考步骤
    if extra_calls:
        result.steps.append(Step(
            icon="💡", title="自动补全",
            content=f"检测到 {len(extra_calls)} 个人名不在数据库，联网搜索并根据结果决定是否追加到数据库",
        ))
        for i, extra_r in enumerate(extra_results):
            step = extra_steps[i]
            step.result = extra_r.get("result", "")
            step.error = extra_r.get("error", "") or ""
            step.done = True
            step.icon = "✅" if not step.error else "❌"
            preview = step.result[:150].replace("\n", " ") if step.result else "(无返回)"
            step.content = f"**auto_enrich_and_save** → {preview}..."
            result.steps.append(step)

    # ── Step 5: LLM 综合结果 ───────────────────────────────────────────
    # 构建 tool_results 列表（供综合用）
    tool_results_for_synth = [
        {
            "name": exec_results[i].get("name", plan[i]["name"]),
            "result": tool_call_steps[i].result,
            "error": tool_call_steps[i].error,
        }
        for i in range(len(plan))
    ]
    # 加入已执行的 auto_enrich 结果供综合
    for extra_r in extra_results:
        tool_results_for_synth.append({
            "name": extra_r.get("name", "auto_enrich_and_save"),
            "result": extra_r.get("result", ""),
            "error": extra_r.get("error", ""),
        })

    result.steps.append(Step(
        icon="🧠", title="综合结果",
        content="正在由 LLM 综合所有工具结果生成最终回答...",
    ))

    # LLM 综合（stream=False 仍返回 generator，需收集）
    try:
        chunks = list(llm_synthesize(query, tool_results_for_synth, stream=False))
        result.answer = "".join(chunks)
    except Exception as e:
        # 综合失败，降级返回第一个有效结果
        for r in tool_results_for_synth:
            if r.get("result") and not r.get("error"):
                result.answer = r["result"]
                break
        else:
            result.answer = "综合失败：" + str(e)

    # 更新综合步骤
    if result.steps[-1].title == "综合结果":
        result.steps[-1].content = "LLM 综合完成，已生成最终回答"
        result.steps[-1].done = True
        result.steps[-1].icon = "✅"

    result.mark_all_done()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 非流式入口（兼容旧代码）
# ─────────────────────────────────────────────────────────────────────────────

def route_and_answer(user_query: str, chat_history: list | None = None) -> str:
    """非流式入口：直接返回完整回答字符串"""
    result = _route_non_stream(user_query, chat_history)
    return result.answer


def _route_non_stream(user_query: str, chat_history: list | None = None) -> RouteResult:
    """内部非流式实现（不走 yield，直接收集结果）"""
    query = user_query.strip()
    result = RouteResult()

    if not query:
        result.answer = "问题为空，请重新输入。"
        return result

    result.steps.append(Step(icon="🔍", title="问题分析",
        content=f"接收到问题：**{query}**"))

    tool_descs = build_tool_descriptions(TOOL_REGISTRY)

    try:
        plan = llm_plan_tools(query, tool_descs, chat_history=chat_history)
    except Exception:
        plan = None

    if plan is None:
        try:
            single = llm_route_single(query, tool_descs, chat_history=chat_history)
            plan = [{"name": single["tool"],
                     "params": _fill_missing_params(single["tool"], single["params"], query),
                     "reason": single.get("reason", "")}]
        except Exception:
            plan = []

    if plan and plan[0]["name"] == "general_chat":
        result.answer = _GENERAL_CHAT
        result.mark_all_done()
        return result

    if not plan:
        plan = [{"name": "rag_summarize", "params": {"query": query}, "reason": "兜底"}]

    tool_names = " / ".join(c["name"] for c in plan)
    result.steps.append(Step(icon="💡", title="工具规划",
        content=f"调用 {len(plan)} 个工具：{tool_names}"))

    exec_results = execute_tools([{"name": c["name"], "params": c["params"]} for c in plan])
    for i, r in enumerate(exec_results):
        preview = r.get("result", "")[:100].replace("\n", " ") or "(无)"
        err = r.get("error", "")
        icon = "❌" if err else "✅"
        content = f"**{r.get('name', '?')}** → {err or preview}..."
        result.steps.append(Step(icon=icon, title=f"工具 {i+1} 结果", content=content, done=True))

    tool_results = [
        {"name": exec_results[i].get("name", plan[i]["name"]),
         "result": exec_results[i].get("result", ""),
         "error": exec_results[i].get("error", "")}
        for i in range(len(plan))
    ]

    try:
        chunks = list(llm_synthesize(query, tool_results, stream=False))
        result.answer = "".join(chunks)
    except Exception as e:
        for r in tool_results:
            if r.get("result") and not r.get("error"):
                result.answer = r["result"]
                break
        else:
            result.answer = "综合失败：" + str(e)

    result.mark_all_done()
    return result


def route_and_answer_stream(
    user_query: str,
    chat_history: list | None = None,
) -> Generator[str, None, None]:
    """流式入口：yield 思考步骤更新，yield 最终答案 token"""
    query = user_query.strip()
    if not query:
        yield "问题为空，请重新输入。"
        return

    tool_descs = build_tool_descriptions(TOOL_REGISTRY)

    # 规划
    try:
        plan = llm_plan_tools(query, tool_descs, chat_history=chat_history)
    except Exception:
        plan = None

    if plan is None:
        try:
            single = llm_route_single(query, tool_descs, chat_history=chat_history)
            plan = [{"name": single["tool"],
                     "params": _fill_missing_params(single["tool"], single["params"], query),
                     "reason": single.get("reason", "")}]
        except Exception:
            plan = [{"name": "rag_summarize", "params": {"query": query}, "reason": "兜底"}]

    if plan and plan[0]["name"] == "general_chat":
        for chunk in _GENERAL_CHAT:
            yield chunk
        return

    if not plan:
        plan = [{"name": "rag_summarize", "params": {"query": query}, "reason": "兜底"}]

    # 执行
    exec_results = execute_tools([{"name": c["name"], "params": c["params"]} for c in plan])
    tool_results = [
        {"name": exec_results[i].get("name", plan[i]["name"]),
         "result": exec_results[i].get("result", ""),
         "error": exec_results[i].get("error", "")}
        for i in range(len(plan))
    ]

    # 流式综合
    for chunk in llm_synthesize(query, tool_results, stream=True):
        yield chunk


def route_and_answer_realtime(
    user_query: str,
    chat_history: list | None = None,
) -> Generator[str, None, None]:
    """
    实时流式路由：每个工具执行完立即 yield 步骤，LLM 综合也流式 yield。
    yield 格式：
      __STEP__<json>  — 步骤更新（json 含 icon/title/content/done 字段）
      __ANSWER__       — 以下为最终答案 token
      <token>          — LLM 生成的答案片段
      __DONE__         — 流结束
    """
    import json
    query = user_query.strip()
    if not query:
        yield "__DONE__"
        return

    def _step(icon, title, content, done=False):
        return "__STEP__" + json.dumps({"icon": icon, "title": title, "content": content, "done": done}, ensure_ascii=False)

    tool_descs = build_tool_descriptions(TOOL_REGISTRY)

    # 规划
    try:
        plan = llm_plan_tools(query, tool_descs, chat_history=chat_history)
    except Exception:
        plan = None

    if plan is None:
        try:
            single = llm_route_single(query, tool_descs, chat_history=chat_history)
            plan = [{
                "name": single["tool"],
                "params": _fill_missing_params(single["tool"], single.get("params") or {}, query),
                "reason": single.get("reason", ""),
            }]
        except Exception:
            plan = []

    if plan and plan[0]["name"] == "general_chat":
        for ch in _GENERAL_CHAT:
            yield ch
        yield "__DONE__"
        return

    if not plan:
        plan = [{"name": "rag_summarize", "params": {"query": query}, "reason": "兜底"}]

    # 规划步骤
    tool_names = " / ".join(c["name"] for c in plan)
    plan_reason = " + ".join(c.get("reason", "") for c in plan[:3])
    yield _step("🔍", "问题分析", f"接收到问题：**{query}**")
    yield _step("💡", "工具规划", f"规划调用 {len(plan)} 个工具：{tool_names}（{plan_reason}）")

    # 执行每个工具（串行，逐个 yield 步骤）
    _EMPTY_PATTERNS = ("未找到当事人", "未找到", "empty", "not found")
    all_results = []   # [{"name": ..., "result": ..., "error": ...}, ...]
    auto_enrich_queue = []  # [(person, step_index), ...]

    for i, call in enumerate(plan):
        name = call["name"]
        params = call.get("params") or {}

        yield _step(
            "⏳", f"调用工具 {i + 1}/{len(plan)}",
            f"**{name}** — {call.get('reason', '')}"
        )

        single_result = invoke_single_tool(name, params)
        err = single_result.get("error", "")
        res = single_result.get("result", "")
        icon = "❌" if err else "✅"
        preview = (err or res[:120].replace("\n", " ")) + "..."
        yield _step(icon, f"工具 {i + 1} 结果", f"**{name}** → {preview}", done=True)

        all_results.append({"name": name, "result": res, "error": err})

        # ── RAG 为空时自动降级到联网搜索 ─────────────────────────────
        if name == "rag_summarize" and res and "未涉及" in res:
            yield _step(
                "💡", "自动联网搜索",
                f"RAG 未找到相关资料，自动切换到联网搜索补充信息"
            )
            ws_result = invoke_single_tool("web_search", {"query": query})
            ws_err = ws_result.get("error", "")
            ws_res = ws_result.get("result", "")
            ws_icon = "❌" if ws_err else "✅"
            ws_preview = (ws_err or ws_res[:120].replace("\n", " ")) + "..."
            yield _step(ws_icon, "联网搜索结果", f"**web_search** → {ws_preview}", done=True)
            all_results.append({"name": "web_search", "result": ws_res, "error": ws_err})

        # ── 人名查询为空时加入自动补全队列 ─────────────────────────
        if name in ("query_corruption_by_name", "query_corruption_by_name_tool"):
            is_empty = any(pat in res for pat in _EMPTY_PATTERNS)
            if is_empty:
                person = params.get("person_name", "")
                if person and person not in _EMPTY_PATTERNS:
                    auto_enrich_queue.append(person)

        # ── "自动更新数据库"：从上一轮回答中提取人名 ───────────────
        if name == "auto_enrich_and_save" and not params.get("person_name"):
            # 尝试从 chat_history 中提取人名
            names_from_history = _extract_names_from_history(chat_history or [], query)
            for person in names_from_history:
                if person and person not in auto_enrich_queue:
                    auto_enrich_queue.append(person)
            if not names_from_history:
                # 连上下文都没有，说明没有上一轮内容
                pass

    # 自动补全（逐个执行并 yield）
    if auto_enrich_queue:
        yield _step(
            "💡", "自动补全",
            f"检测到 {len(auto_enrich_queue)} 个人名不在数据库，自动联网搜索并根据是否贪污决定是否追加到数据库"
        )
        for j, person in enumerate(auto_enrich_queue):
            yield _step("⏳", "自动补全", f"正在搜索「{person}」...")
            enrich_result = invoke_single_tool("auto_enrich_and_save", {"person_name": person})
            err = enrich_result.get("error", "")
            res = enrich_result.get("result", "")
            icon = "❌" if err else "✅"
            preview = (err or res[:120].replace("\n", " ")) + "..."
            yield _step(icon, "自动补全", f"**auto_enrich_and_save** → {preview}", done=True)
            all_results.append({"name": "auto_enrich_and_save", "result": res, "error": err})

    # LLM 综合（流式 yield token，把对话历史也传入以便模型有记忆）
    yield "__ANSWER__"
    for chunk in llm_synthesize(query, all_results, stream=True, chat_history=chat_history):
        yield chunk

    yield "__DONE__"


# ─────────────────────────────────────────────────────────────────────────────
# 调试入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("多工具路由测试（并发执行 + LLM 综合）")
    print("=" * 60)

    tests = [
        "谭瑞松案最新进展如何？",
        "涉案金额最高的前10名是谁？有哪些最新案件？",
        "贪污罪的构成要件是什么？",
        "广东省有哪些案件？最新进展如何？",
    ]

    for q in tests:
        print(f"\n{'=' * 50}")
        print(f"问题：{q}")
        print("-" * 50)
        res = _route_non_stream(q, None)
        print(res.render_trace())
        print()
        print("最终答案：")
        print(res.answer[:300] if res.answer else "(空)")
        print()
