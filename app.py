"""
贪腐记录检索助手 — Streamlit 聊天界面
"""

import streamlit as st
import sys, os, re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def get_agent():
    from agent import AgentFactory, get_context_middleware
    return AgentFactory.create(), get_context_middleware()


st.set_page_config(
    page_title="贪腐记录检索助手",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── 样式 ─────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ========== 悬浮输入框 ========== */
[data-testid="stMain"] > div { padding-bottom: 90px !important; }
[data-testid="stChatInputContainer"] {
    position: fixed !important; bottom: 0 !important;
    left: 0 !important; right: 0 !important; z-index: 9999 !important;
    background: #ffffff !important; border-top: 1px solid #e5e7eb !important;
    padding: 0.7rem 1.5rem !important;
    box-shadow: 0 -2px 16px rgba(0,0,0,0.07) !important;
}
[data-testid="stChatInputContainer"] input {
    background: #f0f2f6 !important; border-radius: 20px !important;
    border: 1px solid #d1d5db !important;
    padding: 0.4rem 1rem !important; font-size: 0.95rem !important;
}

/* ========== 思考过程 — 紧凑单行横条 ========== */
.thinking-card {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 8px;
    padding: 4px 10px;
    margin-bottom: 4px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.thinking-card-header { display: flex; align-items: center; gap: 6px; }
.thinking-card-icon { font-size: 0.85rem; flex-shrink: 0; }
.thinking-card-title {
    font-size: 0.78rem; font-weight: 700; color: #1e40af;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 120px;
}
.thinking-card-done-badge {
    font-size: 0.62rem; padding: 1px 5px; border-radius: 6px;
    font-weight: 600; flex-shrink: 0; margin-left: auto;
}
.badge-done-card    { background: #dcfce7; color: #16a34a; }
.badge-pending-card { background: #fef9c3; color: #a16207; }
.thinking-card-content {
    font-size: 0.72rem; color: #64748b;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    flex-shrink: 1;
}
.thinking-card-content table { display: none; }

/* ========== 助手消息气泡 ========== */
[data-testid="stChatMessageContainer"] > div {
    border-radius: 14px !important;
}
</style>
""", unsafe_allow_html=True)


# ── session_state ───────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "_pending_query" not in st.session_state:
    st.session_state["_pending_query"] = None


# ── 侧边栏 ──────────────────────────────────────────────────────────────────

examples = [
    "贪污的人哪个金额最高？",
    "殷美根的判决是什么？",
    "有什么值得关注的重大案件？",
    "谭瑞松案最新进展如何？",
    "2026年有哪些案件？",
    "什么是贪污罪？",
]

with st.sidebar:
    st.title("⚙️ 设置")

    mode = st.radio(
        "输出模式",
        ["🚀 流式输出", "📦 完整输出"],
        captions=["逐字显示，打字机效果", "等待完整回答后一次性显示"],
    )
    use_stream = (mode == "🚀 流式输出")

    st.divider()
    st.markdown("**📋 报告生成模式**")
    st.checkbox("启用报告模式", value=False, key="_report_mode")

    st.divider()
    st.markdown("**💡 快捷问题**")
    for ex in examples:
        st.button(
            ex,
            key=f"ex_{ex}",
            on_click=lambda q=ex: st.session_state.update({"_pending_query": q}),
        )

    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state["chat_history"] = []
        st.rerun()


# ── 标题 ─────────────────────────────────────────────────────────────────────

st.title("🔍 贪腐记录检索助手")
st.caption("RAG + 精确数据查询双轨架构 · 支持流式输出 · 报告生成")

if st.session_state.get("_report_mode"):
    st.info("📋 报告模式已启用：后续回答将以舆情分析报告格式呈现。")

st.divider()


# ── 工具函数 ────────────────────────────────────────────────────────────────

def _parse_md_table(content: str) -> tuple[list[str], list[list[str]]] | None:
    """
    解析 content 中的 markdown 表格，支持两种格式：
    1. 管道格式：| 案件序号 | 当事人姓名 | ...
    2. 空格分隔格式：案件序号    当事人姓名    ...
    返回 (headers, rows)，解析失败返回 None。
    """
    # ── 管道格式 ────────────────────────────────────────────────────────
    lines = content.strip().split("\n")
    pipe_rows = [
        ln for ln in lines
        if re.match(r"\|\s*[\u4e00-\u9fa5A-Za-z0-9]", ln)
    ]
    if len(pipe_rows) >= 2:
        def parse_pipe_row(ln: str) -> list[str]:
            return [c.strip() for c in ln.strip().strip("|").split("|")]
        headers = parse_pipe_row(pipe_rows[0])
        rows = [parse_pipe_row(ln) for ln in pipe_rows[1:]]
        return headers, rows

    # ── 空格分隔格式（多列以 2+ 空格分隔）─────────────────────────────────
    text_lines = [
        ln.strip() for ln in lines
        if ln.strip() and not ln.strip().startswith("#")
        and not re.match(r"[\|\-\s]+", ln.strip())
    ]
    if not text_lines:
        return None

    # 用 2+ 连续空格拆分
    def split_by_space(ln: str) -> list[str]:
        return [s.strip() for s in re.split(r"\s{2,}", ln) if s.strip()]

    # 找出列数最稳定的行作为表头
    candidates = [(ln, split_by_space(ln)) for ln in text_lines]
    # 过滤出列数 >= 3 的行（至少要有3列）
    valid = [(ln, cells) for ln, cells in candidates if len(cells) >= 3]
    if not valid:
        return None

    # 取列数最多的行作为表头
    headers_line, headers = max(valid, key=lambda x: len(x[1]))
    col_count = len(headers)

    # 其余行按列数过滤（与表头列数接近的视为数据行）
    rows = []
    for ln, cells in candidates:
        # 过滤掉与表头列数差太多、或首列不是数字/中文的行
        if abs(len(cells) - col_count) <= 2 and re.search(r"[\u4e00-\u9fa5A-Za-z0-9]", cells[0]):
            # 截齐到 col_count
            rows.append(cells[:col_count])

    if not rows:
        return None
    return headers, rows


def _render_thinking_cards(steps: list, container=None):
    """渲染思考过程：紧凑单行横条列表"""
    out = container.empty() if container else st

    cards_html = ""
    for s in steps:
        icon = s.get("icon", "⏳")
        title = s.get("title", "")
        content_raw = s.get("content", "")
        is_done = s.get("done", False)
        has_error = bool(s.get("error"))

        if has_error:
            badge = '<span class="thinking-card-done-badge" style="background:#fee2e2;color:#dc2626;">❌</span>'
        elif is_done:
            badge = '<span class="thinking-card-done-badge badge-done-card">✓</span>'
        else:
            badge = '<span class="thinking-card-done-badge badge-pending-card">...</span>'

        # 单行模式：截断超长内容，隐藏表格
        max_char_count = 140
        content_short = content_raw.strip()[:max_char_count].replace("\n", " ").replace("|", " ")
        if len(content_raw.strip()) > max_char_count:
            content_short += "..."

        card_html = f"""
        <div class="thinking-card">
            <div class="thinking-card-header">
                <span class="thinking-card-icon">{icon}</span>
                <span class="thinking-card-title">{title}</span>
            </div>
            <div class="thinking-card-content">{content_short}</div>
            {badge}
        </div>"""
        cards_html += card_html

    out.markdown(cards_html, unsafe_allow_html=True)


# ── 渲染历史消息 ──────────────────────────────────────────────────────────────

for item in st.session_state["chat_history"]:
    with st.chat_message(item["role"]):
        if item["role"] == "assistant" and item.get("thinking"):
            _render_thinking_cards(item["thinking"])
        st.markdown(item["content"])


# ── 聊天输入框 ─────────────────────────────────────────────────────────────

user_input = st.chat_input("请输入您的问题...", key="_chat_input")

_pending = st.session_state.get("_pending_query")
query = user_input.strip() if user_input else (_pending if _pending else None)

if query:
    # 快捷按钮触发的查询：清空 _pending
    if _pending:
        st.session_state["_pending_query"] = None

    # ── 用户消息 ──────────────────────────────────────────────────────────
    st.chat_message("user").markdown(query)
    st.session_state["chat_history"].append({
        "role": "user", "content": query, "thinking": [],
    })

    # ── 执行查询 ──────────────────────────────────────────────────────────
    _use_stream = use_stream

    from agent.tools.router import route_and_answer_realtime

    try:
        gen = route_and_answer_realtime(query, chat_history=st.session_state.get("chat_history", []))
    except Exception as e:
        import traceback
        gen = iter([
            f"⚠️ 初始化失败：{type(e).__name__}: {str(e)}\n\n```\n{traceback.format_exc()}\n```",
            "__DONE__",
        ])

    steps = []
    answer_chunks = []

    if _use_stream:
        # ── 流式输出 ──────────────────────────────────────────────────────
        thinking_ph = st.empty()
        answer_ph   = st.empty()

        for item in gen:
            if item == "__DONE__":
                break
            if item.startswith("__STEP__"):
                import json
                try:
                    step_data = json.loads(item[len("__STEP__"):])
                except Exception:
                    continue
                steps.append(step_data)
                _render_thinking_cards(steps, container=thinking_ph)
            elif item == "__ANSWER__":
                continue
            else:
                answer_chunks.append(item)
                answer_ph.markdown("".join(answer_chunks))

        final_answer = "".join(answer_chunks)
    else:
        # ── 非流式输出 ─────────────────────────────────────────────────────
        with st.spinner("🔄 正在思考，请稍候..."):
            final_steps = []
            final_chunks = []
            for item in gen:
                if item == "__DONE__":
                    break
                if item.startswith("__STEP__"):
                    import json
                    try:
                        step_data = json.loads(item[len("__STEP__"):])
                    except Exception:
                        continue
                    final_steps.append(step_data)
                elif item == "__ANSWER__":
                    continue
                else:
                    final_chunks.append(item)
            final_answer = "".join(final_chunks)
            steps = final_steps

    # ── 保存助手消息到历史 ────────────────────────────────────────────────
    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": final_answer,
        "thinking": steps,
    })

    # ── 刷新页面，让历史循环渲染完整对话 ─────────────────────────────────
    st.rerun()
