# 贪腐记录检索助手 — 项目文档

> 本文档记录项目架构、运行方式及常见坑点，方便后续维护。
> 最后更新：2026-04-16（坑15：scripts/migrate_csv_to_mysql.py 为一次性迁移脚本，不参与日常运行）

---

## 一、项目概述

**名称**：贪腐记录检索助手
**架构**：RAG + 精确数据查询双轨 Agent
**定位**：一个带思考过程展示的对话助手，底层是规则路由 + Tool Calling

核心解决的问题：
- RAG 语义检索在"排序/比较/查找"类问题上不可靠 → 用规则路由 + CSV 工具精确查询
- 模型直接调用工具不可靠（通义千问 bind_tools 行为不稳定）→ 规则路由替代 LLM 路由

---

## 二、快速启动

```bash
# 方式1：Streamlit 界面
F:\Python312\python.exe -m streamlit run app.py --server.headless true --server.port 8501

# 方式2：仅测试路由逻辑
F:\Python312\python.exe agent/tools/router.py

# 方式3：测试 RAG 组件
F:\Python312\python.exe rag/rag_test.py
```

> ⚠️ 必须使用 `F:\Python312\python.exe`，不能用系统默认 python。依赖包（langchain、streamlit、dashscope 等）均装在 Python 3.12 环境。

---

## 三、目录结构

```
agent智能体项目/
├── app.py                        # Streamlit 聊天界面入口
├── requirements.txt               # 依赖清单
│
├── agent/                        # Agent 核心
│   ├── __init__.py               # 统一导出
│   ├── agent_factory.py           # AgentFactory：创建/组装 Agent
│   ├── middleware/               # 中间件
│   │   ├── base.py               # BaseMiddleware + MiddlewareManager
│   │   ├── context_middleware.py # 上下文注入（报告场景）
│   │   └── streaming_middleware.py # 流式输出（当前未直接使用）
│   └── tools/
│       ├── registry.py             # @auto_tool 装饰器 + TOOL_REGISTRY 全局注册表
│       ├── llm_router.py           # LLM 规划/综合/单工具路由（当前禁用）
│       ├── router.py               # 规则路由核心（最重要）
│       ├── data_query_tool.py      # CSV 数据查询工具（4个@tool）
│       └── web_search_tool.py      # 联网搜索 + 新闻推荐工具
│
├── rag/                          # RAG 组件
│   ├── vector_store.py           # ChromaDB 连接封装
│   ├── rag_service.py            # RagSummarizeService（纯组件，依赖注入）
│   ├── rag_factory.py            # 工厂：组装 retriever + prompt + chain
│   └── rag_test.py               # RAG 组件测试
│
├── model/
│   └── factory.py                 # chat_model / embed_model 全局单例
│
├── utils/                        # 基础设施
│   ├── config_handler.py         # YAML 配置加载（rag_conf, chroma_conf, prompts_conf）
│   ├── prompt_loader.py           # 提示词加载（仅 load_rag_prompts 被外部使用）
│   ├── path_tool.py               # 路径工具
│   ├── logger_handler.py          # 日志
│   ├── file_handler.py           # CSV/PDF/TXT 文件读取（被 vector_store 使用）
│   └── db_handler.py            # MySQL 连接池（MySQL 优先，CSV 降级）
│
├── config/
│   ├── rag.yaml                  # chat_model_name, embedding_model_name
│   ├── chroma.yaml               # collection_name, k, chunk_size 等
│   ├── prompts.yaml              # 提示词文件路径
│   └── agent.yaml               # 高德 key、联网超时等
│
├── prompts/
│   ├── main_prompt.txt           # Agent 系统提示词
│   ├── rag_summarize.txt        # RAG 摘要提示词（发给 LLM 的模板）
│   └── report_prompt.txt        # 报告生成提示词
│
├── data/
│   └── 贪污记录.csv              # 30 条贪腐案件记录（核心数据源）
│
├── scripts/
│   └── migrate_csv_to_mysql.py   # 一次性迁移脚本（CSV → MySQL，非日常运行）

└── chroma_db/                   # ChromaDB 向量数据库
    └── chroma.sqlite3           # SQLite 元数据（collections/embeddings/segments）
```

---

## 四、架构流程图

```
用户提问
    ↓
agent/tools/router.py  ← 规则路由（核心入口）
    ↓
┌───────────────────────────────────────────────┐
│  ① 问题分析（Step: 🔍 问题分析）             │
│  ② 规则匹配（Step: 💡 路由决策）             │
│  ③ 参数提取（Step: ⚙️ 提取参数）             │
│  ④ 工具调用（Step: 🔧 调用工具）             │
│  ⑤ 结果返回（Step: ✅ 工具返回）             │
└───────────────────────────────────────────────┘
    ↓
    ├─ rank_corruption_records   → 按金额/时间排序（CSV精确查询）
    ├─ query_corruption_by_name → 按人名精确查找
    ├─ search_corruption_records → 关键词全文搜索
    ├─ get_all_corruption_records → 返回全部/按年份过滤
    ├─ web_search               → DuckDuckGo 联网搜索
    ├─ recommend_news            → 案件推荐（金额Top5 + 最新 + 层级分布）
    └─ rag_summarize_tool       → RAG 语义检索 + LLM 总结
    ↓
Streamlit UI  → 思考过程卡片 + Markdown 答案
```

---

## 五、路由规则说明

### 5.1 规则优先级（列表顺序 = 匹配优先级）

| 优先级 | 规则名 | tool_key | 触发关键词示例 |
|---|---|---|---|
| 1 | 联网搜索 | `web_search` | 最新、进展、新闻、报道、网上搜 |
| 2 | 新闻推荐 | `recommend` | 推荐、值得关注、重要案件 |
| 3 | 按人名精确查找 | `by_name` | XX的判决、XX涉案 |
| 4 | 排序比较 | `rank` | 最高、最低、最新、第N名 |
| 5 | **查全部记录** | `all` | **全部案件**、有哪些案件、2026年有哪些 |
| 6 | 关键词搜索 | `search` | 省份、国企、案子 |

> ⚠️ **`all` 必须放在 `search` 之前**。`search` 里有一段时间匹配规则（`\d{4}年`），如果 `all` 放后面，"全部案件"可能先被 search 里的 `r"案子|案例"` 抢走。

> ⚠️ **`all` 规则里的 `r"有哪些案件"` 和 `search` 规则里的 `r"有哪些.*案件"` 容易冲突**。解决方案：`all` 优先匹配 `r"有哪些案件"`，`search` 只保留日期/地点/职务等客观字段。

### 5.2 规则新增方法

在 `agent/tools/router.py` 的 `ROUTING_RULES` 列表中新增一个 dict：

```python
{
    "name": "规则中文名",
    "tool_key": "tool_key名称",
    "patterns": [
        r"正则表达式1",
        r"正则表达式2",
    ],
    "re_flags": re.IGNORECASE,
},
```

同时在 `_build_tool_result` 和 `TOOLS` 字典中补充对应的处理逻辑。

### 5.3 为什么用规则路由而不是 LLM 路由？

通义千问（ChatTongyi）的 `bind_tools` 行为不稳定，模型可能直接生成回答而不调用工具。规则路由的优势：
- 确定性：匹配规则100%准确
- 可调试：正则匹配一目了然
- 性能：无需额外 LLM 调用做路由判断

---

## 六、工具注册机制

### 6.1 `@auto_tool` 装饰器（registry.py）

统一装饰器，所有工具只需定义一次，自动同时注册到 langchain 工具系统和路由注册表：

```python
from agent.tools.registry import auto_tool

@auto_tool(
    description="按金额或时间排序贪污记录。order_by 可选 amount_desc/amount_asc/date_desc/date_asc。",
    patterns=[r"最高", r"最低", r"最新", r"最[多高少]", r"排名前", r"第.*名"],
    extract_params="rank",
)
@tool
def rank_corruption_records(order_by: str = "amount_desc", top_n: int = 10) -> str:
    ...
```

`TOOL_REGISTRY` 全局单例在模块被 import 时即创建。`registry._tools` 列表按注册顺序存放元数据，`match(query)` 按顺序遍历匹配。

### 6.2 内置参数提取器

| 名称 | 提取参数 | 适用场景 |
|---|---|---|
| `person_name` | `person_name` | "XXX的判决"、"介绍一下XXX是谁" |
| `rank` | `order_by`, `top_n` | "金额最高前10名" |
| `year` | `year` | "2026年有哪些案件" |
| `keyword` | `keyword` | 通用关键词搜索 |
| `web_search` | `query` | 联网搜索（传完整 query） |
| `none` | `{}` | 无参数工具 |

### 6.3 `_extract_person_name` 原理

优先从正则模式（"XXX的判决"、"介绍一下XXX"）提取，再用 `_looks_like_name` 判断是否像人名。兜底策略：从句尾取连续 2~8 个中文字符作为人名（解决"介绍一下田伟"之类的问题）。

---

## 七、数据格式

### 7.1 贪污记录 CSV

| 字段 | 说明 |
|---|---|
| 案件序号 | 从1开始的序号 |
| 当事人姓名 | 可含"某某"脱敏 |
| 职务/身份 | 含省份/单位/职务级别 |
| 涉案金额 | 复杂格式：`8993万元；受贿6.13亿余元` → 需 `_parse_amount` 解析 |
| 主要犯罪事实 | 分号分隔的多条事实 |
| 判决/处理结果 | 判决结果文字 |
| 通报/宣判时间 | `YYYY-MM-DD` 格式 |
| 备注 | 典型案例、双开通报等 |

### 7.2 金额解析规则（`_parse_amount`）

- 优先提取"亿"单位的金额（亿元 × 10000 = 万元）
- 取所有候选中的**最大值**
- "未公开"/"巨额" → 返回 -1，排在最后
- 新增金额格式时，在正则匹配处补充即可

---

## 八、Streamlit UI 关键实现（app.py）

### 8.1 session_state 结构

```python
st.session_state["chat_history"]    # 历史消息列表
st.session_state["_pending_query"]   # str | None（快捷按钮触发）
st.session_state["_report_mode"]     # bool（报告生成模式，必须用 key 固定）
```

### 8.2 思考过程持久化

**问题**：思考过程在 rerun 后消失

**原因**：Streamlit 脚本从上到下顺序执行，`st.rerun()` 会跳过当前脚本继续执行，导致 `_handle_query` 未被调用

**解决方案**：
1. 历史存在 `session_state["chat_history"]` 里，每条记录结构：
   ```python
   {"role": "user/assistant", "content": str, "thinking": list, "time": str}
   ```
2. 每次脚本执行时先渲染历史
3. 用户输入 → 直接调用 `_handle_query()` → 不 rerun
4. 快捷问题 → 也直接调用 `_handle_query()` → 不 rerun

**教训**：Streamlit 里慎用 `st.rerun()`，它会重置整个脚本执行上下文，导致函数调用链断裂。

### 8.3 快捷按钮

必须用 `on_click` 回调设置 `_pending_query`，不能用按钮点击后手动修改 session_state（新版本 Streamlit 禁止在 widget 创建后修改其 key）：

```python
st.button(
    ex,
    key=f"ex_{ex}",
    on_click=lambda q=ex: st.session_state.update({"_pending_query": q}),
)
```

### 8.4 sidebar 变量捕获

sidebar 中的 widget 变量（`use_stream` 等）在 rerun 时行为不稳定。在 `_handle_query` 开头显式 capture 当前值：

```python
def _handle_query(query: str):
    _use_stream = use_stream           # ✓ 在函数开头 capture
    _report_mode = st.session_state.get("_report_mode", False)
```

### 8.5 流式输出

- 流式：`result.stream_answer()` 返回 generator，逐 token yield，`st.empty()` 占位动态刷新
- 非流式：直接取 `result.answer` 完整字符串

---

## 九、已踩过的坑

### 坑1：Python 路径
- **现象**：`ModuleNotFoundError` (langchain_core 等)
- **原因**：系统默认 python 非 3.12，导致找不到已安装的包
- **解决**：统一使用 `F:\Python312\python.exe`

### 坑2：LLM Tool Calling 不生效
- **现象**：`bind_tools` 后模型直接生成回答，不调用工具
- **原因**：通义千问对 tool calling 支持不完整
- **解决**：放弃 LLM 路由，改用正则规则匹配

### 坑3：StreamingMiddleware 的 transform_stream 返回 generator
- **现象**：拼接 `generator + str` 报错
- **原因**：`mw.transform_stream(chunk)` 返回 generator，被当作字符串拼接
- **解决**：`MiddlewareManager.apply_stream` 用 `yield from` 展平嵌套 generator

### 坑4：chain.stream() 返回 TextAccessor 而非 str
- **现象**：流式输出全是对象地址
- **原因**：`langchain_core.runnables` 的 `stream()` yields `TextAccessor` 对象
- **解决**：`if hasattr(event, "content"): token = event.content`

### 坑5：Router 里 `*TOOLS` 展开字典
- **现象**：`AttributeError: 'str' object has no attribute 'name'`
- **原因**：`TOOLS` 是 dict，`*TOOLS` 展开的是 key（字符串），不是 value（函数）
- **解决**：直接引用函数对象，不用解包

### 坑6：st.chat_message 嵌套 st.expander 渲染异常
- **现象**：expander 折叠状态无法正确保存
- **原因**：Streamlit 内部 widget 状态管理问题
- **解决**：移除嵌套，用普通 `st.container` + `st.markdown(html, unsafe_allow_html=True)` 替代

### 坑7：金额解析正则顺序
- **现象**：`"贪污8993万元；受贿6.13亿余元"` 解析成 61300 而非 8993
- **原因**：正则 `re.search` 只匹配第一个候选，应先找亿再找万
- **解决**：用 `re.finditer` 找所有候选，统一归一化到万元后取最大值

### 坑8：`sort_key` 函数定义在 for 循环内
- **现象**：`UnboundLocalError: local variable referenced before assignment`
- **原因**：Python 函数定义时即捕获外层变量，非运行时绑定
- **解决**：在 `_sort_records` 外层定义 `sort_key`，或用 lambda

### 坑9：prompt_loader.py 的 load_system_prompts / load_report_prompts 无外部调用
- **现象**：导入时报错（或其他间接问题）
- **原因**：这两个函数只被 `__main__` 调用，是死代码
- **解决**：删除，只保留被 `rag_factory.py` 调用的 `load_rag_prompts`

### 坑10：web_search 返回 0 条结果（DuckDuckGo HTML 接口失效）
- **现象**：`"谭瑞松案最新进展如何？"` 联网搜索返回 0 条
- **原因**：DuckDuckGo HTML endpoint 改为需要 POST + 复杂 bot 检测
- **解决**：安装 `ddgs` 包（`pip install ddgs`），用官方 Python SDK：
  ```python
  from duckduckgo_search import DDGS
  with DDGS() as ddg:
      for r in ddg.text(query, max_results=5):
          results.append(...)
  ```

### 坑11：快捷消息按钮导致输入框消失
- **现象**：点击快捷问题按钮后，聊天输入框不见了
- **原因**：Streamlit 新版本禁止在 widget 创建后修改 session_state 中的 widget key
- **解决**：用 `on_click` 回调替代手动 `st.session_state[key] = value`：
  ```python
  st.button(ex, key=f"ex_{ex}",
      on_click=lambda q=ex: st.session_state.update({"_pending_query": q}))
  ```

### 坑12："介绍一下田伟" 路由到 RAG 而非人名查询
- **现象**："介绍" 优先匹配 RAG 工具的正则
- **原因**：人名查询工具的 patterns 不够全面，且 `_extract_person_name` 用贪婪匹配
- **解决**：
  1. 在 `data_query_tool.py` 添加 patterns：`r"介绍.+"`, `r"说说.+"`, `r".+?([\u4e00-\u9fa5]{2,8})是谁"`
  2. 在 `registry.py` 的 `_extract_person_name` 中用非贪婪匹配 `+?` 并加句尾兜底提取

### 坑13：llm_router.py 被 linter 破坏
- **现象**：linter 自动将 Python 字符串中的英文引号 `"` 改成中文全角引号 `"`，导致 SyntaxError
- **原因**：工具描述字符串中包含中文双引号 `"..."`，被 linter 误识别为引号边界
- **解决**：工具描述统一用单行，去掉内部引号；当前 `llm_router.py` 已从 `router.py` 导入中被注释掉（禁用）

### 坑14：`auto_enrich_and_save` 搜索到无关内容也写入 DB
- **现象**：查询不在 DB 中的人（如"陈凯"），工具联网搜索后，即便搜索结果与该人完全无关（阮永的音乐考试通知），也写入 DB，且提取了错误的贪腐信息
- **原因**：`auto_enrich_and_save` 只检查提取结果是否为 None，但不验证人名是否出现在原始搜索文本中
- **解决**：在 `data_query_tool.py` 的 `auto_enrich_and_save` 中，Step 4 增加校验：提取信息后，检查 `person_name` 是否出现在 `title` 或 `snippet` 中，若不在则返回"未找到相关贪腐记录"，**不写入 DB**

### 坑15：`migrate_csv_to_mysql.py` 不是日常运行脚本
- **现象**：误以为该脚本是日常启动流程的一部分
- **原因**：`data_query_tool.py` 已内置 MySQL 优先 + CSV 降级逻辑，无需手动切换
- **解决**：`migrate_csv_to_mysql.py` 是一次性迁移工具，首次配置 MySQL 时用一次，之后数据读写由 `data_query_tool` 自动管理

---

## 十、扩展指南

### 新增数据查询工具

1. 在 `agent/tools/data_query_tool.py` 定义 `@tool` 函数
2. 在文件末尾 `__all__` 列表加入函数名（字符串）
3. 在 `agent/tools/router.py` 的 `ROUTING_RULES` 加入匹配规则
4. 在 `_build_tool_result` 加入 `tool_key → tool_args` 映射
5. 在 `TOOLS` 字典加入映射

### 新增工具模块

1. 在 `agent/tools/` 新建 `xxx_tool.py`
2. 定义 `@tool` 函数
3. 在 `agent_factory.py` 顶部 import 并加入 `ALL_AGENT_TOOLS` 列表

### 新增中间件

1. 继承 `agent/middleware/base.py` 的 `BaseMiddleware`
2. 重写 `transform_input` / `transform_output` / `transform_stream`
3. `AgentFactory.create(middleware_manager=MiddlewareManager().add(MyMiddleware()))`

### 扩展贪污数据

1. 编辑 `data/贪污记录.csv`，格式保持一致
2. 金额字段格式：`X万元`、`X亿余元`、`X亿X万元` 均可被 `_parse_amount` 正确解析
3. "未公开"/"巨额"会排在排序最后

---

## 十一、相关文档

- 提示词模板：`prompts/main_prompt.txt`、`prompts/rag_summarize.txt`、`prompts/report_prompt.txt`
- RAG 组件文档：直接阅读 `rag/rag_service.py`（职责单一，无隐藏依赖）
- 工具文档：阅读 `agent/tools/data_query_tool.py` 和 `agent/tools/web_search_tool.py`
