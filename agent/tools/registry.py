"""
Tool Registry — 工具注册中心。

提供 @auto_tool 装饰器：
  - 自动从 description 中提取工具名、参数 schema
  - 自动注册到全局 TOOL_REGISTRY
  - 自动生成路由规则（ROUTE_RULES）和参数提取逻辑

使用方式：
  from agent.tools.registry import auto_tool, TOOL_REGISTRY, ROUTING_RULES

  @auto_tool(description="按金额或时间排序，返回贪污记录。")
  def rank_corruption_records(order_by: str = "amount_desc", top_n: int = 10) -> str:
      ...

所有 @auto_tool 装饰的函数，在模块被 import 时自动注册。
router.py 和 agent_factory.py 只需 import registry 即可获取全部工具。
"""

import re
import inspect
from dataclasses import dataclass, field
from typing import Callable, Any


# ─────────────────────────────────────────────────────────────────────────────
# 标准参数提取器（供装饰器引用）
# ─────────────────────────────────────────────────────────────────────────────

def _extract_person_name(query: str, **kwargs) -> dict:
    """
    从 query 中提取人名。

    支持格式：
      - "XXX的判决" / "XXX涉案" → 提取 XXX
      - "介绍一下XXX" / "XXX是谁" → 提取句尾的中文片段
    """
    # 用非贪婪匹配（+?），避免"介绍一下田伟"被匹配成"介绍一下田"
    patterns = [
        (r"([\u4e00-\u9fa5]+?)的判决",),
        (r"([\u4e00-\u9fa5]+?)涉案",),
        (r"([\u4e00-\u9fa5]+?)的情况",),
        (r"([\u4e00-\u9fa5]+?)的(案件|犯罪|贪腐|贪污|腐败)",),
        (r"关于([\u4e00-\u9fa5]+?)的",),
        (r"查一下([\u4e00-\u9fa5]+?)",),
        (r"([\u4e00-\u9fa5]+?)是什么罪",),
        (r"介绍.+?([\u4e00-\u9fa5]{2,8})$",),
        (r"说说.+?([\u4e00-\u9fa5]{2,8})$",),
        (r".+?([\u4e00-\u9fa5]{2,8})是谁",),
    ]
    for item in patterns:
        p = item[0]
        m = re.search(p, query, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name and _looks_like_name(name):
                return {"person_name": name}

    # 句尾人名格式：介绍一下XXX、XXX是谁、说说你对XXX的看法
    # 提取末尾连续的中文片段（2~8字）作为人名
    chinese_seqs = re.findall(
        r"[\u4e00-\u9fa5]{2,8}(?=[^a-zA-Z0-9\u4e00-\u9fa5]|$)", query
    )
    if chinese_seqs:
        name = chinese_seqs[-1].strip()
        if name and _looks_like_name(name):
            return {"person_name": name}

    # 没有匹配到有效人名
    return {"person_name": query.strip()}


# 人名特征词：职务称谓、常见姓氏
_NAME_INDICATORS = {
    "书", "记", "长", "局", "处", "厅", "部", "委", "总",
    "队", "院", "校", "科", "股", "组", "官", "员",
    # 常见姓氏
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王", "冯", "陈",
    "楚", "卫", "蒋", "沈", "韩", "杨", "朱", "秦", "尤", "许",
    "何", "吕", "施", "张", "孔", "曹", "严", "华", "金", "魏",
    "陶", "姜", "戚", "谢", "邹", "喻", "柏", "水", "窦", "章",
    "云", "苏", "潘", "葛", "奚", "范", "彭", "郎", "鲁", "韦",
    "昌", "马", "苗", "凤", "花", "方", "俞", "任", "袁", "柳",
    "史", "唐", "雷", "贺", "倪", "汤", "滕", "殷", "罗", "毕",
    "郝", "邬", "安", "常", "乐", "于", "时", "傅", "皮", "卞",
    "齐", "康", "伍", "余", "元", "顾", "孟", "平", "黄", "和",
    "穆", "萧", "尹", "姚", "邵", "湛", "汪", "祁", "毛", "禹",
    "狄", "米", "贝", "明", "臧", "计", "伏", "成", "戴", "谈",
    "宋", "茅", "庞", "熊", "纪", "舒", "屈", "项", "祝", "董",
    "梁", "杜", "阮", "蓝", "闵", "席", "季", "麻", "强", "贾",
    "路", "娄", "危", "江", "童", "颜", "郭", "梅", "盛", "林",
    "刁", "钟", "徐", "邱", "骆", "高", "夏", "蔡", "田", "樊",
    "胡", "凌", "霍", "虞", "万", "支", "柯", "昝", "管", "卢",
    "莫", "经", "房", "裘", "缪", "干", "解", "应", "宗", "丁",
    "宣", "贲", "邓", "郁", "单", "杭", "洪", "包", "诸", "左",
    "石", "崔", "吉", "钮", "龚", "程", "嵇", "邢", "滑", "裴",
    "陆", "荣", "翁", "荀", "羊", "於", "惠", "甄", "麴", "家",
    "封", "芮", "羿", "储", "靳", "汲", "邴", "糜", "松", "井",
    "段", "富", "巫", "乌", "焦", "巴", "弓", "牧", "隗", "山",
    "谷", "车", "侯", "宓", "蓬", "全", "郗", "班", "仰", "秋",
    "仲", "伊", "宫", "宁", "仇", "栾", "暴", "甘", "钭", "厉",
    "戎", "祖", "武", "符", "刘", "景", "詹", "束", "龙", "叶",
    "幸", "司", "韶", "郜", "黎", "蓟", "薄", "印", "宿", "白",
    "怀", "蒲", "邰", "从", "鄂", "索", "咸", "籍", "赖", "卓",
    "蔺", "屠", "蒙", "池", "乔", "阴", "郁", "胥", "能", "苍",
    "双", "闻", "莘", "党", "翟", "谭", "贡", "劳", "逄", "姬",
    "申", "扶", "堵", "冉", "宰", "郦", "雍", "却", "璩", "桑",
    "桂", "濮", "牛", "寿", "通", "边", "扈", "燕", "冀", "郏",
    "浦", "尚", "农", "温", "别", "庄", "晏", "柴", "瞿", "阎",
    "充", "慕", "连", "茹", "习", "宦", "艾", "鱼", "容", "向",
    "古", "易", "廖", "庾", "终", "暨", "居", "衡", "步", "都",
    "耿", "满", "弘", "匡", "国", "文", "寇", "广", "禄", "阙",
    "东", "欧", "殳", "沃", "利", "蔚", "越", "夔", "隆", "师",
    "巩", "厍", "聂", "晁", "勾", "敖", "融", "谌", "訾", "承",
    "亓", "佘", "佗", "伽", "仉", "迮",
    # 叠字人名常见字
    "伟", "芳", "军", "杰", "涛", "明", "超", "勇", "艳", "丽",
    "静", "敏", "静", "玲", "华", "红", "霞", "平", "刚", "强",
    "磊", "洋", "勇", "艳", "欢", "乐", "兴", "龙", "凤", "玉",
}

# 描述性短语特征词（不是人名）
_DESCRIPTIVE_WORDS = {
    "最近", "最新", "最大", "最小", "最高", "最低", "最多", "最少",
    "重要", "重大", "典型", "重点", "热门", "著名", "主要",
    "相关", "所有", "全部", "各种", "各个",
    "一些", "部分", "全部", "整个",
    "贪污", "受贿", "腐败", "违法", "犯罪",
    "的情况", "的案件", "的结果", "的内容",
    "一下", "帮我", "请问", "我想", "你能", "可以",
}

def _looks_like_name(text: str) -> bool:
    """判断 text 是否像人名（而非描述性短语）"""
    # 长度不超过 6 个字
    if len(text) > 6:
        return False
    # 包含职位/姓氏特征
    if any(c in _NAME_INDICATORS for c in text):
        return True
    # 2字名：常见姓氏 + 单字
    if len(text) == 2 and text[0] in _NAME_INDICATORS:
        return True
    # 3字名：常见叠字
    if len(text) == 3 and text[1] in _NAME_INDICATORS and text[2] in _NAME_INDICATORS:
        return True
    # 完全不是描述性短语
    if text not in _DESCRIPTIVE_WORDS and not any(w in text for w in ["的", "案", "件"]):
        return True
    return False


def _extract_rank(query: str, **kwargs) -> dict:
    """从 query 中提取排序参数（order_by, top_n）"""
    order = "amount_desc"
    top_n = 10
    num_match = re.search(r"前?([0-9]+)", query)
    if num_match:
        top_n = int(num_match.group(1))
    if re.search(r"最低|最少|最旧|从小", query, re.IGNORECASE):
        order = "amount_asc" if ("金额" in query or "涉案" in query) else "date_asc"
    elif re.search(r"最新|最近", query, re.IGNORECASE):
        order = "date_desc"
    return {"order_by": order, "top_n": top_n}


def _extract_year(query: str, **kwargs) -> dict:
    """从 query 中提取年份（"2026年有哪些案件" → year=2026）"""
    year_match = re.search(r"(\d{4})年", query)
    year = year_match.group(1) if year_match else ""
    return {"year": year}


def _extract_keyword(query: str, **kwargs) -> dict:
    """从 query 中提取关键词（去除语气词）"""
    keyword = re.sub(r"[的吗呀？?，。！!]", "", query).strip()
    return {"keyword": keyword}


def _extract_web_search(query: str, **kwargs) -> dict:
    return {"query": query}


def _extract_web_fetch(query: str, **kwargs) -> dict:
    return {"query": query}


def _extract_no_param(query: str, **kwargs) -> dict:
    return {}


# 内置提取器映射（装饰器通过字符串名引用）
BUILTIN_EXTRACTORS: dict[str, Callable] = {
    "person_name":  _extract_person_name,
    "rank":         _extract_rank,
    "year":         _extract_year,
    "keyword":      _extract_keyword,
    "web_search":   _extract_web_search,
    "web_fetch":    _extract_web_fetch,
    "none":         _extract_no_param,
}


# ─────────────────────────────────────────────────────────────────────────────
# ToolMeta — 单个工具的元数据
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolMeta:
    """
    工具元数据，@auto_tool 装饰时自动填充。
    """
    func: Callable                       # 原始函数（含 @tool 装饰后的属性）
    name: str                             # 工具名（从 func.name 或 description 首行推断）
    description: str                      # 工具描述（来自装饰器参数）
    patterns: list[str]                   # 触发该工具的正则表达式列表
    extract_params: Callable[..., dict]   # 参数提取函数：(query, **extra) -> dict
    re_flags: int = re.IGNORECASE        # 正则匹配标志

    def match(self, query: str) -> bool:
        """判断 query 是否匹配该工具"""
        for p in self.patterns:
            if re.search(p, query, self.re_flags):
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ToolRegistry — 全局注册表
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    全局工具注册表。

    - _tools: list[ToolMeta]   按注册顺序存放
    - _func_map: dict[str, Callable]   name → 函数对象（供 agent_factory.bind_tools 使用）
    - _meta_map:  dict[str, ToolMeta]  name → 元数据（供 router 使用）
    """

    def __init__(self):
        self._tools: list[ToolMeta] = []
        self._func_map: dict[str, Callable] = {}
        self._meta_map: dict[str, ToolMeta] = {}

    def register(
        self,
        func: Callable,
        *,
        description: str,
        patterns: list[str],
        extract_params: Callable[..., dict] | str = "none",
        re_flags: int = re.IGNORECASE,
    ) -> Callable:
        """
        将工具注册到注册表。

        Args:
            func:          @tool 装饰后的函数
            description:   工具描述（必须）
            patterns:      触发该工具的正则表达式列表
            extract_params: 参数提取函数，或内置提取器名称（str）
            re_flags:      正则标志，默认 re.IGNORECASE
        """
        # 解析 extract_params（支持字符串名或直接传函数）
        if isinstance(extract_params, str):
            ep_name = extract_params.lower()
            if ep_name not in BUILTIN_EXTRACTORS:
                raise ValueError(
                    f"未知的内置提取器 '{ep_name}'，"
                    f"可用：{list(BUILTIN_EXTRACTORS.keys())}"
                )
            extractor = BUILTIN_EXTRACTORS[ep_name]
        else:
            extractor = extract_params

        # 工具名：从 func.name（@tool 装饰器设置的）获取
        tool_name = getattr(func, "name", None) or func.__name__

        # 防止重复注册（同名工具直接覆盖，保持幂等性）
        if tool_name in self._func_map:
            self._func_map[tool_name] = func
            self._meta_map[tool_name] = ToolMeta(
                func=func, name=tool_name, description=description,
                patterns=patterns, extract_params=extractor, re_flags=re_flags,
            )
            # 更新列表中的对应条目
            for i, m in enumerate(self._tools):
                if m.name == tool_name:
                    self._tools[i] = self._meta_map[tool_name]
                    break
            return func

        meta = ToolMeta(
            func=func,
            name=tool_name,
            description=description,
            patterns=patterns,
            extract_params=extractor,
            re_flags=re_flags,
        )

        self._tools.append(meta)
        self._func_map[tool_name] = func
        self._meta_map[tool_name] = meta

        return func

    @property
    def tools(self) -> list[Callable]:
        """返回所有工具函数（供 agent_factory.bind_tools 使用）"""
        return list(self._func_map.values())

    @property
    def tool_map(self) -> dict[str, Callable]:
        """name → 函数对象"""
        return self._func_map

    @property
    def routing_rules(self) -> list[dict]:
        """
        返回路由规则列表（供 router._match_rule 使用）。
        格式与原有 ROUTING_RULES 兼容。
        """
        return [
            {
                "name":      meta.name,
                "tool_key":  meta.name,   # tool_key 即 name，保持与 router._build_tool_result 兼容
                "patterns":  meta.patterns,
                "re_flags":  meta.re_flags,
            }
            for meta in self._tools
        ]

    def match(self, query: str) -> ToolMeta | None:
        """根据 query 返回第一个匹配的工具元数据"""
        for meta in self._tools:
            if meta.match(query):
                return meta
        return None

    def get_meta(self, name: str) -> ToolMeta | None:
        return self._meta_map.get(name)


# ─────────────────────────────────────────────────────────────────────────────
# @auto_tool 装饰器
# ─────────────────────────────────────────────────────────────────────────────

def auto_tool(
    *,
    description: str,
    patterns: list[str],
    extract_params: Callable[..., dict] | str = "none",
    re_flags: int = re.IGNORECASE,
) -> Callable[[Callable], Callable]:
    """
    工具注册装饰器。

    用法：
      @auto_tool(
          description="按金额或时间排序贪污记录。order_by 可选 amount_desc/amount_asc/date_desc/date_asc。",
          patterns=[r"最高", r"最低", r"最新", r"最[多高少]", r"排名前", r"第.*名"],
          extract_params="rank",
      )
      @tool
      def rank_corruption_records(order_by: str = "amount_desc", top_n: int = 10) -> str:
          ...

    效果：
      1. 函数被 @tool 装饰（底层 langchain 工具）
      2. 函数被注册到 TOOL_REGISTRY（供路由和 agent 使用）
      3. 无需在 __all__ / ALL_AGENT_TOOLS / TOOLS 中重复注册
    """
    def decorator(func: Callable) -> Callable:
        # 先执行底层的 @tool 装饰（langchain_core 的工具装饰器）
        from langchain_core.tools import tool as _langchain_tool
        decorated = _langchain_tool(description=description)(func)

        # 注册到全局注册表
        TOOL_REGISTRY.register(
            decorated,
            description=description,
            patterns=patterns,
            extract_params=extract_params,
            re_flags=re_flags,
        )

        return decorated

    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# 全局单例（模块被 import 时即创建）
# ─────────────────────────────────────────────────────────────────────────────

TOOL_REGISTRY = ToolRegistry()

# 导出便捷访问
ROUTING_RULES = TOOL_REGISTRY.routing_rules  # 供 router 直接引用
