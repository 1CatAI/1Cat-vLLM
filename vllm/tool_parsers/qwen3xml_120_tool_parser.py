from vllm.envs import VLLM_QWEN3X_TOOL_FIX
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import json
from collections.abc import Sequence
from typing import Any
from xml.parsers.expat import ParserCreate

import regex as re
from xgrammar import Grammar
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
)

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.logger import init_logger
from vllm.sampling_params import (
    StructuredOutputsParams,
)
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.abstract_tool_parser import (
    Tool,
    ToolParser,
)
from vllm.tool_parsers.qwen3_120_utils import (
    TOOL_LIKE_TAG_RE,
    find_common_prefix,
    find_tool_properties,
    missing_required_args as _u_missing_required_args,
    opt22d_should_force_all as _u_opt22d_should_force_all,
    registered_tool_names as _u_registered_tool_names,
    request_forbids_tool_calls,
    split_incomplete_tool_tag_tail,
)
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)
from vllm.tool_parsers.qwen3_120_utils import find_common_prefix, find_tool_properties

logger = init_logger(__name__)

# opt22d: 预编译“形似工具调用”标签匹配（模块级；含无右尖括号的截断残片分支）
# [opt22 去重 2026-10-14] 正则收敛到 utils.TOOL_LIKE_TAG_RE 单一来源；别名保持引用点不变。
_TOOL_LIKE_TAG_RE = TOOL_LIKE_TAG_RE

# opt22d 第十二轮（缺陷B）：参数值**内部**的工具类标签（开闭都含）。
# 体内标签都是**文档内容**（模型写说明文档时原样写出示例），必须让 expat 视作文本。
_PARAM_BODY_TAG_RE = re.compile(
    r"</?function[^<>]*>?|</?parameter[^<>]*>?|</?tool_?call>|</?tool\b[^<>]*>?"
)
# 收尾标签（三方言归一化后）——参数含示例时这些要"扣住"而非直接喂 expat
_CLOSE_TAGS = frozenset({"</parameter>", "</function>", "</tool_call>", "</toolcall>", "</tool>"})

# 末尾真收尾标签序列的**回溯剥除**（v4 保守转义会把它们当内容，finish 时清理）。
# 注：参数增量里的换行是 JSON 转义序列（字面 `\\` + `n`），不是空白字符。
_TOOL_CLOSE_RE = re.compile(r"</tool_?call\b[^<>]*>")
_TRAILING_JSON_TAIL_RE = re.compile(r"(?:\\n|\s|\"|\})*")


def strip_trailing_struct_close(text: str) -> str:
    """剥掉参数值末尾的**真收尾标签序列**（`</parameter></function></tool_call>`）。

    opt22d 第十二轮（缺陷B）：v4 规则在「参数含文档示例」时保守地把所有工具标签
    当内容，连**真收尾**也一并转义进了参数值。本函数在 finish 出口把它剥掉。

    判定：最后一个包裹层闭合标签（`</tool_call>` 三方言）之后**只剩 JSON 收尾字符**
    （`"` / `}` / 空白 / 字面 `\n`）⇒ 它是真收尾；再从该处**回溯**依次剥掉紧邻的
    `</function>` 与 `</parameter>`。文档正文里紧跟 `"}` 的示例序列不存在，故不误伤。
    """
    last = None
    for m in _TOOL_CLOSE_RE.finditer(text):
        last = m
    if last is None:
        return text
    tail = text[last.end():]
    if not _TRAILING_JSON_TAIL_RE.fullmatch(tail):
        return text
    head = text[: last.start()]
    for pat in (
        r"(?:\\n|\s)*</function>(?:\\n|\s)*$",
        r"(?:\\n|\s)*</parameter>(?:\\n|\s)*$",
    ):
        head = re.sub(pat, "", head, count=1)
    return head + tail


# 参数体内出现即说明"本参数含文档示例"的开标签（真调用里不会有嵌套的 function/tool_call）
_PARAM_EXAMPLE_OPEN_RE = re.compile(r"<function=|</?tool_?call>|<parameter=")
# 仅匹配 `<parameter=…>` 开标签（区别于闭合的 `</parameter>`）。
_PARAM_OPEN_ONLY_RE = re.compile(r"<parameter=[^<>]*>?")
# 包裹层闭合标签的三种方言（踩坑：初版只认 `</tool_` 带下划线形态，把
# `<toolcall>`（Anthropic 方言，模板教示形态）的真闭合误判为文档示例 ⇒
# 参数永不闭合、把 `</parameter></function></toolcall>` 全吞进参数值）。
_WRAP_CLOSE_RE = re.compile(r"</tool_?call\b|</tool\b")


def _at_or_prefix(s: str, token: str) -> bool:
    """``s`` 是 ``token`` 的完整匹配或**残缺前缀**（流式下判据可能未到齐）。"""
    return s.startswith(token) or token.startswith(s)


def _is_real_param_close(peek: str) -> bool:
    """opt22d 第十二轮（缺陷B）：参数体内 ``</parameter>`` 是否为**真闭合**。

    **判定只看 "收尾形态"，不看其后的正文。** 这是踩了三次坑后的结论：

    - ❌ 「看 `</tool_call>` 之后是不是普通正文」——真调用之后**也**跟正文
      （模型调用完继续解释），线上实测把最常见的形态判坏：一个 Write 调用的
      `content` 吞掉了 `</parameter></function></tool_call>` 及其后所有分析文字。
    - ❌ 「开闭配对计数」——被模型必写的「错误示例：`<parameter=command>` 没有
      闭合」击穿（只有开标签 ⇒ 计数永久失衡）。
    - ✅ 「看 `</function>` 是否**紧跟包裹层闭合**」——真调用的收尾必然是
      ``</parameter> </function> </tool_call>`` 这一串；而模型写的示例**大多
      省略 `</tool_call>`**（实测 EXH[08] 就是）。

    规则（``peek`` = 该 ``</parameter>`` 之后已到达的文本）：

    - 空 ⇒ 判据未到齐 ⇒ 按真闭合放行（否则参数永不闭合）
    - ``<parameter=…`` ⇒ 后面还有参数 ⇒ 真
    - ``</function>`` ⇒ 再看其**紧后**是否为包裹层闭合（三方言）⇒ 真
    - 以上 token 的**残缺前缀** ⇒ 判据未到齐 ⇒ 放行
    - 其它 ⇒ 文档示例 ⇒ 转义

    已知弱点（记录在案）：模型写出**自带 ``</tool_call>`` 的示例**（罕见）会被
    误判为真闭合，代价是参数被截断、无幽灵调用（安全侧）。
    """
    s = peek.lstrip()
    if not s:
        return True
    if _at_or_prefix(s, "<parameter="):
        return True
    if not _at_or_prefix(s, "</function>"):
        return False
    if not s.startswith("</function>"):
        return True                      # `</function>` 还没到齐 ⇒ 放行
    rest = s[len("</function>"):].lstrip()
    if not rest:
        return True                      # 文本到此结束 ⇒ 真收尾
    for _wrap in ("</tool_call>", "</toolcall>", "</tool>"):
        if _at_or_prefix(rest, _wrap):
            return True
    return False                         # 缺包裹层闭合 ⇒ 是文档示例


def escape_param_body_example_tags(
    chunk: str, peek: str = "", allow_pending: bool = True, state: dict | None = None
) -> tuple[str, bool]:
    """opt22d 第十二轮（缺陷B）：转义**参数体内部**的文档示例标签。

    真调用里 ``</function>`` / ``</tool_call>`` **必然出现在最后一个参数闭合之后**
    （参数体**外**）⇒ 参数体内的工具类标签一律是文档内容。

    **`</parameter>` 的判定分两级**（三级方案都被线上实测打回过，见下）：

    1. **本参数已被判定"含示例"**（此前出现过 ``<function=`` / ``<toolcall>`` 等
       开标签）⇒ 一律转义，**不再截断**。理由：模型写说明文档时会原样写出**完整**
       示例 ``<toolcall>…</parameter></function></tool_call>``，与真收尾**字节同形**，
       逐点判定不可能正确（实测线上 3/3 采样都被切成幽灵 Bash 调用）。
       牺牲增量显示、换取"内容完整 + 无幽灵调用"。
    2. 否则按 `_is_real_param_close` 的 peek 结构判定（真收尾的 `</function>`
       紧后必然是包裹层闭合）。

    **踩坑（三版推倒重来）**：
    - v1 「只看后跟 `</function>`+包裹闭合」→ 示例自带完整闭合 ⇒ 被击穿
    - v2 「开闭配对计数」→ 模型必写「错误示例：`<parameter=command>` 没有闭合」⇒ 计数失衡
    - v3 「看 `</tool_call>` 之后是不是正文」→ **真调用之后也跟正文**（模型调用完
      继续解释）⇒ 线上 P1 稳定 2 FAIL，content 吞掉收尾串及全部分析文字
    - v4（本版）「参数体内见过示例开标签 ⇒ 不再截断」——不依赖后文、不依赖计数

    Returns:
        ``(文本, 是否待定)``。**待定**表示 ``</parameter>`` 是流式下已到达的最后
        一个元素、peek 为空而无法判定，调用方应扣住它等下一元素。
    """
    if "<" not in chunk:
        return chunk, False
    if state is None:
        state = {}

    pending = False
    # 参数体内出现这些**开标签** ⇒ 本参数含文档示例（真调用里不会有嵌套）
    _has_example_open = _PARAM_EXAMPLE_OPEN_RE.search(chunk) is not None
    if _has_example_open:
        state["has_example"] = True
    _example = state.get("has_example", False)

    def repl(m: re.Match) -> str:
        nonlocal pending
        tag = m.group(0)
        if tag.startswith("</parameter>"):
            if _example:
                return _as_xml_text(tag)          # 已判定含示例 ⇒ 不截断
            if not peek.strip():
                if allow_pending:
                    pending = True
                    return ""
                return tag
            if _is_real_param_close(peek):
                return tag
        return _as_xml_text(tag)

    return _PARAM_BODY_TAG_RE.sub(repl, chunk), pending


def _unescape_body_example_tags(text: str) -> str:
    """还原 `_as_xml_text` 的转义（**仅供不经 expat 解码的 raw 缓冲出口使用**）。

    raw 缓冲路径把参数正文原样交给 `_end_element`，不经过 expat ⇒ 实体不会被
    解码。若不在出口还原，`&lt;function=Bash&gt;` 这类实体会字面留在参数值里。
    ⚠️ 仅对 raw 路径使用；expat 路径已由 expat 自动解码，再还原会二次误伤。
    """
    return text.replace("&lt;", "<").replace("&gt;", ">")


def _as_xml_text(tag: str) -> str:
    """把标签转成 expat 会**还原成原文字符**的实体形式。

    用 XML 实体（``&lt;``/``&gt;``）而非占位字符，是因为参数值最终由 expat
    解码后直接下发客户端，实体能被 expat 还原为 ``<``，**不会出现在输出里**。
    —— 这与本项目"显示用转义符号"（``⟨'…'⟩``）是两回事，``&lt;`` 在此只是
    喂给 expat 的 XML 转义，与 ``_escape_xml_special_chars()`` 同一性质。
    """
    return tag.replace("<", "&lt;").replace(">", "&gt;")

# opt22d 第十轮（xml 侧 L-D 的**结构条件**）：
#   配对区间 = <function=NAME>…</function>；区间**外**的 <function= 即“未闭合孤立标签”。
#   注：用非贪婪配对（DOTALL），多个 function 时逐个配对，不会跨调用误判。
_XML_FN_PAIR_RE = re.compile(r"<function=[A-Za-z0-9_.\-]*>.*?</function>", re.DOTALL)
_XML_FN_OPEN_RE = re.compile(r"<function=[A-Za-z0-9_.\-]*>?")


def _unclosed_fn_decision(text: str) -> tuple[int, str] | None:
    """opt22d 第十一轮（方案 D）：对最后一个**未闭合**的顶层 ``<function=`` 给出判定。

    Returns:
        ``(起始下标, 判定)``，判定 ∈ ``{"hold", "escape"}``；``None`` 表示无需干预
        （无未闭合标签，或已见判据 ⇒ 真实调用）。

    判定规则（全部基于**完整 current_text**，故流式也能看全上下文）：

    =========================  ======================================
    ``<function=X>`` 之后      判定
    =========================  ======================================
    见 ``<parameter=``/``</function>``   真实调用（``None``）
    仅空白                    ``hold`` —— 判据未到齐，暂存等下一轮
    其它非空白文本              ``escape`` —— 展示性文本，转义
    =========================  ======================================
    """
    if "<function=" not in text:
        return None
    paired = [(m.start(), m.end()) for m in _XML_FN_PAIR_RE.finditer(text)]
    last = None
    for m in _XML_FN_OPEN_RE.finditer(text):
        if any(s <= m.start() < e for s, e in paired):
            continue  # 落在配对区间内 → 真实调用
        last = m
    if last is None:
        return None
    after = text[last.end():]
    if "<parameter=" in after or "</function>" in after:
        return None  # 判据已到齐 → 真实调用
    # 判据**未到齐**：空白 / 残缺标记前缀（如 `<para`、`</fun`）都要继续等。
    # ⚠️ 不可把"残缺"当成"否决"——流式下标记会被切开，若在 `<para` 时就判 escape
    # 会把真实调用的参数标签误转义（实测：步长 20 时 REAL[09] 因此丢调用）。
    st = after.lstrip()
    if any(mk.startswith(st) for mk in ("<parameter=", "</function>")):
        return (last.start(), "hold")
    # 已是确定的其它文本 ⇒ 展示性（如 `<function=Write>` 后跟中文说明）
    return (last.start(), "escape")


def _esc_open_fn(m) -> str:
    """re.sub 替换函数：把匹配到的开标签转义为显示安全形式。"""
    return m.group(0).replace("<", "⟨'").replace(">", "'⟩")


def _hold_start(text: str) -> int | None:
    """opt22d 第十一轮（方案 D）：应从下标何处开始暂存；``None`` = 无需暂存。

    **关键约束**：必须在标签的**任何部分**被喂给 expat **之前**就扣住。
    expat 是流式解析器，喂进去的内容无法撤回——若先喂了 `<fu` 再想扣住，
    会造成 XML 流错乱（实测出现 ``not well-formed`` 且丢失后续调用）。
    故这里**连同残缺前缀一起扣**（`<`、`<f`、…、`<function`）。
    """
    if not text:
        return None
    lt = text.rfind("<")
    if lt >= 0 and "<function=".startswith(text[lt:]):
        return lt  # 可能是 <function= 的残缺前缀
    d = _unclosed_fn_decision(text)
    return d[0] if d is not None else None


def _escape_unclosed_function_tags(text: str, *, streaming: bool = False) -> str:
    """opt22d 第十轮：把**未闭合**的裸 ``<function=…>`` 转义（xml 侧 L-D 结构条件）。

    **背景**：xml 的 L-D（``_should_skip_element``）此前只要求**行首**，缺结构条件。
    实测 ``verify_r9`` 的 G2：正文里孤立输出一个 ``<function=Write>``（无 ``</function>``）
    会被当成真实调用执行（``calls=['Write']``，参数为空）——而它显然是展示性文本。
    按「只有有明显意图与特征的才算真调用」原则补上：``<function=`` 需**行首 + 有闭合**。

    **与 coder 的差异**：coder 是文本 diff、能看全文，可对"判据未到齐"的尾部**暂存**；
    xml 是 SAX 逐元素流式，``<function=`` 开元素到达时**无法预知**后续是否有 ``</function>``
    （那是未来事件）。故：

    - ``streaming=True``（流式）：仅对**确定未闭合**的做处理——不适用，直接返回原文；
    - ``streaming=False``（非流式 / 消息结束）：全文已知，可精确配对判定。

    Args:
        text: 待处理文本。
        streaming: 是否处于流式增量路径（流式下无法判定，保留原文）。
    """
    if streaming or not text or "<function=" not in text:
        return text
    paired = [(m.start(), m.end()) for m in _XML_FN_PAIR_RE.finditer(text)]
    if not paired:
        # 全文没有任何闭合对 ⇒ 所有 <function= 都是孤立标签
        return _XML_FN_OPEN_RE.sub(
            lambda m: m.group(0).replace("<", "⟨'").replace(">", "'⟩"), text
        )
    out: list[str] = []
    last = 0
    for m in _XML_FN_OPEN_RE.finditer(text):
        i = m.start()
        if any(s <= i < e for s, e in paired):
            continue  # 落在配对区间内 → 真实调用，保持原样
        out.append(text[last:i])
        out.append(m.group(0).replace("<", "⟨'").replace(">", "'⟩"))
        last = m.end()
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


class StreamingXMLToolCallParser:
    """
    Simplified streaming XML tool call parser
    Supports streaming input, parsing, and output
    """

    def __init__(self):
        self.reset_streaming_state()

        # Tool configuration information
        self.tools: list[Tool] | None = None
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.function_start_token: str = "<function="
        self.function_end_token: str = "</function>"
        self.parameter_start_token: str = "<parameter="
        self.parameter_end_token: str = "</parameter>"

    def reset_streaming_state(self):
        """Reset streaming parsing state"""

        self.deltas = []
        # state for streaming
        self.tool_call_index = 0
        self.current_call_id = None
        self.last_completed_call_id = None
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.current_param_value_converted = ""
        self.current_param_is_first = False
        self.should_emit_end_newline = False
        self.start_quote_emitted = False

        self.streaming_buffer = ""
        self.last_processed_pos = 0

        self.text_content_buffer = ""

        # opt22d: 防误触发（过识别）状态
        self._demote_current_call = False   # 未注册工具名 → 本次调用降级为文本
        self._opt22d_force_all = False      # opt22d A/B/C：整轮降级
        self._demote_note_emitted = False   # 降级文本已回吐（幂等）
        # opt22d（L-C）：markdown 代码围栏状态（仅顶层文本；跨元素/跨 chunk 保持）
        self._in_code_fence = False
        self._fence_line_start = True
        self._fence_run = 0
        # opt22d（L-D）：裸标签行首约束（仅顶层文本）
        self._at_line_start = True            # 下一个元素是否位于行首（仅空白前缀）
        self._pending_tool_call_wrapper = False  # 刚见过 <tool_call> 开标签
        # opt22d 第十二轮（缺陷A）：本次调用是否为“无 <tool_call> 包裹”的裸调用
        # （由 _start_element 自动补包裹而来）。裸调用没有 </tool_call> 可触发复位，
        # 故需在 </function> 时自行收尾，否则下一个裸 <function=> 会并进同一个调用。
        self._autofilled_call: bool = False
        # opt22d 第十一轮（方案 D）：暂存“未定”的 <function=> 开标签文本
        self._hold_buf: str = ""

        # 当前元素之后已到达的文本（供 `</parameter>` 真伪判定做前瞻）
        self._peek = ""
        # 是否流式（流式才允许"扣住 `</parameter>` 等下一元素"）
        self._streaming_mode = False
        # opt22e A（xml 侧）：**上下文判据**。模型会先给出真实调用、**再**在正文里
        # 写"格式完全正确"的示例 ⇒ 绕过 L-C（不在围栏内）与 L-D（在行首）被当成
        # 第二个真实调用；且其参数**非空** ⇒ 下游真的会执行（安全风险，非"空参数
        # 失败"这类轻后果）。真调用的多个调用**连续**输出、中间不夹正文，故可判。
        self._opt22e_seen_call = False    # 本次是否已产出过真实调用
        self._opt22e_seen_prose = False   # 其后顶层（调用之外）是否出现过正文
        # 当前参数内是否已出现文档示例开标签（跨 chunk 保持；参数结束/调用复位时清）
        self._param_example_state: dict = {"has_example": False}
        # 流式结束时是否需要强制收尾（参数含示例 ⇒ 收尾标签被当内容，expat 收不到闭合）
        self._needs_finish_flush = False
        # opt22d 第十二轮（缺陷B·滚动扣住）：参数含示例时，收尾标签**扣住**不喂
        # expat——等下一个"确证是内容"的元素到了再释放，否则 finish 时丢弃（真收尾）。
        # 这样**参数正文照常即时下发**，只延迟几十字节（v4 曾缓冲整个参数 ⇒ 大文件
        # 场景 20k-30k token 会 5 分钟无数据、客户端 120s 断链）。
        self._hold_tags: list[str] = []
        # 已扣住、待补发的 `</parameter>`
        self._pending_param_close = False
        # opt22d 第十二轮（缺陷B·续）：本 delta 中 expat **真的看到**了 function /
        # tool_call 的结束元素（被转义成文档内容的那些不算）。
        # `parse_single_streaming_chunks` 的「漏掉 end 事件」兜底检查的是**未转义的
        # 原文**，会把文档示例里被转义掉的 `</function>` 误当收尾 ⇒ 参数被提前闭合
        # （实测步长 20：content 在示例 `</function>` 后 2 字符处截断、Write 变
        # Write+幽灵 Bash）。兜底条件改为额外要求本标志置位。
        self._saw_fn_end = False
        self._saw_tc_end = False

        # state for preprocessing and deferred parsing
        self._pre_inside_parameter = False
        self._pre_param_buffer = ""
        self._pre_current_param_name = None
        self.defer_current_parameter = False
        self.deferred_param_raw_value = ""

        # recreate parser
        self.parser = ParserCreate()
        self.setup_parser()

    def parse_single_streaming_chunks(self, xml_chunk: str) -> DeltaMessage:
        """
        Parse single streaming XML chunk and return Delta response
        This is the actual streaming interface that receives chunks
        one by one and maintains internal state

        Args:
            xml_chunk: Single XML chunk string
        Returns:
            DeltaMessage: Contains delta information generated by this chunk,
            returns empty response if no complete elements
        """
        # Record delta count before processing
        initial_delta_count = len(self.deltas)
        # opt22d 第十二轮（缺陷B·续）：兜底判定只看 expat **真的看到**的结束元素
        self._saw_fn_end = False
        self._saw_tc_end = False

        self.streaming_buffer += xml_chunk

        # A chunk may close one call and open another. Closing tags in that
        # chunk must never make the fallback close the newly opened call.
        initial_call_id = self.current_call_id
        found_elements = self._process_complete_xml_elements()

        if found_elements:
            # If complete elements found, check if end events were missed
            # some tags may not have been triggered
            try:
                new_deltas = self.deltas[initial_delta_count:]
                # If this chunk contains </function>
                # but didn't generate '}', then complete it
                if (
                    self.current_call_id is not None
                    and self.current_call_id == initial_call_id
                    and self.current_function_open
                    and self.function_end_token in xml_chunk
                    and self._saw_fn_end
                ):
                    # - Added '}' (non-empty parameter ending)
                    # - Added '{}' (empty parameter function)
                    has_function_close = any(
                        (
                            td.tool_calls
                            and any(
                                (
                                    tc.function
                                    and tc.id == self.current_call_id
                                    and isinstance(tc.function.arguments, str)
                                    and (tc.function.arguments in ("}", "{}"))
                                )
                                for tc in td.tool_calls
                            )
                        )
                        for td in new_deltas
                    )
                    if not has_function_close:
                        # Close potentially unclosed element
                        if self.current_param_name:
                            self._end_element("parameter")
                        if self.current_function_name:
                            self._end_element("function")
                # If this chunk contains </tool_call>
                # but didn't generate final empty delta, then complete it
                if (
                    self.current_call_id is not None
                    and self.current_call_id == initial_call_id
                    and self.tool_call_end_token in xml_chunk
                    and self._saw_tc_end
                ):
                    has_toolcall_close = any(
                        (
                            td.tool_calls
                            and any(
                                (
                                    tc.type == "function"
                                    and tc.function
                                    and tc.function.arguments == ""
                                    and tc.id == self.current_call_id
                                )
                                for tc in td.tool_calls
                            )
                        )
                        for td in new_deltas
                    )
                    if not has_toolcall_close:
                        # Close potentially unclosed element
                        if self.current_param_name:
                            self._end_element("parameter")
                        if self.current_function_name:
                            self._end_element("function")
                        self._end_element("tool_call")
            except Exception as e:
                logger.warning("Error with fallback parsing: %s", e)
            # Merge newly generated deltas into single response
            result_delta = self._merge_new_deltas_to_single_response(
                initial_delta_count
            )
            return result_delta
        else:
            # No complete elements, check if there's unoutput text content
            if self.text_content_buffer and self.tool_call_index == 0:
                # Has text content but no tool_call yet, output text content
                # opt22d（L-E）：转义在冲刷时统一施加；未闭合尾部留在缓冲里
                drained = self._drain_text_buffer()
                if drained:
                    text_delta = DeltaMessage(content=drained)
                    self._emit_delta(text_delta)
                    return text_delta

            # If this chunk contains end tags but wasn't triggered by parser,
            # manually complete end events
            # Only execute when still on the same call as when entered,
            # to prevent accidentally closing new calls
            # in multi <tool_call> scenarios
            if (
                self.current_call_id is not None
                and self.current_call_id == initial_call_id
                and (
                    self.function_end_token in xml_chunk
                    or self.tool_call_end_token in xml_chunk
                )
            ):
                # Close potentially unclosed element
                if self.current_param_name:
                    self._end_element("parameter")
                if self.function_end_token in xml_chunk and self.current_function_name:
                    self._end_element("function")
                if self.tool_call_end_token in xml_chunk:
                    self._end_element("tool_call")
                # Return the merged delta result generated by this fallback
                result_delta = self._merge_new_deltas_to_single_response(
                    initial_delta_count
                )
                return result_delta

            # No complete elements, return empty response
            return DeltaMessage(content=None)

    def _escape_xml_special_chars(self, text: str) -> str:
        """
        Escape XML special characters
        Args:
            text: Original text
        Returns:
            Escaped text
        """
        xml_escapes = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&apos;",
        }

        for char, escape in xml_escapes.items():
            text = text.replace(char, escape)

        return text

    def _process_complete_xml_elements(self) -> bool:
        """
        Process complete XML elements in buffer

        Returns:
            bool: Whether complete elements were found and processed
        """
        found_any = False

        while self.last_processed_pos < len(self.streaming_buffer):
            # Find next complete xml element
            element, end_pos = self._find_next_complete_element(self.last_processed_pos)
            if element is None:
                # No complete element found, wait for more data
                break

            # opt22: normalize Anthropic-style tag names (toolcall / tool /
            # param) to Qwen3-native (tool_call / function / parameter) so the
            # startswith checks below and the expat handlers recognize them.
            # Streaming + non-streaming both route through here.
            element = self._normalize_tag_names(element)

            # opt22d 第十二轮（缺陷B·滚动扣住）：参数含示例时，收尾标签有歧义
            # （真调用收尾 vs 文档示例），**扣住**不喂 expat；等出现"非收尾、非空白"
            # 的元素确证它们是**内容**再释放（转义后喂 expat）；否则 finish 时丢弃。
            # 关键：**参数正文照常即时下发**，只延迟这几十字节——v4 曾把整个参数
            # 缓冲到 finish，大文件（20k-30k token）会 5 分钟无数据、客户端断链。
            _el = element.strip()
            _is_close = _el in _CLOSE_TAGS
            if self._hold_tags:
                # 已有扣住的收尾标签：本元素若既非收尾、又非空白 ⇒ 确证前者是内容
                if not _is_close and _el:
                    for _ht in self._hold_tags:
                        self.parser.Parse(_as_xml_text(_ht), False)
                    self._hold_tags = []
            if _is_close and self._param_example_state.get("has_example"):
                self._hold_tags.append(_el)
                self.last_processed_pos = end_pos
                continue

            # Check if this element should be skipped
            if self._should_skip_element(element):
                self.last_processed_pos = end_pos
                continue

            # Found complete XML element, process it
            try:
                # opt22d 第十二轮（缺陷B）：把「该元素之后已到达的文本」交给预处理，
                # 供参数体 `</parameter>` 的真伪判定用（流式下可能为空 ⇒ 按真闭合）。
                self._peek = self.streaming_buffer[end_pos:]
                # opt22d 第十二轮（缺陷B）：先补发上一轮扣住的 `</parameter>`
                # （**独立 Parse** ⇒ expat 状态先更新；否则当前元素会被误判为
                # "仍在参数体内"）。^ 顺序：补发 → 本元素预处理 → 本元素 Parse。
                self._flush_pending_param_close(element)
                preprocessed_element = self._preprocess_xml_chunk(element)
                # opt22d 第十二轮（缺陷B·续）：被转义成文档内容的 `</function>` /
                # `</tool_call>` **不算** expat 看到了收尾——置位只认放行给 expat 的。
                _pe = preprocessed_element.strip()
                if _pe == self.function_end_token:
                    self._saw_fn_end = True
                elif _pe == self.tool_call_end_token:
                    self._saw_tc_end = True
                # Check if this is the first tool_call start
                if (
                    (
                        preprocessed_element.strip().startswith("<tool_call>")
                        or preprocessed_element.strip().startswith("<function name=")
                    )
                    and self.tool_call_index == 0
                ) and self.text_content_buffer:
                    # First tool_call starts,
                    # output previously collected text content first
                    # opt22d（L-E）：转义在冲刷时统一施加
                    drained = self._drain_text_buffer()
                    if drained:
                        self._emit_delta(DeltaMessage(content=drained))

                # If a new tool_call starts and
                # there are already completed tool_calls
                if (
                    preprocessed_element.strip().startswith("<tool_call>")
                    and self.tool_call_index > 0
                    and self.current_call_id
                ):
                    # Reset parser state but preserve generated deltas
                    if self.current_param_name:
                        self._end_element("parameter")
                    if self.current_function_open or self.current_function_name:
                        self._end_element("function")
                    # Output final tool_call tail delta
                    final_delta = DeltaMessage(
                        role=None,
                        content=None,
                        reasoning=None,
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                type="function",
                                function=DeltaFunctionCall(name=None, arguments=""),
                            )
                        ],
                    )
                    self._emit_delta(final_delta)
                    # Reset XML parser and current call state
                    self._reset_xml_parser_after_tool_call()
                # Parse preprocessed element
                self.parser.Parse(preprocessed_element, False)
                found_any = True

            except Exception as e:
                # [上游 #842 融入 2026-10-04] Expat retains an unmatched
                # parameter in its XML stack even when our value handler has
                # recovered at the next parameter. Recover only the closing
                # element being processed, so a close earlier in a burst cannot
                # accidentally finish its next call.
                #
                # ⚠️ 本 fork 备注（2026-10-04 三重验证）：此分支在**本仓为死代码**，
                # 永不命中 —— ① 消融「有/无 #842」结果完全一致（畸形输入均回
                # ['weather']）；② 在 except 处插桩从未打印；③ 上游
                # tests/tool_parsers/test_qwen3xml_chunk_boundaries.py 有/无均 13 fail。
                # 原因：本 fork 的**分支归一化 + _preprocess_xml_chunk** 已把畸形输入
                # 转成 expat 不抛异常的形态，走不到这里。**保留仅为与上游对齐**（无害）；
                # 本仓的畸形恢复由归一化路径承担。详见
                # patches/opt22-upstream-842-xml-malformed-close-v1.patch 与优化文档 opt22 段。
                closing = element.strip()
                if self.current_call_id is not None and closing in (
                    self.function_end_token,
                    self.tool_call_end_token,
                ):
                    if closing == self.tool_call_end_token:
                        self._end_element("tool_call")
                    elif self.current_function_open:
                        self._end_element("function")
                    found_any = True
                else:
                    logger.warning("Error when parsing XML elements: %s", e)

            # Update processed position
            self.last_processed_pos = end_pos

        return found_any

    def _scan_code_fence(self, text: str) -> None:
        """opt22d（L-C）：在顶层文本里跟踪 markdown 代码围栏状态。

        只认 **行首** 的三连反引号（CommonMark 语义）——行内出现的反引号不计入，
        避免正文里提到反引号时误开围栏。逐字符扫描，状态跨元素、跨 chunk 保持，
        天然处理流式把围栏标记切开的场景。
        """
        for ch in text:
            if ch == "\n":
                self._fence_line_start = True
                self._fence_run = 0
                continue
            if self._fence_line_start:
                if ch in " \t":
                    continue
                if ch == "`":
                    self._fence_run += 1
                    if self._fence_run == 3:
                        self._in_code_fence = not self._in_code_fence
                    continue
                self._fence_line_start = False
                self._fence_run = 0

    def _update_line_start(self, text: str) -> None:
        """opt22d（L-D）：依顶层文本更新“下一个元素是否在行首”。

        规则：文本含换行 → 取最后一个换行之后的部分，全空白才算行首；
        不含换行 → 仅当此前已在行首**且**本段全空白时，仍算行首。
        """
        if not text:
            return
        nl = text.rfind("\n")
        if nl != -1:
            self._at_line_start = text[nl + 1:].strip() == ""
        else:
            self._at_line_start = self._at_line_start and text.strip() == ""

    def _drain_text_buffer(self) -> str:
        """opt22d（L-E）：把顶层文本缓冲**整体转义**后取出。

        末尾若是不闭合的工具类标签片段（流式下标签会被切开），则**留在缓冲里**
        等下一轮补齐后再一起转义——否则残缺片段上转义会漏出裸 `<` 或产出畸形。
        """
        head, held = split_incomplete_tool_tag_tail(self.text_content_buffer)
        self.text_content_buffer = held
        # [example 区间 2026-10-18] 示例容器内原样，只转义区间外
        return self._escape_tool_like_tags_outside_examples(head) if head else ""

    def _flush_text_buffer_forced(self) -> str:
        """opt22d（L-E）：消息结束时强制刷出缓冲（含未闭合尾部），转义后返回。

        仅在**确定不再有新输入**时调用（非流式），否则会把残缺片段转义成畸形。
        """
        if not self.text_content_buffer:
            return ""
        buf = self.text_content_buffer
        self.text_content_buffer = ""
        return self._escape_tool_like_tags_outside_examples(buf)

    def _should_skip_element(self, element: str) -> bool:
        """
        Determine whether an element should be skipped

        Args:
            element: Element to evaluate

        Returns:
            bool: True means should skip, False means should process
        """

        # opt22d（L-C）：顶层文本运行先更新围栏状态（真实调用内部不跟踪，
        # 参数值里的反引号只是数据）
        if self.current_call_id is None and not element.startswith("<"):
            self._scan_code_fence(element)
            # opt22e A：已产出调用之后又出现顶层正文 ⇒ 进入"说明区"，
            # 此后的形似调用一律按示例处理（见下方 L-D 判定）。
            if self._opt22e_seen_call and element.strip():
                self._opt22e_seen_prose = True

        # opt22d（L-C）：围栏内的一切（含工具调用标签）都是“展示给用户的示例”文本，
        # 一律转义后并入 content，不作为真实调用解析
        if self.current_call_id is None and self._in_code_fence:
            if element:
                self.text_content_buffer += element
                self._update_line_start(element)
            return True

        # opt22d（L-D）：顶层文本里的**裸标签**必须在行首才算真实调用。
        # 模型展示格式时惯用行内代码块，例如
        #     - **`<function=Write>`**：指定要调用的工具名为 `Write`
        # 这类示例带真实注册名（绕过 L-A 的工具名白名单）、也不在围栏内（绕过 L-C），
        # 只能靠位置区分。实测 8/8 真实调用（probe_linestart.py）均位于行首。
        # `<parameter=` 在顶层（未进入任何调用）本就无意义，一律降级为文本。
        if self.current_call_id is None and (
            element.startswith(self.tool_call_start_token)
            or element.startswith(self.function_start_token)
            or element.startswith(self.parameter_start_token)
        ):
            is_bare_param = element.startswith(self.parameter_start_token)
            is_wrapped_function = (
                element.startswith(self.function_start_token)
                and self._pending_tool_call_wrapper
            )
            # opt22e A：已产出调用 + 其后出现正文 ⇒ 形似调用降级为文本，
            # 防「正文里的完整示例被当成真实调用执行」。
            if (
                is_bare_param
                or (not is_wrapped_function and not self._at_line_start)
                or (self._opt22e_seen_call and self._opt22e_seen_prose)
            ):
                self.text_content_buffer += element
                self._update_line_start(element)
                self._pending_tool_call_wrapper = False
                return True

        # If it's a tool_call XML tag, don't skip
        if (
            element.startswith(self.tool_call_start_token)
            or element.startswith(self.function_start_token)
            or element.startswith(self.parameter_start_token)
        ):
            if element.startswith(self.tool_call_start_token):
                self._pending_tool_call_wrapper = True
                self._opt22e_seen_call = True      # opt22e A：已产出真实调用
            elif element.startswith(self.function_start_token):
                self._pending_tool_call_wrapper = False
            return False

        # If currently not parsing tool calls and not blank,
        # collect this text instead of skipping
        # Only process other XML elements after tool_call appears,
        # otherwise treat as plain text
        if self.current_call_id is None and element:
            # Collect text content to buffer (opt22d L-B: 转义推迟到 _drain_text_buffer)
            self.text_content_buffer += element
            self._update_line_start(element)
            if element.strip():
                self._pending_tool_call_wrapper = False
            return True  # Still skip, but content has been collected

        # If currently parsing tool calls,
        # this might be parameter value, don't skip
        if self.current_call_id is not None:
            return False

        # Skip blank content
        return not element

    def _find_next_complete_element(self, start_pos: int) -> tuple[str | None, int]:
        """
        Find next complete XML element from specified position

        Args:
            start_pos: Position to start searching

        Returns:
            (Complete element string, element end position),
            returns (None, start_pos) if no complete element found
        """
        buffer = self.streaming_buffer[start_pos:]

        if not buffer:
            return None, start_pos

        if buffer.startswith("<"):
            # Need to ensure no new < appears,
            # find the nearest one between < and >
            tag_end = buffer.find("<", 1)
            tag_end2 = buffer.find(">", 1)
            if tag_end != -1 and tag_end2 != -1:
                # Next nearest is <
                if tag_end < tag_end2:
                    return buffer[:tag_end], start_pos + tag_end
                # Next nearest is >, means found XML element
                else:
                    return buffer[: tag_end2 + 1], start_pos + tag_end2 + 1
            elif tag_end != -1:
                return buffer[:tag_end], start_pos + tag_end
            elif tag_end2 != -1:
                return buffer[: tag_end2 + 1], start_pos + tag_end2 + 1
            else:
                # If currently not parsing tool calls (entering a tool_call),
                # check if starts with <tool_call> or <function=
                if self.current_call_id is None:
                    # Check if might be start of <tool_call>
                    if buffer == "<tool_call>"[: len(buffer)]:
                        # Might be start of <tool_call>, wait for more data
                        return None, start_pos
                    elif (
                        buffer.startswith("<function=")
                        or buffer == "<function="[: len(buffer)]
                    ):
                        # Might be start of <function=, wait for more data
                        # to get the complete function tag
                        return None, start_pos
                    else:
                        # Not start of <tool_call> or <function=, treat as text
                        return buffer, start_pos + len(buffer)
                else:
                    # When parsing tool calls,
                    # wait for more data to get complete tag
                    return None, start_pos
        else:
            # Find text content (until next < or buffer end)
            next_tag_pos = buffer.find("<")
            if next_tag_pos != -1:
                # Found text content
                text_content = buffer[:next_tag_pos]
                return text_content, start_pos + next_tag_pos
            else:
                # Buffer end is all text, process
                # (no longer wait for more data)
                remaining = buffer
                return remaining, start_pos + len(remaining)

    def _merge_new_deltas_to_single_response(self, initial_count: int) -> DeltaMessage:
        """
        Merge newly generated deltas from this processing
        into a single DeltaMessage

        Args:
            initial_count: Delta count before processing

        Returns:
            Merged DeltaMessage containing all newly generated delta information
        """
        if len(self.deltas) <= initial_count:
            return DeltaMessage(content=None)

        # Get newly generated deltas
        new_deltas = self.deltas[initial_count:]

        if len(new_deltas) == 1:
            # Only one new delta, return directly
            return new_deltas[0]

        # Merge multiple new deltas
        merged_tool_calls: list[DeltaToolCall] = []
        merged_content: str = ""

        for delta in new_deltas:
            if delta.content:
                merged_content += delta.content
            if delta.tool_calls:
                # For tool_calls, we need to intelligently merge arguments
                for tool_call in delta.tool_calls:
                    # Find if there's already a tool_call with the same call_id
                    existing_call = None
                    for existing in merged_tool_calls:
                        if existing.id == tool_call.id:
                            existing_call = existing
                            break

                    if existing_call and existing_call.function:
                        # Merge to existing tool_call
                        if tool_call.function and tool_call.function.name:
                            existing_call.function.name = tool_call.function.name
                        if (
                            tool_call.function
                            and tool_call.function.arguments is not None
                        ):
                            if existing_call.function.arguments is None:
                                existing_call.function.arguments = ""

                            # For streaming JSON parameters,
                            # simply concatenate in order
                            new_args = tool_call.function.arguments
                            existing_call.function.arguments += new_args
                        if tool_call.type:
                            existing_call.type = tool_call.type
                    else:
                        # Add new tool_call
                        merged_tool_calls.append(tool_call)

        return DeltaMessage(
            content=merged_content if merged_content else None,
            tool_calls=merged_tool_calls,
        )

    def _normalize_tag_names(self, element: str) -> str:
        """
        opt22: Normalize Anthropic-style tool-call tag names to the Qwen3-native
        names this parser recognizes, so the startswith checks and expat handlers
        treat them as tool elements instead of plain text.

        Maps to the equals-attribute form the existing re.sub + is_tool_call
        detection already handle natively:
          toolcall -> tool_call
          tool name="X" -> function=X
          param name="X" -> parameter=X
          closing tool / param -> function / parameter
        Native tool_call / function / parameter tags and unrelated text such as
        "toolbar" are left untouched.
        """
        e = re.sub(r'<tool name="([^"]*)">', r'<function=\1>', element)
        e = re.sub(r'<param name="([^"]*)">', r'<parameter=\1>', e)
        e = re.sub(r'<toolcall>', '<tool_call>', e)
        e = re.sub(r'</toolcall>', '</tool_call>', e)
        e = re.sub(r'</tool>', '</function>', e)
        e = re.sub(r'</param>', '</parameter>', e)
        return e

    def _preprocess_xml_chunk(self, chunk: str) -> str:
        """预处理单个元素（见 `_preprocess_xml_chunk_inner`）。

        注：上一轮**扣住**的 `</parameter>` 的补发**不在这里**——它必须作为
        **独立一次 `Parse`** 先喂给 expat（见 `_flush_pending_param_close`），
        否则 expat 尚未消费它、`current_param_name` 仍非空，当前元素会被
        误判成"仍在参数体内"而被转义（实测：`</function>` 被转义 ⇒ 整个调用丢失）。
        """
        return self._preprocess_xml_chunk_inner(chunk)

    def _flush_pending_param_close(self, upcoming: str = "") -> None:
        """补发上一轮**扣住**的 `</parameter>`（**单独 Parse**，先于当前元素）。

        流式下 `</parameter>` 若是该 delta 的最后一个元素，其后可能紧跟
        `</function>`（真收尾）或普通正文（文档示例），当时判不了 ⇒ 先扣住不喂
        expat，等下一元素到达、peek 有内容再定夺（见 `escape_param_body_example_tags`）。
        """
        if not self._pending_param_close:
            return
        # ⚠️ 判据 = **当前即将处理的元素 + 其后已到达的文本**。
        # 极细步长（step=1）下 `self._peek` 常为空——因为紧跟的 `</function>`
        # 此刻正作为"当前元素"待处理，只有把它算进来才看得到收尾。
        _peek = (upcoming or "") + (self._peek or "")
        # 判据**仍未到齐**（只到空白）⇒ **继续扣住**等更多内容。若此时就判，
        # `_is_real_param_close("")` 会按"文本结束 ⇒ 真收尾"放行，把**文档示例**
        # 的 `</parameter>` 误当真闭合 ⇒ 幽灵调用（实测 step=1）。
        if not _peek.strip():
            return
        self._pending_param_close = False
        # ⚠️ 注意**不要**写成 `if self._peek.strip() and _is_real_param_close(...)`
        # 之类的短路形式——曾因此把该补发的真闭合转义掉、整个调用丢失。
        if _is_real_param_close(_peek):
            _head = self.parameter_end_token
        else:
            _head = _as_xml_text(self.parameter_end_token)
        # ⚠️ 补发必须**禁止再次扣留**：极细步长下 peek 可能仍为空，若不禁止，
        # `inner(_head)` 会把 `_head` 又扣住 ⇒ 死循环（`</parameter>` 被反复
        # 扣住/补发，永远到不了 expat）。
        # ⚠️ 同时必须**同步 `_peek`**：`inner` 会用 `self._peek` 再判一次，而
        # 这里用的是「当前元素 + peek」的合并判据；不同步就会出现"外层判真闭合、
        # 内层判文档示例"的自相矛盾 ⇒ 真闭合被转义、参数吞掉后续（实测 EXH[08]）。
        # ⚠️ **不能**把 `_head` 再喂给 `inner` 走通用预处理：`_head` 可能已经是
        # 转义形式（`&lt;/parameter&gt;`），inner 会对它再 `_escape_xml_special_chars`
        # 一次 ⇒ `&amp;lt;` ⇒ expat 解码后留下**字面** `&lt;` 泄漏进参数值
        # （实测 EXH[08]：参数值里出现 `&lt;/parameter&gt;`）。
        # 只有 raw 缓冲收尾这一种情况需要走 inner（冲刷累积原文）。
        _saved = self._streaming_mode
        _saved_peek = self._peek
        self._streaming_mode = False
        self._peek = _peek
        try:
            if _head == self.parameter_end_token and self._pre_inside_parameter:
                _out = self._preprocess_xml_chunk_inner(self.parameter_end_token)
            else:
                _out = _head
            self.parser.Parse(_out, False)
        finally:
            self._streaming_mode = _saved
            self._peek = _saved_peek

    def _preprocess_xml_chunk_inner(self, chunk: str) -> str:
        """
        Preprocess XML chunk, handle non-standard formats,
        and escape special characters

        Args:
            chunk: Original XML chunk

        Returns:
            Processed XML chunk
        """

        # opt22d 第十二轮（缺陷B）：位于**参数值内部**时，形似工具调用的标签是
        # **文档内容**而非真实调用——模型写说明文档时会原样写出 `<function=...>`
        # 示例（如"如何手写 <function=Bash> 调用"），甚至自带完整闭合
        # `</parameter></function></tool_call>`。若不处理，expat 会把它们当成
        # 新元素 ⇒ 提前闭合参数、覆盖 current_function_name、参数错乱
        # （线上实测：一次 Write 文档被切成 Write + 3 个 Bash 幽灵调用）。
        # 规则见 `escape_param_body_example_tags`（纯 peek 结构判定，不用计数）。
        if not self._pre_inside_parameter:
            _body_at = -1
            if self.current_param_name is not None:
                # 已在参数值内：整个 chunk 都是参数正文
                _body_at = 0
            else:
                # 本 chunk 自身可能刚打开参数（**非流式**下整段会作为一个 chunk
                # 进来，此时 expat 尚未处理 <parameter=>，故 current_param_name
                # 仍为空）。取最后一个 <parameter=…> 之后的正文。
                _opens = list(_PARAM_OPEN_ONLY_RE.finditer(chunk))
                if _opens:
                    _close = chunk.find("</parameter>", _opens[-1].end())
                    if _close == -1:  # 尚未闭合 ⇒ 其后全为参数正文
                        _body_at = _opens[-1].end()
            if 0 <= _body_at < len(chunk):
                _head, _body = chunk[:_body_at], chunk[_body_at:]
                _esc, _pending = escape_param_body_example_tags(
                    _body,
                    self._peek or "",
                    allow_pending=self._streaming_mode,
                    state=self._param_example_state,
                )
                # 流式下 `</parameter>` 是末尾元素 ⇒ 扣住等下一元素；非流式
                # （`self._streaming_mode` 为假）下消息已完整，不存在后续判据。
                if _pending:
                    self._pending_param_close = True
                if _esc != _body or _pending:
                    return _head + _esc

        # Check if this is a tool_call related element
        is_tool_call = False
        if chunk.startswith(self.tool_call_start_token) or chunk.startswith(
            self.tool_call_end_token
        ):
            is_tool_call = True
        if chunk.startswith(self.function_start_token) or chunk.startswith(
            self.function_end_token
        ):
            is_tool_call = True
        if chunk.startswith(self.parameter_start_token) or chunk.startswith(
            self.parameter_end_token
        ):
            is_tool_call = True
        # Handle <function=name> format -> <function name="name">
        processed = re.sub(r"<function=([^>]+)>", r'<function name="\1">', chunk)
        # Handle <parameter=name> format -> <parameter name="name">
        processed = re.sub(r"<parameter=([^>]+)>", r'<parameter name="\1">', processed)

        original_chunk = chunk
        # If in parameter value accumulation mode
        if self._pre_inside_parameter:
            # Parameter end: output accumulated raw text
            # safely then return </parameter>
            if processed.startswith("</parameter>"):
                # opt22d 第十二轮（缺陷B）：raw 缓冲**不经 expat 解码** ⇒ 把此前
                # 为防 expat 误解析而转义的工具标签**还原回原字符**，否则实体会
                # 原样泄漏进参数值（实测 EXH[08]：参数值里出现字面 `&lt;/parameter&gt;`）。
                body_text = _unescape_body_example_tags(self._pre_param_buffer)
                # Trigger deferred parsing mode
                # literal_eval+json output in end_element
                self.defer_current_parameter = True
                self.deferred_param_raw_value = body_text
                # Clean up state
                self._pre_inside_parameter = False
                self._pre_param_buffer = ""
                self._pre_current_param_name = None
                safe_text = self._escape_xml_special_chars(body_text)
                return f"{safe_text}</parameter>"
            else:
                # If this is the first block of content after entering parameter
                # evaluate if deferred parsing is needed;
                # If not needed, exit accumulation mode
                # and pass through directly
                if self._pre_param_buffer == "":
                    # Get current parameter type
                    param_type = (
                        self._get_param_type(self._pre_current_param_name)
                        if self._pre_current_param_name
                        else "string"
                    )
                    # Only these types need deferred parsing to
                    # handle Python literals containing single quotes
                    is_object_type = param_type in ["object"]
                    is_complex_type = (
                        param_type in ["array", "arr", "sequence"]
                        or param_type.startswith("dict")
                        or param_type.startswith("list")
                    )

                    # Only delay when contains container symbols
                    # and has single quotes and is complex type
                    has_container_hint = (
                        ("[" in original_chunk)
                        or ("{" in original_chunk)
                        or ("(" in original_chunk)
                    )

                    # Determine if deferred parsing is needed
                    need_defer = False
                    if is_complex_type:
                        # Complex type, always need deferred parsing
                        need_defer = True
                    elif (
                        is_object_type
                        and has_container_hint
                        and ("'" in original_chunk)
                    ):
                        # Object type with container symbols
                        # and single quotes, need deferred parsing
                        need_defer = True

                    if not need_defer:
                        # No need for deferred parsing,
                        # exit parameter mode directly
                        self._pre_inside_parameter = False
                        return self._escape_xml_special_chars(original_chunk)
                self._pre_param_buffer += original_chunk
                return ""

        # Parameter start: enable accumulation
        if processed.startswith("<parameter name="):
            m = re.match(r'<parameter name="([^"]+)">', processed)
            if m:
                self._pre_current_param_name = m.group(1)
            self._pre_inside_parameter = True
            self._pre_param_buffer = ""
            return processed

        # If processed doesn't contain special_token, escape processed
        # This is because XML parsing encounters special characters
        # and reports errors, so escaping is needed
        if not is_tool_call:
            processed = self._escape_xml_special_chars(processed)
        return processed

    def _registered_tool_names(self) -> set:
        """已注册工具名集合。tools 未提供时为空集。
        [opt22 去重 2026-10-14] 实现已上提 utils（coder/xml 双份逐字重复），薄委托。"""
        return _u_registered_tool_names(self.tools)

    def _is_registered_tool(self, name) -> bool:
        """opt22d L-A: 名称是否在已注册工具内。
        tools 为空（非工具请求）时一律视为已注册，保持宽松、绝不误伤真实调用。"""
        if self._opt22d_force_all:
            return False  # opt22d A/B/C：整轮降级
        names = self._registered_tool_names()
        if not names:
            return True
        return name in names

    def _escape_tool_like_tags_outside_examples(self, text: str) -> str:
        """[example 区间 2026-10-18] 示例容器内原样，只转义区间外。"""
        from vllm.tool_parsers.qwen3_120_utils import escape_tool_like_tags_outside_examples as _f

        return _f(text)

    def _escape_tool_like_tags(self, text: str) -> str:
        """opt22d L-B: 把顶层文本里“形似工具调用”的裸标签换成显示安全的形式，
        使下游 harness 不再把这些文本当成真实调用执行。

        转义形如 `<function=Write>` -> `⟨'function=Write'⟩`（数学尖括号 + 成对单引号）。
        不用 `&lt;`/`&gt;` 实体的原因：① 纯文本渲染下很难看；② 若下游做 HTML 解码
        会被还原成 `<` 而失效。不用全角 `＜＞` 的原因：NFKC 会映射回 ASCII `<`/`>`。

        仅命中工具调用词族的标签名；a<b、20<3、<p> 等普通文本不受影响。
        正则要求 `<` 紧跟标签名，替换后不再有 `<` ⇒ 幂等。
        （本转义**不是**前缀稳定的：末尾残缺标签是否命中取决于标签名是否完整，
        这与替换成什么符号无关；coder 流式的 startswith 失配分支本就有兜底。）"""
        if not text:
            return text
        return _TOOL_LIKE_TAG_RE.sub(
            lambda m: m.group(0).replace("<", "⟨'").replace(">", "'⟩"), text
        )

    def _emit_demoted_call_text(self):
        """opt22d L-A: 把降级的未注册调用以纯文本形式并回内容缓冲（幂等）。"""
        if self._demote_note_emitted:
            return
        self._demote_note_emitted = True
        name = self.current_function_name or "?"
        try:
            args = json.dumps(self.parameters, ensure_ascii=False)
        except Exception:
            args = "{}"
        if self._opt22d_force_all:
            note = "\n[本轮不产生工具调用，已按文本处理：%s(%s)]\n" % (name, args)
        else:
            note = "\n[未注册的工具调用，已按文本处理：%s(%s)]\n" % (name, args)
        self.text_content_buffer += note
        self._demote_current_call = False

    def _emit_delta(self, delta: DeltaMessage):
        """Emit Delta response (streaming output)"""
        # opt22d L-A: 降级调用期间吞掉其结构化 tool_call 增量（保留其中的 content）
        if self._demote_current_call and delta.tool_calls:
            if delta.content:
                self.deltas.append(DeltaMessage(content=delta.content))
            return
        self.deltas.append(delta)

    def _auto_close_open_parameter_if_needed(self, incoming_tag: str | None = None):
        """Before starting to process new elements,
        if there are unclosed tags from before,
        automatically complete their endings to the parser.
        - If there are unclosed parameters,
        it's equivalent to feeding `</parameter>`
        - When about to start a new function or tool_call,
        if there are unclosed functions, complete `</function>`.
        - When about to start a new tool_call,
        if there are unclosed tool_calls, complete `</tool_call>`.
        """
        # First close unclosed parameters
        if self.current_param_name:
            self._end_element("parameter")

        # If about to start new function or tool_call,
        # and there are unclosed functions, close function first
        if incoming_tag in ("function", "tool_call") and self.current_function_name:
            self._end_element("function")

        # If about to start new tool_call,
        # and there are unclosed tool_calls, close tool_call first
        if incoming_tag == "tool_call" and self.current_call_id:
            self._end_element("tool_call")

    def _start_element(self, name: str, attrs: dict[str, str]):
        """Handle XML start element events"""

        if name == "root":
            return

        if name == "tool_call":
            # Before opening new tool_call,
            # automatically complete previous unclosed tags
            self._auto_close_open_parameter_if_needed("tool_call")

            self.parameters = {}
            self.current_call_id = make_tool_call_id()
            self.current_param_is_first = True
            self.tool_call_index += 1
        elif name.startswith("function") or (name == "function"):
            # If missing tool_call, manually complete
            if not self.current_call_id:
                self._start_element("tool_call", {})
                # opt22d 第十二轮（缺陷A）：标记为“自动补包裹”的裸调用
                self._autofilled_call = True
            # Before opening new function,
            # automatically complete previous unclosed tags (parameter/function)
            self._auto_close_open_parameter_if_needed("function")
            function_name = self._extract_function_name(name, attrs)
            self.current_function_name = function_name
            self.current_function_open = True
            # opt22d L-A: 名称未注册 → 本次调用降级为文本
            if function_name and not self._is_registered_tool(function_name):
                self._demote_current_call = True
                self._demote_note_emitted = False
            if function_name and not self._demote_current_call:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            type="function",
                            function=DeltaFunctionCall(
                                name=function_name, arguments=""
                            ),
                        )
                    ]
                )
                self._emit_delta(delta)
        elif name.startswith("parameter") or (name == "parameter"):
            # If previous parameter hasn't ended normally,
            # complete its end first, then start new parameter
            self._auto_close_open_parameter_if_needed("parameter")
            param_name = self._extract_parameter_name(name, attrs)
            self.current_param_name = param_name
            self.current_param_value = ""
            self.current_param_value_converted = ""
            self.start_quote_emitted = False  # Reset start quote flag

            # Only output parameter name and colon,
            # don't output quotes
            # decide after parameter value type is determined
            if param_name:
                if not self.parameters:
                    # First parameter
                    # start JSON, only output parameter name and colon
                    json_start = f'{{"{param_name}": '
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                type="function",
                                function=DeltaFunctionCall(
                                    name=None, arguments=json_start
                                ),
                            )
                        ]
                    )
                    self._emit_delta(delta)
                    self.current_param_is_first = True
                else:
                    # Subsequent parameters
                    # add comma and parameter name, no quotes
                    json_continue = f', "{param_name}": '
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                type="function",
                                function=DeltaFunctionCall(
                                    name=None, arguments=json_continue
                                ),
                            )
                        ]
                    )
                    self._emit_delta(delta)
                    self.current_param_is_first = False

    def _char_data(self, data: str):
        """Handle XML character data events"""
        if data and self.current_param_name:
            # If preprocessing stage determines deferred parsing is needed,
            # only cache character data, no streaming output
            if self.defer_current_parameter:
                original_data = data
                if self.should_emit_end_newline:
                    original_data = "\n" + original_data
                    self.should_emit_end_newline = False
                if original_data.endswith("\n"):
                    self.should_emit_end_newline = True
                    original_data = original_data[:-1]
                self.current_param_value += original_data
                return

            param_type = self._get_param_type(self.current_param_name)

            # Check if this is the first time receiving data for this parameter
            # If this is the first packet of data and starts with \n, remove \n
            if not self.current_param_value and data.startswith("\n"):
                data = data[1:]

            # Output start quote for string type (if not already output)
            if (
                param_type in ["string", "str", "text", "varchar", "char", "enum"]
                and not self.start_quote_emitted
            ):
                quote_delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            type="function",
                            function=DeltaFunctionCall(name=None, arguments='"'),
                        )
                    ]
                )
                self._emit_delta(quote_delta)
                self.start_quote_emitted = True

            if not data:
                return

            original_data = data
            # Delay output of trailing newline
            if self.should_emit_end_newline:
                original_data = "\n" + original_data
                self.should_emit_end_newline = False
            if original_data.endswith("\n"):
                self.should_emit_end_newline = True
                original_data = original_data[:-1]
            self.current_param_value += original_data

            # convert parameter value by param_type
            converted_value = self._convert_param_value(
                self.current_param_value, param_type
            )
            output_data = self._convert_for_json_streaming(converted_value, param_type)

            delta_data = output_data[len(self.current_param_value_converted) :]
            self.current_param_value_converted = output_data

            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.tool_call_index - 1,
                        id=self.current_call_id,
                        type="function",
                        function=DeltaFunctionCall(name=None, arguments=delta_data),
                    )
                ]
            )
            self._emit_delta(delta)

    def _end_element(self, name: str):
        """Handle XML end element events"""

        if name == "root":
            return

        # If function or tool_call ends and there are still unclosed parameters,
        # complete parameter end first
        if (
            name.startswith("function") or name == "function" or name == "tool_call"
        ) and self.current_param_name:
            self._auto_close_open_parameter_if_needed()

        if (
            name.startswith("parameter") or name == "parameter"
        ) and self.current_param_name:
            # End current parameter
            param_name = self.current_param_name
            param_value = self.current_param_value

            # If in deferred parsing mode,
            # perform overall parsing on raw content
            # accumulated in preprocessing stage and output once
            if self.defer_current_parameter:
                raw_text = (
                    self.deferred_param_raw_value
                    if self.deferred_param_raw_value
                    else param_value
                )
                parsed_value = None
                output_arguments = None
                try:
                    # If previously delayed trailing newline,
                    # add it back before parsing
                    if self.should_emit_end_newline:
                        raw_for_parse = raw_text + "\n"
                    else:
                        raw_for_parse = raw_text
                    try:
                        parsed_value = json.loads(raw_for_parse)
                    except json.JSONDecodeError:
                        parsed_value = ast.literal_eval(raw_for_parse)
                    output_arguments = json.dumps(parsed_value, ensure_ascii=False)
                except Exception:
                    # Fallback: output as string as-is
                    output_arguments = json.dumps(raw_text, ensure_ascii=False)
                    parsed_value = raw_text

                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            type="function",
                            function=DeltaFunctionCall(
                                name=None, arguments=output_arguments
                            ),
                        )
                    ]
                )
                self._emit_delta(delta)

                # Clean up and store
                self.should_emit_end_newline = False
                self.parameters[param_name] = parsed_value
                self.current_param_name = None
                self.current_param_value = ""
                self.current_param_value_converted = ""
                self.start_quote_emitted = False
                self.defer_current_parameter = False
                self.deferred_param_raw_value = ""
                # opt22d 第十二轮（缺陷B）：参数结束 ⇒ 示例标记复位（每个参数独立判定）
                self._param_example_state = {"has_example": False}
                return

            param_type = self._get_param_type(param_name)

            # convert complete parameter value by param_type
            converted_value = self._convert_param_value(param_value, param_type)

            # Decide whether to add end quote based on parameter type
            if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
                # For empty string parameters, need special handling
                if not param_value and not self.start_quote_emitted:
                    # No start quote output,
                    # directly output complete empty string
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                type="function",
                                function=DeltaFunctionCall(name=None, arguments='""'),
                            )
                        ]
                    )
                    self._emit_delta(delta)
                else:
                    # Non-empty parameter value, output end quote
                    delta = DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                type="function",
                                function=DeltaFunctionCall(name=None, arguments='"'),
                            )
                        ]
                    )
                    self._emit_delta(delta)

            self.should_emit_end_newline = False
            # Store converted value
            self.parameters[param_name] = converted_value
            self.current_param_name = None
            self.current_param_value = ""
            self.current_param_value_converted = ""
            self.start_quote_emitted = False

        elif name.startswith("function") or name == "function":
            # if there are parameters, close JSON object
            if self.parameters:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            type="function",
                            function=DeltaFunctionCall(name=None, arguments="}"),
                        )
                    ]
                )
                self._emit_delta(delta)
            # return empty object
            else:
                delta = DeltaMessage(
                    tool_calls=[
                        DeltaToolCall(
                            index=self.tool_call_index - 1,
                            id=self.current_call_id,
                            type="function",
                            function=DeltaFunctionCall(name=None, arguments="{}"),
                        )
                    ]
                )
                self._emit_delta(delta)
            self.current_function_open = False
            # opt22d 第十二轮（缺陷A）：**裸调用**（无 <tool_call> 包裹）在此收尾。
            # 裸调用没有 </tool_call> 可触发 _reset_xml_parser_after_tool_call，
            # 若不收尾，下一个裸 <function=> 会因 current_call_id 仍在而被并入同一
            # 调用 ⇒ **多个裸调用只产出 1 个**（实测「裸+裸」非流式与流式均丢）。
            if self._autofilled_call:
                self._emit_delta(
                    DeltaMessage(
                        tool_calls=[
                            DeltaToolCall(
                                index=self.tool_call_index - 1,
                                id=self.current_call_id,
                                type="function",
                                function=DeltaFunctionCall(name=None, arguments=""),
                            )
                        ]
                    )
                )
                if self.text_content_buffer.strip():
                    _drained = self._drain_text_buffer()
                    if _drained.strip():
                        self._emit_delta(DeltaMessage(content=_drained))
                self._reset_xml_parser_after_tool_call()

        elif name == "tool_call":
            # Before ending tool_call,
            # ensure function is closed to complete missing right brace
            if self.current_function_open:
                # If there are still unclosed parameters, close them first
                if self.current_param_name:
                    self._end_element("parameter")
                # Close function, ensure output '}' or '{}'
                self._end_element("function")
            # opt22d L-A: 降级调用 → 以文本形式回吐（在 content 缓冲被 flush 之前）
            if self._demote_current_call:
                self._emit_demoted_call_text()
            # Final Delta
            delta = DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.tool_call_index - 1,
                        id=self.current_call_id,
                        type="function",
                        function=DeltaFunctionCall(name=None, arguments=""),
                    )
                ]
            )
            self._emit_delta(delta)

            # Check if there's text content to output (between tool_calls)
            # opt22d（L-E）：转义在冲刷时统一施加
            if self.text_content_buffer.strip():
                drained = self._drain_text_buffer()
                if drained.strip():
                    self._emit_delta(DeltaMessage(content=drained))

            self._reset_xml_parser_after_tool_call()

    def setup_parser(self):
        """Set up XML parser event handlers"""
        self.parser.buffer_text = True
        self.parser.StartElementHandler = self._start_element
        self.parser.EndElementHandler = self._end_element
        self.parser.CharacterDataHandler = self._char_data

    def set_tools(self, tools: list[Tool] | None):
        """Set tool configuration information"""
        self.tools = tools

    def _extract_function_name(self, name: str, attrs: dict[str, str]) -> str | None:
        """Extract function name from various formats"""
        if attrs and "name" in attrs:
            return attrs["name"]

        if "=" in name:
            parts = name.split("=", 1)
            if len(parts) == 2 and parts[0] == "function":
                return parts[1]

        return None

    def _extract_parameter_name(self, name: str, attrs: dict[str, str]) -> str | None:
        """Extract parameter name from various formats"""
        if attrs and "name" in attrs:
            return attrs["name"]

        if "=" in name:
            parts = name.split("=", 1)
            if len(parts) == 2 and parts[0] == "parameter":
                return parts[1]

        return None

    def _get_param_type(self, param_name: str) -> str:
        """Get parameter type based on tool configuration, defaults to string
        Args:
            param_name: Parameter name

        Returns:
            Parameter type
        """
        if not self.tools or not self.current_function_name:
            return "string"

        properties = find_tool_properties(self.tools, self.current_function_name)
        if param_name in properties and isinstance(properties[param_name], dict):
            return self.repair_param_type(
                str(properties[param_name].get("type", "string"))
            )
        return "string"

    def repair_param_type(self, param_type: str) -> str:
        """Repair unknown parameter types by treating them as string
        Args:
            param_type: Parameter type

        Returns:
            Repaired parameter type
        """
        if (
            param_type in ["string", "str", "text", "varchar", "char", "enum"]
            or param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
            or param_type.startswith("num")
            or param_type.startswith("float")
            or param_type in ["boolean", "bool", "binary"]
            or (
                param_type in ["object", "array", "arr", "sequence"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            )
        ):
            return param_type
        else:
            return "string"

    def _convert_param_value(self, param_value: str, param_type: str) -> Any:
        """Convert value based on parameter type
        Args:
            param_value: Parameter value
            param_type: Parameter type

        Returns:
            Converted value
        """
        if param_value.lower() == "null":
            return None

        param_type = param_type.strip().lower()
        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not an integer "
                    "in tool '%s', degenerating to string.",
                    param_value,
                )
            return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value: float = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                logger.warning(
                    "Parsed value '%s' of parameter '%s' is not a float "
                    "in tool '%s', degenerating to string.",
                    param_value,
                )
            return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            return param_value == "true"
        else:
            return param_value

    def _convert_for_json_streaming(self, converted_value: Any, param_type: str) -> str:
        """Convert converted_value based on
        whether it's empty and if type is string
        Args:
            converted_value: Converted value
            param_type: Parameter type

        Returns:
            Converted string for streaming output
        """
        # Check if value is empty, but exclude numeric 0
        if converted_value is None or converted_value == "":
            return ""

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            # String type, remove double quotes
            return json.dumps(converted_value, ensure_ascii=False)[1:-1]
        else:
            # Non-string type, return complete JSON string
            if not isinstance(converted_value, str):
                return json.dumps(converted_value, ensure_ascii=False)
            else:
                return converted_value

    def _reset_xml_parser_after_tool_call(self):
        """
        Each tool_call is treated as a separate XML document,
        so we need to reset the parser after each tool_call.
        """

        # recreate XML parser
        self.parser = ParserCreate()
        self.setup_parser()

        # Reset current tool_call state
        if self.current_call_id:
            self.last_completed_call_id = self.current_call_id
        self.current_call_id = None
        self.current_function_name = None
        self.current_function_open = False
        self.parameters = {}
        self.current_param_name = None
        self.current_param_value = ""
        self.current_param_value_converted = ""
        self.current_param_is_first = False
        self.should_emit_end_newline = False
        self.start_quote_emitted = False
        self.text_content_buffer = ""
        self._demote_current_call = False
        self._demote_note_emitted = False

        # Reset preprocessing and deferred parsing state
        self._pre_inside_parameter = False
        self._pre_param_buffer = ""
        self._pre_current_param_name = None
        self.defer_current_parameter = False
        self.deferred_param_raw_value = ""
        # opt22d 第十二轮（缺陷B）：待补发的 `</parameter>` 不跨调用
        self._pending_param_close = False
        self._peek = ""
        self._param_example_state = {"has_example": False}


class Qwen3XMLToolParser(ToolParser):
    def __init__(self, tokenizer: TokenizerLike, tools: list[Tool] | None = None):
        super().__init__(tokenizer, tools)
        self.parser = StreamingXMLToolCallParser()

        # Add missing attributes for compatibility with serving_chat.py
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

        # opt23 §11.22: JSON 路由状态（对齐 qwen3-coder 机制——qwen3xml 内部实现）
        self.current_tool_index: int = 0
        self.header_sent: bool = False
        self.current_tool_id: str | None = None
        self.current_function_name: str | None = None
        self.in_function: bool = False
        self.json_started: bool = False
        self.json_closed: bool = False
        self.accumulated_params: dict = {}
        self._json_processed_end: int = 0
        self._partial_emit_offset: int = 0
        self._partial_param_start: int = -1
        self._partial_emitted: str = ""
        self._partial_prefix: str = ""
        self._pending_brace: bool = False
        self._func_dup_detected: bool = False
        self._stream_last_fp: str | None = None
        self._stream_partial_name: str | None = None
        self._stream_partial_value: str | None = None
        # opt22d 第十一轮（方案 D）：暂存“未定”的 <function=> 开标签文本
        self._hold_buf: str = ""

        logger.info(
            "vLLM Successfully import tool parser %s !", self.__class__.__name__
        )

    @property
    def _fix_force_json(self) -> bool:
        """opt23 §11.22: fix=2 强制 JSON 路由（对齐 qwen3-coder）。"""
        return VLLM_QWEN3X_TOOL_FIX == 2

    @property
    def _fix_force_xml(self) -> bool:
        """opt23 §11.22: fix=3/4 强制 XML 路由（对齐 qwen3-coder）。"""
        return VLLM_QWEN3X_TOOL_FIX in (3, 4)

    def _compute_json_diff(self, current_json: str, prev_streamed: str) -> str:
        """Safe-prefix diff on JSON strings for incremental tool args
        （对齐 qwen3-coder _compute_json_diff）。"""
        if not prev_streamed:
            return current_json
        if not current_json:
            return ""
        common = find_common_prefix(prev_streamed, current_json)
        return current_json[len(common):]

    def _is_inside_json_string(self, json_text: str) -> bool:
        """对齐 qwen3-coder：判断文本结尾是否在 JSON 字符串值内。"""
        in_string = False
        escape_next = False
        for c in json_text:
            if escape_next:
                escape_next = False
                continue
            if c == '\\':
                escape_next = True
                continue
            if c == '"':
                in_string = not in_string
        return in_string

    def _unescape_display_chars(self, s: str) -> str:
        """对齐 qwen3-coder：把 \\n/\\t/\\r 转义序列还原为字面字符（前端显示）。"""
        result: list[str] = []
        i = 0
        while i < len(s):
            c = s[i]
            if c == '\\' and i + 1 < len(s):
                nxt = s[i + 1]
                if nxt == '\\' or nxt == '"':
                    result.append(c)
                    result.append(nxt)
                elif nxt == 'n':
                    result.append('\n')
                elif nxt == 't':
                    result.append('\t')
                elif nxt == 'r':
                    result.append('\r')
                else:
                    result.append(c)
                    result.append(nxt)
                i += 2
                continue
            result.append(c)
            i += 1
        return ''.join(result)

    @staticmethod
    def _escape_raw_control_chars(s: str) -> str:
        """模型可能把真实换行/控制字符（XML 换行习惯）直接输出进 JSON
        字符串值，导致 JSON 非法（Write content 参数丢失根因）。仅在
        字符串值内部把控制字符转义为 JSON 合法序列（\\n → \\\\n）。"""
        out = []
        i = 0
        n = len(s)
        in_str = False
        _esc_map = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}
        while i < n:
            c = s[i]
            if c == "\\":
                out.append(c)
                if i + 1 < n:
                    out.append(s[i + 1])
                    i += 2
                else:
                    i += 1
                continue
            if c == '"':
                in_str = not in_str
                out.append(c)
                i += 1
                continue
            if in_str and ord(c) < 0x20:
                out.append(_esc_map.get(c, "\\u%04x" % ord(c)))
                i += 1
                continue
            out.append(c)
            i += 1
        return "".join(out)

    def _handle_json_tool_streaming(
        self, current_text: str, delta_text: str
    ) -> DeltaMessage | None:
        """JSON 格式工具调用流式处理（对齐 qwen3-coder _handle_json_tool_streaming）。

        Supported format (single tool)::

            {"name": "Write", "arguments": {"file_path": "/x", "content": "..."}}

        每个 delta 是合法 JSON 前缀片段，适合 Anthropic input_json_delta 累积。
        """
        if not self.header_sent:
            _name_kw = current_text.find('"name"')
            if _name_kw == -1:
                return None
            _colon = current_text.find(":", _name_kw)
            if _colon == -1:
                return None
            _q1 = current_text.find('"', _colon + 1)
            if _q1 == -1:
                return None
            _q2 = current_text.find('"', _q1 + 1)
            if _q2 == -1:
                return None
            name = current_text[_q1 + 1:_q2]
            if not name:
                return None

            self.current_function_name = name
            self.current_tool_id = make_tool_call_id()
            self.header_sent = True
            self.in_function = True
            self.json_started = True
            self.accumulated_params = {}

            self.prev_tool_call_arr.append({"name": name, "arguments": "{}"})
            self.streamed_args_for_tool.append("")

            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=self.current_tool_index,
                        id=self.current_tool_id,
                        function=DeltaFunctionCall(name=name, arguments=""),
                        type="function",
                    )
                ]
            )

        _args_kw = current_text.find('"arguments"')
        if _args_kw == -1:
            return None
        _args_colon = current_text.find(":", _args_kw)
        if _args_colon == -1:
            return None
        _val_start = _args_colon + 1
        while _val_start < len(current_text) and current_text[_val_start] in " \t\n\r":
            _val_start += 1
        if _val_start >= len(current_text):
            return None
        if current_text[_val_start] != "{":
            return None

        _depth = 0
        _in_str = False
        _esc = False
        _json_end = -1
        for i in range(_val_start, len(current_text)):
            c = current_text[i]
            if _esc:
                _esc = False
                continue
            if c == "\\":
                _esc = True
                continue
            if c == '"' and not _esc:
                _in_str = not _in_str
                continue
            if _in_str:
                continue
            if c == "{":
                _depth += 1
            elif c == "}":
                _depth -= 1
                if _depth == 0:
                    _json_end = i + 1
                    break

        if _json_end > 0:
            current_args = current_text[_val_start:_json_end]
        else:
            current_args = current_text[_val_start:]

        # opt23: 模型把真实换行/控制字符（XML 换行习惯）直接输出进 JSON
        # 字符串值时 → 字符串值内转义为 JSON 合法序列，否则参数提取
        # 非法/截断（Write content 丢失、工具调用参数不完整根因）。
        current_args = self._escape_raw_control_chars(current_args)

        idx = self.current_tool_index
        prev = self.streamed_args_for_tool[idx] if idx < len(self.streamed_args_for_tool) else ""
        # opt23 §11.29: 对齐主线 hermes `_compute_args_diff`（前缀截取增量）。
        if len(current_args) <= len(prev):
            return None
        delta = current_args[len(prev):]
        if not delta:
            return None

        if idx < len(self.streamed_args_for_tool):
            self.streamed_args_for_tool[idx] += delta

        if (_json_end > 0 and idx < len(self.prev_tool_call_arr)
                and (_json_end == len(current_text)
                     or current_text[_json_end] == '}')):
            self.prev_tool_call_arr[idx]["arguments"] = current_args
            self.json_closed = True
            self.in_function = False
            self._json_processed_end = _json_end

        # opt23 §11.34: 对齐 qwen3-coder——之前为前端显示打字机效果把转义
        # 序列（\n \t \r）反转义为真实控制字符，但前端累积拼接后 JSON 非法
        # （真实换行在字符串值内），工具执行 InputValidationError。直接发射
        # 转义版增量（字面 \n），前端拼接即可 json.parse。
        display_delta = delta

        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=idx,
                    function=DeltaFunctionCall(arguments=display_delta),
                )
            ]
        )

    def _detect_json_output(self, current_text: str) -> bool:
        """检测 JSON 格式工具调用输出（对齐 coder 的 _json_tool_active 检测）。"""
        if self._fix_force_xml:
            return False
        _name = current_text.find('"name"')
        _args = current_text.find('"arguments"')
        _xml = current_text.find("<function=")
        return _name != -1 and _args != -1 and _name < _args and _xml == -1

    def _extract_tool_calls_json(self, model_output: str) -> list[ToolCall]:
        """opt23 §11.22: 提取 JSON 格式工具调用（<tool_call>{...}</tool_call> 或裸 JSON）。

        qwen3xml 是 expat XML parser，原生只解析 XML；模型输出 JSON 时由
        此 fallback 提取，保证 JSON 格式调用也能成功（fix=1/2/3/4 通用）。
        """
        raw_calls: list[dict] = []
        for _m in re.finditer(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            model_output,
            re.DOTALL,
        ):
            try:
                raw_calls.append(json.loads(_m.group(1)))
            except Exception:
                pass
        if not raw_calls:
            try:
                _start = model_output.find("{")
                if _start != -1:
                    _d = json.loads(model_output[_start:])
                    if isinstance(_d, dict) and (
                        "name" in _d or "arguments" in _d
                    ):
                        raw_calls.append(_d)
            except Exception:
                pass
        tcs: list[ToolCall] = []
        for _d in raw_calls:
            _name = _d.get("name")
            _args = _d.get("arguments")
            if _name is None:
                continue
            if not isinstance(_args, str):
                _args = json.dumps(_args, ensure_ascii=False)
            tcs.append(
                ToolCall(
                    id=make_tool_call_id(),
                    type="function",
                    function=FunctionCall(name=str(_name), arguments=_args),
                )
            )
        return tcs

    def _opt22d_should_force_all(self, request, raw_text: str = "") -> bool:
        """opt22d A/B/C：依“请求侧信号”判断是否整轮降级为文本。

        A `tool_choice == "none"`：客户端显式禁止调用 → 形似调用必为示例
        B 请求无工具且解析器无工具：无对象可调 → 形似调用必为示例
        [opt22 去重 2026-10-14] 实现已上提 utils（双份逐字重复），薄委托。
        """
        return _u_opt22d_should_force_all(request, self.tools)

    def _missing_required_args(self, tc) -> bool:
        """opt22e B：该调用的参数是否**缺少 schema 的必填字段**。

        参数无法解析（非法 JSON / 空）时也视为缺失 —— 真实调用的参数总能解析。
        `tools` 未提供或该工具无 `required` 声明时返回 False（**宽松兜底、不误伤**）。
        [opt22 去重 2026-10-14] 实现已上提 utils（双份重复；xml 侧函数内
        `import json as _json` 与 coder 模块级 json 逻辑同构），薄委托。
        """
        return _u_missing_required_args(self.tools, tc)

    def _opt22d_salvage_dropped_calls(self, model_output, tool_calls, request):
        """opt22d 防御纵深（2026-10-04）：退化输入下补捞被吞的相邻调用。

        主解析在**参数缺闭合**（如 `<parameter=city>Dallas\\n<parameter=state>`）或
        **标签间无换行**时，「扣住」逻辑可能把后续**完整**的 `<tool_call>` 一并吞掉。
        本方法从原始输出切出**结构完整**的块，用**独立**解析器重解析，凡其 function
        名未被已发出调用覆盖者，追加之。

        ⚠️ 三重保守门（防与 opt22d/opt22e 幽灵调用防御冲突）：
          ① 仅当**已有 ≥1 真实调用**时启用（纯正文/示例 ⇒ 主解析 0 调用 ⇒ 不补捞）；
          ② 无工具上下文（force_all）⇒ 不补捞；
          ③ 补捞到的调用同样过 `_missing_required_args` 必填兜底。
        ⚠️ 开关 `VLLM_QWEN3X_TOOL_SALVAGE`（默认 "1"；"0" 关闭）。
        """
        import os

        if os.environ.get("VLLM_QWEN3X_TOOL_SALVAGE", "1") == "0":
            return tool_calls
        if not tool_calls:
            return tool_calls
        # 仅在「**参数标签不平衡**」（开 > 闭）时补捞 —— depth bug 特征（见 coder 侧注）。
        # 平衡时不得补捞，否则会把「真调用之后的正文示例」误当独立调用（幽灵）。
        if model_output.count("<parameter=") <= model_output.count("</parameter>"):
            return tool_calls
        try:
            if self._opt22d_should_force_all(request, model_output):
                return tool_calls
        except Exception:
            pass
        seen = {
            tc.function.name
            for tc in tool_calls
            if getattr(tc, "function", None) and tc.function.name
        }
        for blk in re.findall(r"<tool_call>.*?</tool_call>", model_output, re.S):
            m = re.search(r"<function=([\w.\-]+)\s*>", blk)
            if not m:
                continue
            name = m.group(1)
            if name in seen:
                continue
            try:
                sub_parser = StreamingXMLToolCallParser()
                sub_parser.set_tools(self.tools)
                sub_parser._streaming_mode = False
                sub_parser._opt22d_force_all = False
                sub = sub_parser.parse_single_streaming_chunks(
                    _escape_unclosed_function_tags(blk)
                )
            except Exception:
                continue
            if sub is None or not sub.tool_calls:
                continue
            for t in sub.tool_calls:
                if not (t.function and t.function.name):
                    continue
                cand = ToolCall(
                    id=t.id,
                    type=t.type,
                    function=FunctionCall(
                        name=t.function.name,
                        arguments=t.function.arguments,
                    ),
                )
                if self._missing_required_args(cand):
                    continue
                tool_calls.append(cand)
                seen.add(t.function.name)
        return tool_calls

    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        # Narrowly target the one combination the base class mishandles: tools
        # present, tool_choice "auto", AND a response_format set. There,
        # to_sampling_params turns response_format into a grammar over the WHOLE
        # output while get_json_schema_from_tools() returns None for "auto" (no
        # tool grammar is built), so the answer-schema grammar leaves the literal
        # "<tool_call>" prefix outside the allowed token set and tool calls are
        # silently dropped.
        if not isinstance(request, ChatCompletionRequest) or not request.tools:
            return super().adjust_request(request)

        # Handle "auto" only. An ABSENT tool_choice is already normalized to
        # "auto" upstream when tools are present, so nothing is lost; explicit
        # None / "none" / "required" / named must keep base behaviour (serving
        # treats a falsy tool_choice as "none" and would otherwise surface the
        # raw XML branch as message content).
        if request.tool_choice != "auto":
            return super().adjust_request(request)

        # If a native structured-outputs constraint is already set, adding a
        # structural_tag would stack a second constraint and trip
        # StructuredOutputsParams' single-constraint check. Leave those to the
        # base path (response_format wins there, as it does today).
        if (
            request.structured_outputs is not None
            and not request.structured_outputs.all_constraints_none()
        ):
            return super().adjust_request(request)

        # Derive the answer-side format from response_format the way
        # to_sampling_params reads it.
        answer_format = None
        response_format = request.response_format
        if response_format is not None:
            if response_format.type == "json_object":
                answer_format = JSONSchemaFormat(json_schema={"type": "object"})
            elif response_format.type == "json_schema":
                json_schema = response_format.json_schema
                schema = json_schema.json_schema if json_schema is not None else None
                if schema is not None:
                    answer_format = JSONSchemaFormat(json_schema=schema)
            # "text" -> no constraint; "structural_tag" -> caller composed its
            # own tag. Both leave answer_format=None.

        # No answer constraint => nothing to union with; preserve today's
        # free-generation "auto" tool calling (tool text is parsed post-hoc).
        if answer_format is None:
            return super().adjust_request(request)

        reasoning = get_enable_structured_outputs_in_reasoning()

        def build_tag_json(tools):
            # Reuse the registry for the tool-call branch so the wire format
            # stays in sync with vLLM. tool_choice="required"
            # (TagsWithSeparatorFormat) not "auto" (TriggeredTagsFormat): on the
            # production tokenizer (vocab 248077) the "auto" shape leaves ~248075
            # tokens allowed at position 0, making the union degenerate and the
            # answer branch unreachable; "required" keeps the mask tight.
            tools_tag = get_model_structural_tag(
                model="qwen_3_5",
                tools=tools,
                tool_choice="required",
                reasoning=False,
            )
            assert tools_tag is not None
            # qwen_3_5 hardcodes separator="" in the registry, but the chat
            # template emits "\n<tool_call>" between parallel calls, so a real
            # </tool_call>\n<tool_call> sequence would otherwise be rejected. Fix
            # the separator at this use site (do NOT edit the shared registry).
            tools_format = tools_tag.format.model_copy(update={"separator": "\n"})
            # Allow the model's natural leading whitespace: with
            # enable_in_reasoning=False the grammar binds right after "</think>"
            # while the template emits "\n</think>\n\n", so the first real token
            # is "\n\n". Without this the tool-vs-answer decision would be made on
            # a masked token. Whitespace alone cannot satisfy the OrFormat, so it
            # cannot terminate the match.
            union = SequenceFormat(
                elements=[
                    OptionalFormat(content=RegexFormat(pattern="[ \\n\\t]{1,8}")),
                    OrFormat(elements=[tools_format, answer_format]),
                ]
            )
            if reasoning:
                prefix = SequenceFormat(
                    elements=[
                        TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                        ConstStringFormat(value="\n\n"),
                    ]
                )
                union = SequenceFormat(elements=[prefix, union])
            return json.dumps(StructuralTag(format=union).model_dump())

        def compiles(tag_json):
            try:
                Grammar.from_structural_tag(tag_json)
            except Exception as exc:
                return exc
            return True

        # Self-validate: the union compiles every tool's parameters through the
        # qwen_xml converter, which today's "auto" path never does. A schema
        # feature xgrammar cannot express (e.g. regex lookahead) would raise and,
        # under backend="auto", fall back to the guidance backend which crashes
        # on v2 structural tags. Degrade gracefully instead of erroring.
        tag_json = build_tag_json(request.tools)
        outcome = compiles(tag_json)
        if outcome is not True:
            logger.warning(
                "Qwen3XMLToolParser: tool schemas are not xgrammar-compatible "
                "(%s); retrying with unconstrained tool arguments.",
                outcome,
            )
            # Relax every tool's parameters to unconstrained. Not lossy vs. the
            # status quo: today's "auto" path does not constrain tool arguments
            # at all, and the XML call structure stays constrained.
            relaxed_tools = [tool.model_copy(deep=True) for tool in request.tools]
            for tool in relaxed_tools:
                tool.function.parameters = None
            tag_json = build_tag_json(relaxed_tools)
            outcome = compiles(tag_json)
            if outcome is not True:
                logger.warning(
                    "Qwen3XMLToolParser: structural tag still uncompilable (%s); "
                    "falling back to default tool calling.",
                    outcome,
                )
                return super().adjust_request(request)

        # Match AbstractToolParser.adjust_request's structural-tag style.
        if request.structured_outputs is None:
            request.structured_outputs = StructuredOutputsParams(
                structural_tag=tag_json,
            )
        else:
            request.structured_outputs.structural_tag = tag_json
        # Mandatory: otherwise to_sampling_params derives a second constraint
        # from response_format and the single-constraint validation raises.
        request.response_format = None
        return request

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:
        self.parser.reset_streaming_state()
        # Reset tool call tracking arrays for new extraction
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self.parser.set_tools(self.tools)
        self.parser._opt22d_force_all = self._opt22d_should_force_all(request, model_output)
        # opt23 §11.22: JSON 路由（fix=2 强制 / fix=1 检测）→ 内部 JSON 提取
        # （expat 对 JSON 输出会产生无效 XML tool_call，需绕过）。
        if self._fix_force_json or self._detect_json_output(model_output):
            json_tcs = self._extract_tool_calls_json(model_output)
            if json_tcs:
                self.prev_tool_call_arr = [
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in json_tcs
                ]
                return ExtractedToolCallInformation(
                    tool_calls=json_tcs,
                    tools_called=True,
                    content=None,
                )
            return ExtractedToolCallInformation(
                tool_calls=[],
                tools_called=False,
                content=model_output,
            )
        # opt22d 第十轮：非流式下全文已知 ⇒ 先摘掉**未闭合**的孤立 <function=>
        # （xml 侧 L-D 的结构条件），避免展示性文本被当成真实调用执行。
        # 参数体内「文档示例」（第十二轮缺陷B）由 `_preprocess_xml_chunk` 的
        # peek 结构判定统一处理——非流式与此共用同一条逐元素路径，无需在此分叉。
        # 非流式**消息已完整** ⇒ 不允许"扣住 `</parameter>` 等下一元素"。
        self.parser._streaming_mode = False
        _esc_model = _escape_unclosed_function_tags(model_output)
        try:
            result = self.parser.parse_single_streaming_chunks(_esc_model)
            # opt22d 第十二轮（缺陷B·续）：非流式下**消息已完整** ⇒ 若仍有未闭合的
            # function（真收尾被文档示例干扰而未放行，如参数体内出现孤立的
            # `<parameter=` 使计数失衡），强制收尾，避免整个调用丢失。
            # 流式做不到这一点（不知道本轮是否还有后续 delta），故仅非流式。
            _dn = len(self.parser.deltas)
            if self.parser.current_function_name:
                if self.parser.current_param_name:
                    self.parser._end_element("parameter")
                self.parser._end_element("function")
                _extra = self.parser._merge_new_deltas_to_single_response(_dn)
                # 收尾只补 JSON 闭合片段（如 `"}` / `}`），追加到最后一个调用上
                _suffix = "".join(
                    (t.function.arguments or "")
                    for t in (_extra.tool_calls or [])
                    if t.function
                ) if _extra else ""
                if _suffix and result is not None and result.tool_calls:
                    _last = result.tool_calls[-1]
                    if _last.function:
                        _last.function.arguments = (
                            _last.function.arguments or ""
                        ) + _suffix
        except Exception:
            result = None
        # opt22d（L-E）：消息已完整，强制刷出缓冲里残留的未闭合尾部片段
        _tail = self.parser._flush_text_buffer_forced()
        if _tail:
            if result is None:
                result = DeltaMessage(content=_tail)
            else:
                result.content = (result.content or "") + _tail
        if result is None or not result.tool_calls:
            # opt23 §11.22: expat 未解析出 XML tool_call（模型输出 JSON 或
            # expat 异常）→ 内部 JSON 提取兜底。
            json_tcs = self._extract_tool_calls_json(model_output)
            if json_tcs:
                self.prev_tool_call_arr = [
                    {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                    for tc in json_tcs
                ]
                return ExtractedToolCallInformation(
                    tool_calls=json_tcs,
                    tools_called=True,
                    content=None,
                )
            return ExtractedToolCallInformation(
                tool_calls=[],
                tools_called=False,
                content=result.content if result else _esc_model,
            )
        else:
            tool_calls = []
            for tool_call in result.tool_calls:
                if tool_call.function and tool_call.function.name:
                    tool_calls.append(
                        ToolCall(
                            id=tool_call.id,
                            type=tool_call.type,
                            function=FunctionCall(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        )
                    )

                    # Update tool call tracking arrays for compatibility
                    tool_index = (
                        tool_call.index
                        if tool_call.index is not None
                        else len(self.prev_tool_call_arr) - 1
                    )

                    # Ensure we have enough entries in our tracking arrays
                    while len(self.prev_tool_call_arr) <= tool_index:
                        self.prev_tool_call_arr.append({"name": "", "arguments": ""})
                    while len(self.streamed_args_for_tool) <= tool_index:
                        self.streamed_args_for_tool.append("")

                    # Update tool call information
                    self.prev_tool_call_arr[tool_index]["name"] = (
                        tool_call.function.name
                    )
                    self.prev_tool_call_arr[tool_index]["arguments"] = (
                        tool_call.function.arguments
                    )

                    # Update streamed arguments
                    if tool_call.function.arguments:
                        self.streamed_args_for_tool[tool_index] = (
                            tool_call.function.arguments
                        )

            # opt22e B：**必填参数兜底**。真实调用必然带齐 schema 的 required
            # 参数；缺失（或参数非法 JSON）⇒ 不是真实调用（多为正文里的示例）。
            # 与 A（上下文判据）互补，且**不依赖枚举形态**。
            tool_calls = [
                tc for tc in tool_calls if not self._missing_required_args(tc)
            ]

            # opt22d 防御纵深（2026-10-04）：退化输入下补捞被吞的相邻调用
            tool_calls = self._opt22d_salvage_dropped_calls(
                model_output, tool_calls, request
            )

            return ExtractedToolCallInformation(
                tool_calls=tool_calls,
                tools_called=len(tool_calls) > 0,
                content=result.content,
            )

    def flush_streaming_finish(self):
        """opt22d 第十二轮（缺陷B）：流式结束时的**强制收尾**。

        参数含示例时，收尾标签被"滚动扣住"、一直没喂 expat ⇒ 调用无法完成。
        消息此刻已完整 ⇒ 丢弃扣住者（它们是真收尾，不是内容），强制闭合
        expat 的未完成元素，把收尾片段（如 `"}`）作为增量下发。

        **参数正文此前已即时下发**（滚动扣住只延迟几十字节），故此处只需补收尾。

        Returns:
            ``DeltaMessage``（仅含收尾增量）；无可收尾内容时返回 None。
        """
        p = self.parser
        if not p.current_function_name:
            return None
        _dn = len(p.deltas)
        p._hold_tags = []          # 真收尾 ⇒ 丢弃（绝不进参数值）
        p._pending_param_close = False
        if p.current_param_name:
            p._end_element("parameter")
        if p.current_function_name:
            p._end_element("function")
        if p.current_call_id:
            p._end_element("tool_call")
        return p._merge_new_deltas_to_single_response(_dn)

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        # opt22d 第十二轮（缺陷B）：流式允许"扣住 `</parameter>` 等下一元素"
        self.parser._streaming_mode = True
        # 参数含示例 ⇒ 收尾标签被当内容 ⇒ 需在 finish 时强制收尾
        if self.parser._param_example_state.get("has_example") and (
            self.parser.current_function_name
        ):
            self._needs_finish_flush = True
        if not previous_text:
            self.parser.reset_streaming_state()
            self.parser._streaming_mode = True
            # Reset tool call tracking arrays for new streaming session
            self.prev_tool_call_arr = []
            self.streamed_args_for_tool = []
            self.parser.set_tools(self.tools)
            self.parser._opt22d_force_all = self._opt22d_should_force_all(request, current_text)
            # Reset JSON 路由状态（对齐 qwen3-coder _reset_streaming_state）
            self.current_tool_index = 0
            self.header_sent = False
            self.current_tool_id = None
            self.current_function_name = None
            self.in_function = False
            self.json_started = False
            self.json_closed = False
            self.accumulated_params = {}
            self._json_processed_end = 0

        # opt23 §11.22: JSON 路由（fix=2 强制 / fix=1 检测）→ 内部
        # _handle_json_tool_streaming（对齐 qwen3-coder 的 JSON 真流式增量）。
        # fix=2 强制 JSON 但模型实际输出 XML（含 <function= 标记）时兜底走
        # expat XML 路径——「xml 也强制走 json」指最终输出统一 JSON tool_calls。
        # fix=2 强制需在 JSON 特征（"name"/"arguments"）出现后才进入——
        # 无条件拦截会把纯文本正文吞进 JSON 工具解析（think 后无正文根因）。
        if self._fix_force_json:
            if current_text.find("<function=") == -1:
                # opt23 §11.34: 对齐 qwen3-coder——<tool_call> 标记（模板 json
                # 分支引导模型输出 <tool_call> 包裹 JSON）或 JSON 特征
                # （"name"/"arguments"）出现即激活 JSON 路由；handler 返回
                # None 时同样 return None（BLOCK expat XML 双路径——双路径
                # 竞争导致增量不一致/参数非法）。<function= 表示模型实际
                # 输出 XML，不强制 JSON（走 expat XML 解析）。
                _json_name = current_text.find('"name"')
                _json_args = current_text.find('"arguments"')
                _has_marker = current_text.find("<tool_call>") != -1
                if (_has_marker or (_json_name != -1 and _json_args != -1
                                    and _json_name < _json_args)):
                    return self._handle_json_tool_streaming(
                        current_text, delta_text
                    ) or None
        elif self._detect_json_output(current_text):
            return self._handle_json_tool_streaming(
                current_text, delta_text
            ) or None

        # Model sometimes outputs separately causing delta_text to be empty.
        # If there were tool_calls before and all current tool_calls have ended,
        # return an empty tool_call for outer streaming output
        # to correctly output tool_call field
        if not delta_text and delta_token_ids:
            # opt22d 第十一轮（方案 D）：消息结束——此前暂存的 <function=> 若始终
            # 没等到判据，即为**孤立标签**（展示性），转义后释放为正文。
            # 不释放会被永久扣留 ⇒ 用户看不到该文本（与第九轮 coder 同款缺陷）。
            if self._hold_buf:
                _held = _XML_FN_OPEN_RE.sub(_esc_open_fn, self._hold_buf)
                self._hold_buf = ""
                if _held:
                    return DeltaMessage(content=_held)
            open_calls = current_text.count(
                self.parser.tool_call_start_token
            ) - current_text.count(self.parser.tool_call_end_token)
            if (
                open_calls == 0
                and self.parser.tool_call_index > 0
                or not self.parser.tool_call_index
                and current_text
            ):
                return DeltaMessage(content="")
            return None

        # opt22d 第十一轮（方案 D）：<function=> 开标签到达时**无法预知**其后是否
        # 跟 <parameter=/</function>。此前 xml 直接 autofill 出 <tool_call> 并产出
        # 调用（_start_element），孤立标签因此被当真实调用（实测 EXH：'\n\n
        # <function=Read>' → calls=['Read']、正文被吞）。
        # 现改为**看判据再决定**：判据未到齐则暂存（不喂给 expat），下一轮连同
        # 新文本一起喂；若后续是非空白文本 ⇒ 判为展示性 → 转义后喂；到消息结束
        # 仍无判据 ⇒ 转义释放（见下方 EOS 分支）。
        # 注意：判定基于**完整 current_text**，故虽在流式路径也能看到全上下文。
        _feed = self._hold_buf + delta_text
        self._hold_buf = ""
        _fed_abs = len(current_text) - len(_feed)  # 已喂给 expat 的绝对下标
        _hs = _hold_start(current_text)
        if _hs is not None and _hs >= _fed_abs:
            _off = _hs - _fed_abs
            _tail = _feed[_off:]
            _feed = _feed[:_off]
            _dec = _unclosed_fn_decision(current_text)
            if _dec is not None and _dec[1] == "escape":
                # 判据明确否决（后面已是确定的其它文本）→ 就地转义后立即喂
                _feed += _XML_FN_OPEN_RE.sub(_esc_open_fn, _tail, count=1)
            else:
                self._hold_buf = _tail  # 判据未到齐 → 继续扣，下一轮再判

        # Parse the delta text and get the result
        delta = self.parser.parse_single_streaming_chunks(_feed)

        # Update tool call tracking arrays based on incremental parsing results
        if delta and delta.tool_calls:
            for tool_call in delta.tool_calls:
                if tool_call.function:
                    tool_index = (
                        tool_call.index
                        if tool_call.index is not None
                        else len(self.prev_tool_call_arr) - 1
                    )

                    # Ensure we have enough entries in our tracking arrays
                    while len(self.prev_tool_call_arr) <= tool_index:
                        self.prev_tool_call_arr.append({"name": "", "arguments": ""})
                    while len(self.streamed_args_for_tool) <= tool_index:
                        self.streamed_args_for_tool.append("")

                    # Update tool name if provided
                    if tool_call.function.name:
                        self.prev_tool_call_arr[tool_index]["name"] = (
                            tool_call.function.name
                        )

                    # Update arguments incrementally
                    if tool_call.function.arguments is not None:
                        # Concatenate the incremental arguments
                        # to the existing streamed arguments
                        self.prev_tool_call_arr[tool_index]["arguments"] += (
                            tool_call.function.arguments
                        )
                        self.streamed_args_for_tool[tool_index] += (
                            tool_call.function.arguments
                        )
        if delta.content is None and not delta.tool_calls and delta.reasoning is None:
            # If no content and no tool calls, return None to indicate no update
            return None
        return delta
