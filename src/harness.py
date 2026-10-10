# -*- coding: utf-8 -*-
"""Harness：LLM 决策的结构化校验 + 工具网关 + 证据引用校验。

对应 POC 的"管得住"目标，全部判定由确定性代码执行，LLM 只填问卷：
- ToolCallValidator：LLM 想调的每个工具调用先过 schema 校验
  （工具白名单、参数类型、URI 命名空间与越界、长度上限）；
- ToolGateway：统一入口，所有调用落审计；非法调用不执行、返回纠错反馈；
- EvidencePolicy：最终答案里的每条 [viking://...] 引用必须来自
  本次实际 Read 的证据集合 —— 编造引用直接拦截；
- 无证据拒绝作答：answer 前没有读过任何证据 → 拒绝生成。

可被 Meta-Harness 调参的部分在 config.SEARCH_SPACE；
本模块的校验规则本身属于安全层，冻结。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from vikingrag.storage import Storage
from vikingrag.tools import ToolKit

URI_RE = re.compile(r"^viking://[\w\-./#]+$")


@dataclass
class HarnessConfig:
    """调度与校验参数（Meta-Harness 的搜索对象）。安全规则不在这里。"""
    top_k: int = 5
    round_budget: int = 15
    history_compress: int = 3500
    experience_min_sim: float = 1.5
    experience_min_overlap: int = 2
    seed_reads: int = 6
    one_round_reads: int = 2
    evidence_min_reads: int = 2
    numeric_threshold: float = 0.85
    prompt_variant: str = "A"
    answer_temperature: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HarnessConfig":
        base = cls().as_dict()
        base.update({k: v for k, v in d.items() if k in base})
        return cls(**base)


# ---------------------------------------------------------------- 决策校验

TOOL_SCHEMAS: Dict[str, Dict[str, type]] = {
    "Search": {"query": str, "scope": str, "k": int},
    "List": {"uri": str},
    "Grep": {"pattern": str, "scope": str},
    "Read": {"uri": str},
}
REQUIRED_ARGS = {"Search": ["query"], "List": ["uri"],
                 "Grep": ["pattern"], "Read": ["uri"]}
ALLOWED_ACTIONS = {"tool_call", "answer"}


class ToolCallValidator:
    """Harness 第一层：LLM 决策的结构化校验（Schema/命名空间/越界）。"""

    @staticmethod
    def _valid_uri(uri: str) -> bool:
        return (isinstance(uri, str) and len(uri) <= 200
                and bool(URI_RE.match(uri)) and ".." not in uri)

    @classmethod
    def validate(cls, decision: Dict[str, Any]) -> Tuple[bool, Dict, str]:
        """返回 (是否放行, 净化后的参数, 拒绝原因)。"""
        if not isinstance(decision, dict):
            return False, {}, "决策不是 JSON 对象"
        action = decision.get("action")
        if action not in ALLOWED_ACTIONS:
            return False, {}, f"非法 action: {action!r}（只允许 {sorted(ALLOWED_ACTIONS)}）"
        if action == "answer":
            return True, {}, ""
        tool = decision.get("tool")
        if tool not in TOOL_SCHEMAS:
            return False, {}, f"非法工具: {tool!r}（白名单: {sorted(TOOL_SCHEMAS)}）"
        args = decision.get("args")
        if not isinstance(args, dict):
            return False, {}, "args 必须是对象"
        schema = TOOL_SCHEMAS[tool]
        clean: Dict[str, Any] = {}
        for k, v in args.items():
            if k not in schema:
                return False, {}, f"{tool} 不接受参数 {k!r}"
            if not isinstance(v, schema[k]):
                return False, {}, f"{tool}.{k} 类型错误（期望 {schema[k].__name__}）"
            clean[k] = v
        for req in REQUIRED_ARGS[tool]:
            if req not in clean or not str(clean[req]).strip():
                return False, {}, f"{tool} 缺少必填参数 {req}"
        for key in ("uri", "scope"):
            if key in clean and not cls._valid_uri(clean[key]):
                return False, {}, f"{key} 越界或非法: {clean[key]!r}（必须 viking:// 命名空间内）"
        for key in ("query", "pattern"):
            if key in clean and len(clean[key]) > 200:
                return False, {}, f"{key} 超长（>200 字符）"
        if "k" in clean and not (1 <= clean["k"] <= 10):
            return False, {}, "k 必须在 1~10"
        return True, clean, ""


class EvidencePolicy:
    """Harness 第二层：证据与引用策略（无证据不作答、引用必须真实）。"""

    CITE_RE = re.compile(r"\[?\s*(viking://[^\s\]\)）>\"'\]]+)\s*\]?")

    @classmethod
    def verify_citations(cls, answer: str,
                         evidence_uris: List[str]) -> List[str]:
        """答案中每条引用都必须在本次 Read 过的证据集合内，否则返回违规列表。"""
        violations = []
        for m in cls.CITE_RE.finditer(answer):
            cited = m.group(1).rstrip("。，；、).,;]")
            if not any(cited == e or e.startswith(cited) or cited.startswith(e)
                       for e in evidence_uris):
                violations.append(cited)
        return violations

    @staticmethod
    def has_evidence(evidence_uris: List[str]) -> bool:
        return len(evidence_uris) > 0


# ---------------------------------------------------------------- 工具网关

class ToolGateway:
    """LLM/Agent 与工具之间的唯一通道：校验 -> 审计 -> 执行 -> 审计。"""

    HARD_ROUND_CAP = 25  # 冻结上限：Meta-Harness 的 round_budget 不得超过此值

    def __init__(self, storage: Storage, audit=None,
                 toolkit: Optional[ToolKit] = None) -> None:
        self.st = storage
        self.tk = toolkit or ToolKit(storage)
        self.validator = ToolCallValidator()
        self.audit = audit
        self._fail_streak: Dict[str, int] = {}

    def _log(self, event: str, **kw: Any) -> None:
        if self.audit:
            self.audit.log(event, **kw)

    def execute(self, decision: Dict[str, Any], round_no: int) -> Tuple[str, bool]:
        """校验并执行一个 LLM 决策。
        返回 (结果文本, 是否实际执行了工具)。被拦时结果是纠错反馈。"""
        if round_no > self.HARD_ROUND_CAP:
            self._log("HARNESS_BLOCK", reason="超过冻结轮数上限", round=round_no)
            return "【Harness】已达轮数硬上限，请立即作答", False

        ok, args, reason = self.validator.validate(decision)
        tool = decision.get("tool", "?") if isinstance(decision, dict) else "?"
        if not ok:
            self._fail_streak[tool] = self._fail_streak.get(tool, 0) + 1
            self._log("HARNESS_BLOCK", tool=tool, reason=reason,
                      decision=str(decision)[:300])
            if self._fail_streak[tool] >= 3:
                self._fail_streak[tool] = 0
                return (f"【Harness】{tool} 已连续 3 次被拒（{reason}）。"
                        f"请改换工具或作答。", False)
            return f"【Harness】工具调用被拒绝：{reason}。请修正后重试。", False

        self._fail_streak[tool] = 0
        self._log("TOOL_CALL", tool=tool, args=args, round=round_no)
        result = self.tk.dispatch(tool, args)
        self._log("TOOL_RESULT", tool=tool, result=result[:300], round=round_no)
        return result, True
