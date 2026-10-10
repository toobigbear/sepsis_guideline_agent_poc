# -*- coding: utf-8 -*-
"""证据缺口驱动的多轮检索（论文 Algorithm 1）。

Agent 每轮观察：问题 + 历史工具调用与结果 + 当前证据缺口，
决定"证据是否充分"：充分则作答，不足则选择一个工具补齐缺口。
结构上下文（目录）始终留在存储侧按需暴露，不进 prompt。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
from .llm import LLM, approx_tokens
from .prompts import (RETRIEVER_VARIANTS, RETRIEVER_ROUND, ANSWER_SYSTEM)
from .tools import ToolKit


@dataclass
class RunTrace:
    question: str
    steps: List[Dict] = field(default_factory=list)      # 检索轨迹（工具调用历史）
    evidence: Dict[str, str] = field(default_factory=dict)  # uri -> 已读内容
    prompt_tokens: int = 0
    llm_calls: int = 0

    def add_tokens(self, text: str) -> None:
        self.prompt_tokens += approx_tokens(text)

    def read_uris(self) -> List[str]:
        return list(self.evidence.keys())


class EvidenceGapRetriever:
    """VikingRAG 基础版：多轮证据缺口驱动检索 + 生成。"""

    def __init__(self, toolkit: ToolKit, budget: int = None,
                 history_compress: int = None, prompt_variant: str = "A",
                 executor=None, require_evidence: bool = True, audit=None) -> None:
        self.tk = toolkit
        self.budget = budget if budget is not None else config.DEFAULT_HARNESS["round_budget"]
        self.history_compress = (history_compress if history_compress is not None
                                 else config.DEFAULT_HARNESS["history_compress"])
        self.system_prompt = RETRIEVER_VARIANTS.get(prompt_variant, RETRIEVER_VARIANTS["A"])
        # Harness 钩子：executor(decision, round_no) -> (result, executed)
        self.executor = executor
        self.require_evidence = require_evidence
        self.audit = audit
        self.blocked_reason = ""

    # ---------------- 一轮决策 ----------------

    def _decide(self, question: str, trace: RunTrace, gap_hint: str) -> Dict:
        history = self._format_history(trace)
        prompt = RETRIEVER_ROUND.format(question=question, history=history,
                                        gap=gap_hint or "（首轮：尚无证据）")
        msgs = [{"role": "system", "content":
                 self.system_prompt.format(tools="\n".join(self.tk.catalog()))},
                {"role": "user", "content": prompt}]
        trace.add_tokens(msgs[0]["content"] + prompt)
        trace.llm_calls += 1
        decision = LLM.chat_json(msgs, max_tokens=400)
        trace.add_tokens(str(decision))
        return decision

    def _format_history(self, trace: RunTrace) -> str:
        parts = []
        for i, s in enumerate(trace.steps, 1):
            res = s["result"]
            if len(res) > 500:
                res = res[:500] + "…（截断）"
            parts.append(f"{i}. {s['tool']}({s['args']})\n【工具结果-{s['tool']}】{res}")
        text = "\n".join(parts)
        if len(text) > self.history_compress:  # 历史压缩：交互历史也计入 token
            text = "…（早期历史已压缩）\n" + text[-self.history_compress:]
        return text or "（无）"

    # ---------------- 主循环 ----------------

    def run(self, question: str, seed_uris: Optional[List[str]] = None,
            verbose: bool = False) -> RunTrace:
        self.blocked_reason = ""
        trace = RunTrace(question=question)
        # 经验边复用：先把历史轨迹命中的 URI 作为种子读入（VikingRAG-E）
        for uri in (seed_uris or []):
            result = self.tk.read(uri)
            trace.steps.append({"tool": "Read", "args": {"uri": uri},
                                "result": result, "from_edge": True})
            trace.evidence[uri] = result
            trace.add_tokens(result)
            if verbose:
                print(f"  [经验边] Read {uri}")

        gap = ""
        for rnd in range(self.budget):
            decision = self._decide(question, trace, gap)
            action = decision.get("action", "tool_call")
            if action == "answer":
                if self.require_evidence and not trace.evidence:
                    # Harness 策略：无证据拒绝作答
                    self.blocked_reason = "无证据作答被拒绝"
                    if self.audit:
                        self.audit.log("HARNESS_BLOCK", reason=self.blocked_reason,
                                       round=rnd + 1)
                    trace.steps.append({"tool": "HARNESS_BLOCK", "args": {},
                                        "result": self.blocked_reason})
                    break
                trace.steps.append({"tool": "FINISH", "args": {},
                                    "result": decision.get("think", "")})
                break
            tool = decision.get("tool", "Search")
            args = decision.get("args", {}) or {}
            if tool == "Read" and args.get("uri") == "auto-pick":  # 离线引擎的占位
                args = {"uri": self._auto_pick(trace)}
            if self.executor is not None:
                result, executed = self.executor(
                    {"action": "tool_call", "tool": tool, "args": args}, rnd + 1)
            else:
                result, executed = self.tk.dispatch(tool, args), True
            trace.steps.append({"tool": tool, "args": args, "result": result,
                                "blocked": not executed})
            trace.add_tokens(result)
            if executed and tool == "Read":
                trace.evidence[args.get("uri", "")] = result
            gap = decision.get("think", "")
            if verbose:
                print(f"  [第{rnd + 1}轮] {tool} {args} -> {result[:60]}…")
        return trace

    def _auto_pick(self, trace: RunTrace) -> str:
        """离线模式：从最近一次 Search 结果里挑未读的首条。"""
        for s in reversed(trace.steps):
            if s["tool"] == "Search":
                for line in s["result"].splitlines():
                    if line.startswith("["):
                        uri = line.split("]")[1].strip().split(" ")[0]
                        if uri not in trace.evidence:
                            return uri
        return "viking://"

    # ---------------- 生成 ----------------

    def answer(self, question: str, trace: RunTrace) -> str:
        evidence_text = "\n\n".join(trace.evidence.values()) or "（无）"
        prompt = f"【问题】{question}\n\n【证据】\n{evidence_text}\n\n请作答。"
        msgs = [{"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": prompt}]
        trace.add_tokens(msgs[0]["content"] + prompt)
        trace.llm_calls += 1
        ans = LLM.chat(msgs, max_tokens=1800)
        trace.add_tokens(ans)
        return ans
