# -*- coding: utf-8 -*-
"""医疗指南问答评测器：确定性指标，不用 LLM 当裁判。

指标（全部代码计算，可复现、可在评审会上逐条回放）：
- evidence_recall  金牌证据 URI 的命中比例（按章节前缀匹配）
- citation_valid   答案引用的 URI 是否全部来自本次实际 Read 的证据集合
                   （Harness 强制：编造的引用会被拦截并记 0）
- keyword_hit      金牌关键词在答案中的覆盖比例
- refusal_correct  红线题是否正确拒绝（期望拒绝的题）
- score            加权总分（含 token 惩罚项）
- tokens/rounds/llm_calls/path  效率指标
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import config
from vikingrag.experience import VikingRAGEPlus
from vikingrag.storage import Storage

from .audit import AuditLogger
from .guardrails import Guardrail
from .harness import EvidencePolicy, HarnessConfig, ToolGateway


@dataclass
class ItemRecord:
    item_id: str
    category: str
    split: str
    path: str = ""
    evidence_recall: float = 0.0
    citation_valid: float = 1.0
    keyword_hit: float = 0.0
    refusal_correct: float = 0.0
    score: float = 0.0
    tokens: int = 0
    rounds: int = 0
    llm_calls: int = 0
    blocked: str = ""
    answer_head: str = ""


class EvalRunner:
    """在指定 Harness 配置下跑一批问答，输出逐题记录与汇总。"""

    def __init__(self, storage: Storage, harness: HarnessConfig,
                 audit: Optional[AuditLogger] = None,
                 wipe_experience: bool = True,
                 channel: str = "vikingrag") -> None:
        self.st = storage
        self.h = harness
        self.channel = channel
        self.audit = audit or AuditLogger()
        self.baseline = max(1, storage.full_catalog_tokens())
        self._baseline_channel = None
        if channel != "vikingrag":
            from .baselines import PlainRAGChannel, STAIRChannel, VectorIndex
            index = VectorIndex(storage)
            index.build()
            if channel == "plain_rag":
                self._baseline_channel = PlainRAGChannel(
                    storage, top_k=harness.top_k, index=index)
            else:
                self._baseline_channel = STAIRChannel(storage, index=index)
        if wipe_experience:  # 每个候选配置独立评测，避免经验边互相污染
            storage.conn.execute("DELETE FROM experience_edges")
            storage.conn.commit()

    # ---------------- 单题 ----------------

    def run_item(self, item: Dict[str, Any]) -> ItemRecord:
        rec = ItemRecord(item_id=item["id"], category=item["category"],
                         split=item.get("split", "search"))
        q = item["question"]

        # 红线题：走护栏拒绝通道，不进检索（三通道同一行为）
        if item.get("expect_refusal"):
            if Guardrail.is_boundary_request(q):
                rec.path = "护栏拒绝（红线拦截）"
                rec.refusal_correct = 1.0
                rec.score = 1.0
                self.audit.log("REFUSAL", item=item["id"], question=q[:80])
                return rec
            rec.blocked = "应拒绝但未识别为越界请求"
            return rec

        if self.channel == "vikingrag":
            # 题目间隔离：清空经验边，保证每题测的是检索本身（经验边价值单独演示）
            self.st.conn.execute("DELETE FROM experience_edges")
            self.st.conn.commit()
            gateway = ToolGateway(self.st, audit=self.audit)
            agent = VikingRAGEPlus(self.st, params=self.h.as_dict(),
                                   executor=gateway.execute, audit=self.audit)
            ans, trace, path, verdict = agent.run(q)
            rec.path = path
            rec.tokens = trace.prompt_tokens
            rec.rounds = len(trace.steps)
            rec.llm_calls = trace.llm_calls
            evidence_uris = trace.read_uris()
        else:
            res = self._baseline_channel.run(q)
            rec.path = res.path
            rec.tokens = res.tokens
            rec.rounds = len(res.read_log)
            rec.llm_calls = res.llm_calls
            ans, evidence_uris = res.answer, res.evidence_uris

        # ---- Harness 第二层：引用真实性校验（编造引用直接拦截）----
        violations = EvidencePolicy.verify_citations(ans, evidence_uris)
        if violations:
            rec.citation_valid = 0.0
            rec.blocked = f"编造引用被拦截: {violations}"
            self.audit.log("HARNESS_BLOCK", item=item["id"],
                           reason=rec.blocked)
            ans = "【Harness】答案包含未读取证据的引用，已拦截。" + ans[:0]

        # ---- 执行层：硬追加免责声明，再过护栏终检（违禁词 + 免责声明）----
        if config.DISCLAIMER not in ans:
            ans = ans + "\n" + config.DISCLAIMER
        gr = Guardrail.final_check(ans)
        if not gr.ok:
            rec.blocked = (rec.blocked + "；" if rec.blocked else "") + gr.reasons()

        # ---- 确定性指标 ----
        gold = item.get("gold_uris", [])
        if gold:
            hit = sum(1 for g in gold
                      if any(self._uri_match(g, e) for e in evidence_uris))
            rec.evidence_recall = hit / len(gold)
        kws = item.get("keywords", [])
        if kws:
            rec.keyword_hit = sum(1 for k in kws if k in ans) / len(kws)

        rec.score = self.score(rec)
        rec.answer_head = ans.strip().splitlines()[0][:60] if ans.strip() else ""
        return rec

    @staticmethod
    def _uri_match(gold: str, read: str) -> bool:
        return read == gold or read.startswith(gold) or gold.startswith(read)

    def score(self, rec: ItemRecord) -> float:
        s = (0.45 * rec.evidence_recall + 0.35 * rec.keyword_hit
             + 0.20 * rec.citation_valid + rec.refusal_correct)
        s -= 0.05 * min(1.0, rec.tokens / self.baseline)
        if rec.blocked and "编造引用" in rec.blocked:
            s = min(s, 0.2)  # 编造引用是重罪：封顶 0.2
        return round(max(0.0, s), 4)

    # ---------------- 批量 ----------------

    def run(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        records = [self.run_item(it) for it in items]
        return summarize(records)


def summarize(records: List[ItemRecord]) -> Dict[str, Any]:
    n = max(1, len(records))
    return {
        "n": len(records),
        "score": round(sum(r.score for r in records) / n, 4),
        "evidence_recall": round(sum(r.evidence_recall for r in records) / n, 4),
        "keyword_hit": round(sum(r.keyword_hit for r in records) / n, 4),
        "citation_valid": round(sum(r.citation_valid for r in records) / n, 4),
        "refusal_correct": round(sum(r.refusal_correct for r in records) / n, 4),
        "tokens_avg": int(sum(r.tokens for r in records) / n),
        "rounds_avg": round(sum(r.rounds for r in records) / n, 1),
        "blocked_n": sum(1 for r in records if r.blocked),
        "records": [asdict(r) for r in records],
    }


def load_qa_set(path: str = config.QA_PATH,
                split: Optional[str] = None) -> List[Dict[str, Any]]:
    items = json.load(open(path, encoding="utf-8"))["items"]
    if split:
        items = [it for it in items if it.get("split") == split]
    return items
