# -*- coding: utf-8 -*-
"""经验边复用（VikingRAG-E，论文 Algorithm 2/3）与自适应升级（VikingRAG-E+）。

- 经验边：把一次成功的多轮检索轨迹（query -> 证据 URI 序列）物化存储；
  相似新查询直接沿边走，把"多轮探索"压缩成"一轮读证据"。
- 自适应升级：先做一轮"经验增强检索"，用结构化充分性审查验证证据；
  通过了直接作答（省掉多轮交互），不通过才升级完整的 agentic 多轮检索。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional

import config
from .llm import LLM, approx_tokens
from .prompts import SUFFICIENCY_SYSTEM
from .retriever import EvidenceGapRetriever, RunTrace
from .storage import Storage, tokenize
from .tools import ToolKit


@dataclass
class ExperienceEdge:
    query: str
    trace: List[str]      # 支撑答案的 URI 序列（检索轨迹的物化）
    answer: str


class ExperienceStore:
    """经验边表：SQLite 持久化 + BM25 相似查询匹配。"""

    def __init__(self, storage: Storage) -> None:
        self.conn = storage.conn

    def build(self, trace: RunTrace, answer: str) -> None:
        """检索成功后物化经验边：问题 -> Read 过的证据 URI 序列。"""
        uris = trace.read_uris()
        if not uris:
            return
        self.conn.execute(
            "INSERT INTO experience_edges(query, trace, answer) VALUES(?,?,?)",
            (trace.question, json.dumps(uris, ensure_ascii=False), answer))
        self.conn.commit()

    def match(self, question: str, top: int = 2,
              min_sim: float = None,
              min_overlap: int = None) -> List[ExperienceEdge]:
        """用 BM25 思想做查询-查询相似度：对历史 query 打分，超阈值即复用。"""
        min_sim = min_sim if min_sim is not None else config.DEFAULT_HARNESS["experience_min_sim"]
        min_overlap = (min_overlap if min_overlap is not None
                       else config.DEFAULT_HARNESS["experience_min_overlap"])
        rows = self.conn.execute(
            "SELECT query, trace, answer FROM experience_edges").fetchall()
        if not rows:
            return []
        # 只用长度>=2的词项（中文二元组/英文词）计算重叠，避免"是什么"这类噪声
        qterms = {t for t in tokenize(question) if len(t) >= 2}
        scored = []
        for r in rows:
            tterms = {t for t in tokenize(r["query"]) if len(t) >= 2}
            if not tterms:
                continue
            overlap = len(qterms & tterms)
            score = overlap / (len(qterms) ** 0.5 * len(tterms) ** 0.5 + 1e-9)
            if overlap >= min_overlap:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        edges = []
        for score, r in scored[:top]:
            if score * (len(qterms) ** 0.5) >= min_sim / 5:  # 归一化到可调阈值
                edges.append(ExperienceEdge(
                    query=r["query"], trace=json.loads(r["trace"]),
                    answer=r["answer"]))
        return edges


class VikingRAGEPlus:
    """VikingRAG-E+：一轮经验增强检索 + 证据充分性审查 + 自适应升级。"""

    def __init__(self, storage: Storage, params: Dict = None,
                 executor=None, audit=None) -> None:
        self.st = storage
        p = dict(config.DEFAULT_HARNESS)
        if params:
            p.update(params)
        self.p = p
        self.audit = audit
        self.retriever = EvidenceGapRetriever(
            ToolKit(storage, top_k=p["top_k"]),
            budget=p["round_budget"], history_compress=p["history_compress"],
            prompt_variant=p["prompt_variant"],
            executor=executor, audit=audit)
        self.edges = ExperienceStore(storage)

    # ---------------- 一轮低成本路径 ----------------

    def _one_round(self, question: str, edge: Optional[ExperienceEdge]) -> RunTrace:
        trace = RunTrace(question=question)
        seed = list(edge.trace) if edge else []
        for uri in seed[:self.p["seed_reads"]]:  # 沿经验边：直接读历史证据
            result = self.retriever.tk.read(uri)
            trace.steps.append({"tool": "Read", "args": {"uri": uri},
                                "result": result, "from_edge": True})
            trace.evidence[uri] = result
            trace.add_tokens(result)
        if not edge:  # 无经验：一轮 Search 粗定位 + Read 验证（摘要命中必须展开原文块）
            hits = self.st.bm25_search(question, top_k=self.p["top_k"])
            for obj, _ in hits[:self.p["one_round_reads"]]:
                result = self.retriever.tk.read(obj["uri"])
                trace.steps.append({"tool": "Read", "args": {"uri": obj["uri"]},
                                    "result": result})
                trace.evidence[obj["uri"]] = result
                trace.add_tokens(result)
                if obj["kind"] == "abstract":  # 导航摘要 -> 必须读对应原文块
                    blk_uri = obj["node_uri"] + "/blk0"
                    if self.st.get_object(blk_uri) is not None and blk_uri not in trace.evidence:
                        blk = self.retriever.tk.read(blk_uri)
                        trace.steps.append({"tool": "Read", "args": {"uri": blk_uri},
                                            "result": blk, "expanded": True})
                        trace.evidence[blk_uri] = blk
                        trace.add_tokens(blk)
        return trace

    def _sufficiency_check(self, question: str, trace: RunTrace) -> Dict:
        """结构化充分性审查（约束：引用必须支撑判断，缺则升级）。"""
        evidence_text = "\n\n".join(trace.evidence.values()) or "（无证据）"
        prompt = (f"【问题】{question}\n【证据】\n{evidence_text}\n\n"
                  f"请审查：证据是否足以回答上述问题？关键判断是否都有出处？")
        msgs = [{"role": "system", "content": SUFFICIENCY_SYSTEM},
                {"role": "user", "content": prompt}]
        trace.add_tokens(msgs[0]["content"] + prompt)
        trace.llm_calls += 1
        verdict = LLM.chat_json(msgs, max_tokens=300)
        trace.add_tokens(str(verdict))
        # 离线兜底启发式（LM 在线时由模型审查）：
        # 证据条数达标 且 问题内容词在证据中有最低覆盖度，否则升级多轮检索
        if not verdict:
            stop = set("的了是在和与或有哪些什么多少为及请对应的之其该此患者"
                       "什么怎么如何是否")
            terms = {t for t in tokenize(question)
                     if len(t) > 1 or (t not in stop and t not in "。，？！、（）")}
            ev_text = "\n".join(trace.evidence.values())
            cover = (sum(1 for t in terms if t in ev_text) / len(terms)
                     if terms else 1.0)
            sufficient = (len(trace.evidence) >= self.p["evidence_min_reads"]
                          and cover >= 0.35)
            verdict = {"sufficient": sufficient, "missing": [],
                       "reason": f"离线启发式 cover={cover:.2f}"}
        return verdict

    # ---------------- 对外入口 ----------------

    def run(self, question: str, verbose: bool = False):
        """返回 (answer, trace, path, verdict)。"""
        matched = self.edges.match(
            question, min_sim=self.p["experience_min_sim"],
            min_overlap=self.p["experience_min_overlap"])
        if verbose and matched:
            print(f"  [经验边命中] {len(matched)} 条历史轨迹")

        # ---- E+ 低成本路径：一轮 + 审查 ----
        trace = self._one_round(question, matched[0] if matched else None)
        verdict = self._sufficiency_check(question, trace)
        if verdict.get("sufficient"):
            ans = self.retriever.answer(question, trace)
            # 一轮成功轨迹同样物化为经验边（下次相似查询直接复用）
            self.edges.build(trace, ans)
            return ans, trace, "E+ 一轮路径（经验增强检索 + 审查通过）", verdict

        # ---- 升级：完整 agentic 多轮检索 ----
        if verbose:
            print(f"  [自适应升级] 证据不足（{verdict.get('reason', '')}），"
                  f"启动多轮 agentic 检索…")
        seed = list(trace.read_uris())
        if matched:
            for e in matched:
                seed += e.trace
        trace = self.retriever.run(question, seed_uris=seed[:8], verbose=verbose)
        ans = self.retriever.answer(question, trace)
        # 成功后把新轨迹也物化为经验边
        self.edges.build(trace, ans)
        return ans, trace, "E+ 升级路径（多轮证据缺口驱动检索）", verdict
