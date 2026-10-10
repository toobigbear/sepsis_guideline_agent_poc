# -*- coding: utf-8 -*-
"""消融实验基线通道：STAIR（ToC 寻址）与普通向量 RAG。

与 VikingRAG 通道对照的三个基线实现（同一评测集、同一评分函数、同一护栏）：

- PlainRAGChannel（普通向量 RAG）：块 -> embedding -> 余弦 top-k -> 作答。
  没有任何结构信息（模拟"长度分块丢结构"的主流做法）。
- STAIRChannel（STAIR 的零样本适配）：STAIR 原文是让微调 LLM 直接"生成
  叶章节标题"作为生成式检索（ToC 即寻址空间）。POC 版不做微调，改为：
  路由器（Qwen3-7B 读目录选叶节点，离线用标题 BM25 兜底）-> Read 叶节点
  全部内容 -> 作答。忠实保留其核心思想：目录结构作为寻址与检索的第一公民。
  差异已在 POC 报告中声明（未微调、未用 DSI 式生成式打分）。
- DeepRead 基线仅作 token 参照（demo4），不在此作答。

幻觉率判定（三通道同一标准，确定性代码执行）：
答案中每条 [viking://...] 引用必须来自本通道实际取回的 URI 集合；
引用编造或护栏违禁词命中即计一次幻觉/越界。
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from vikingrag.llm import LLM, approx_tokens
from vikingrag.prompts import ANSWER_SYSTEM
from vikingrag.storage import Storage

from .embedder import build_embedder, cosine

RESULT_FIELDS = ("answer", "evidence_uris", "tokens", "llm_calls", "path",
                 "read_log")


@dataclass
class ChannelResult:
    answer: str = ""
    evidence_uris: List[str] = field(default_factory=list)
    tokens: int = 0
    llm_calls: int = 0
    path: str = ""
    read_log: List[Dict] = field(default_factory=list)

    def as_tuple(self):
        return (self.answer, self.evidence_uris, self.tokens,
                self.llm_calls, self.path, self.read_log)


# ---------------------------------------------------------------- 普通向量 RAG

class VectorIndex:
    """对存储层 chunk 对象建向量索引（带磁盘缓存）。"""

    def __init__(self, storage: Storage, embedder=None) -> None:
        self.st = storage
        self.emb = embedder or build_embedder()
        self.cache_path = os.path.join(config.OUTPUT_DIR, "vector_index.json")
        self.rows: List[Dict] = []          # [{uri, text, vec}]
        self.backend = self.emb.backend

    def build(self) -> None:
        if self._load_cache():
            return
        rows = self.st.conn.execute(
            "SELECT uri, text FROM objects WHERE kind='chunk'").fetchall()
        vecs = self.emb.embed([r["text"] for r in rows]) if rows else []
        self.rows = [{"uri": r["uri"], "text": r["text"], "vec": v}
                     for r, v in zip(rows, vecs)]
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump({"backend": self.backend,
                       "rows": [{"uri": r["uri"], "text": r["text"], "vec": r["vec"]}
                                for r in self.rows]}, f)

    def _load_cache(self) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        try:
            data = json.load(open(self.cache_path, encoding="utf-8"))
            if data.get("backend") != self.emb.backend:
                return False
            self.rows = data["rows"]
            return bool(self.rows)
        except Exception:
            return False

    def search(self, question: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        if not self.rows:
            return []
        qv = self.emb.embed([question], is_query=True)[0]
        scored = [(r["uri"], r["text"], cosine(qv, r["vec"])) for r in self.rows]
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:top_k]


class PlainRAGChannel:
    """普通向量 RAG：无结构、无工具、无多轮。token = 检索结果 + 问答 prompt。"""

    def __init__(self, storage: Storage, top_k: int = 5,
                 index: Optional[VectorIndex] = None) -> None:
        self.st = storage
        self.top_k = top_k
        self.index = index or VectorIndex(storage)
        self.index.build()

    def run(self, question: str) -> ChannelResult:
        res = ChannelResult(path=f"plain_rag(top_k={self.top_k})")
        hits = self.index.search(question, top_k=self.top_k)
        res.tokens += approx_tokens(question)
        evidence_lines = []
        for uri, text, _score in hits:
            res.evidence_uris.append(uri)
            evidence_lines.append(f"[{uri}]\n{text}")
            res.read_log.append({"tool": "vector_search", "uri": uri})
            res.tokens += approx_tokens(text)
        res.answer = self._answer(question, evidence_lines, res)
        return res

    def _answer(self, question, evidence_lines, res: ChannelResult) -> str:
        evidence = "\n\n".join(evidence_lines) or "（无检索结果）"
        prompt = f"【问题】{question}\n\n【证据】\n{evidence}\n\n请作答。"
        res.tokens += approx_tokens(ANSWER_SYSTEM + prompt)
        res.llm_calls += 1
        if LLM.backend != "offline":
            return LLM.chat([{"role": "system", "content": ANSWER_SYSTEM},
                             {"role": "user", "content": prompt}], max_tokens=900)
        return _offline_excerpt(evidence_lines, question)


# ---------------------------------------------------------------- STAIR 通道

STAIR_ROUTER_SYSTEM = """你是目录导航员。给定一份指南的目录（文档 > 章 > 节，每行末尾是 URI）
和一个问题，输出最可能包含答案的"节"的 URI。只输出一行 URI，不要其他文字。"""


class STAIRChannel:
    """STAIR 零样本适配：ToC 寻址 —— 路由到叶节 -> 读节内全部内容。"""

    def __init__(self, storage: Storage, max_sections: int = 6,
                 index: Optional[VectorIndex] = None) -> None:
        self.st = storage
        self.max_sections = max_sections
        self.index = index          # 复用向量索引作为离线路由器（标题向量）
        self.toc = self._build_toc()

    def _build_toc(self) -> List[Dict]:
        """目录 = 全部节节点（含文档/章标题作为结构前缀）。"""
        rows = self.st.conn.execute(
            "SELECT n.uri, n.title, n.parent_uri, p.title AS ptitle, n.doc "
            "FROM nodes n LEFT JOIN nodes p ON n.parent_uri = p.uri "
            "WHERE n.kind='dir' AND n.level=2 ORDER BY n.uri").fetchall()
        return [{"uri": r["uri"], "title": r["title"],
                 "chapter": r["ptitle"] or "", "doc": r["doc"]} for r in rows]

    def _route_llm(self, question: str) -> List[str]:
        lines = [f"- {t['doc']} > {t['chapter']} > {t['title']}  {t['uri']}"
                 for t in self.toc]
        prompt = (f"【目录】\n" + "\n".join(lines) +
                  f"\n\n【问题】{question}\n\n请输出最相关的节 URI（一行）：")
        raw = LLM.chat([{"role": "system", "content": STAIR_ROUTER_SYSTEM},
                        {"role": "user", "content": prompt}], max_tokens=60)
        uri = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        return [uri] if uri.startswith("viking://") else []

    def _route_offline(self, question: str) -> List[str]:
        """离线路由器：节标题 + 章标题 的向量相似度（复用 nomic 索引向量）。"""
        if not self.index.rows:
            return []
        qv = self.index.emb.embed([question], is_query=True)[0]
        title_vec = {}
        for t in self.toc:
            title_text = f"{t['doc']} {t['chapter']} {t['title']}"
            if title_text not in title_vec:
                title_vec[title_text] = self.index.emb.embed([title_text])[0]
        scored = sorted(
            ((cosine(qv, title_vec[f"{t['doc']} {t['chapter']} {t['title']}"]), t["uri"])
             for t in self.toc), reverse=True)
        return [uri for _s, uri in scored[:self.max_sections]]

    def _section_objects(self, sec_uri: str) -> List[Tuple[str, str]]:
        rows = self.st.conn.execute(
            "SELECT uri, text FROM objects WHERE node_uri=? ORDER BY uri",
            (sec_uri,)).fetchall()
        out = [(r["uri"], r["text"]) for r in rows]
        if not out:  # 目录 URI 本身没有对象时，取该节名下全部块
            rows = self.st.conn.execute(
                "SELECT uri, text FROM objects WHERE uri LIKE ? ORDER BY uri",
                (sec_uri + "/%",)).fetchall()
            out = [(r["uri"], r["text"]) for r in rows]
        return out

    def run(self, question: str) -> ChannelResult:
        res = ChannelResult(path="stair(toc-routing)")
        res.tokens += approx_tokens(question) + sum(
            approx_tokens(t["title"]) for t in self.toc)  # ToC 进路由 prompt
        res.llm_calls += 1
        if LLM.backend != "offline":
            uris = self._route_llm(question)
            if not uris:
                uris = self._route_offline(question)
        else:
            uris = self._route_offline(question)
        evidence_lines: List[str] = []
        for uri in uris[: self.max_sections]:
            for obj_uri, text in self._section_objects(uri):
                res.evidence_uris.append(obj_uri)
                evidence_lines.append(f"[{obj_uri}]\n{text}")
                res.tokens += approx_tokens(text)
                res.read_log.append({"tool": "stair_read_section", "uri": obj_uri})
        if not evidence_lines:
            res.answer = "（STAIR 路由未命中任何节）"
            return res
        res.tokens += approx_tokens(ANSWER_SYSTEM + question)
        res.llm_calls += 1
        if LLM.backend != "offline":
            res.answer = LLM.chat(
                [{"role": "system", "content": ANSWER_SYSTEM},
                 {"role": "user", "content":
                  f"【问题】{question}\n\n【证据】\n" + "\n\n".join(evidence_lines)}],
                max_tokens=900)
        else:
            res.answer = _offline_excerpt(evidence_lines, question)
        return res


def _offline_excerpt(evidence_lines: List[str], question: str) -> str:
    """离线作答：证据摘录（每段带 [uri] 引用，供引用校验），附规则结论。"""
    excerpt = "\n".join(ln for ln in evidence_lines)[:900]
    if any(k in excerpt for k in ("去甲肾上腺素", "血管活性", "4 分", "3 分")):
        judge = "综合上述指南依据，问题所指条款已在引用章节中列明。"
    else:
        judge = "依据所取回的指南内容作答如上。"
    return f"【离线降级：证据摘录作答】\n{excerpt}\n{judge}"
