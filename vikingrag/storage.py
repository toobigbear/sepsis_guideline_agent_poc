# -*- coding: utf-8 -*-
"""层次化语义存储（对应论文 3.1 节）。

设计要点：
- 每份 Markdown 文档解析为目录树：文档 -> 章(#) -> 节(##) -> 块(段落)
- 每个结构节点与向量索引对象（块 / 摘要）都分配 URI，结构信息不进 prompt，
  由 List/Search/Grep/Read 按需访问 —— "结构 token 跟访问片段走"
- 摘要自底向上生成（块 -> 节摘要 -> 章摘要），可用 LLM，也可用抽取式兜底
- 检索用 BM25 倒排索引替代向量索引（零额外依赖，中文按 unigram+bigram 分词）
"""
from __future__ import annotations

import json
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .llm import LLM, approx_tokens

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\.]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> List[str]:
    """中英文混合分词：英文按词、中文按单字；另补中文二元组提升短语匹配。"""
    base = _TOKEN_RE.findall(text.lower())
    cjk = [t for t in base if len(t) == 1 and '\u4e00' <= t <= '\u9fff']
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return base + bigrams


# ---------------------------------------------------------------- 目录树

@dataclass
class Block:
    text: str
    uri: str = ""


@dataclass
class Section:
    title: str
    level: int          # 1 = 章(#), 2 = 节(##)
    uri: str = ""
    blocks: List[Block] = field(default_factory=list)
    children: List["Section"] = field(default_factory=list)
    abstract: str = ""


@dataclass
class DocTree:
    name: str           # 文档名（URI 命名空间）
    title: str
    sections: List[Section] = field(default_factory=list)   # 章
    orphan_blocks: List[Block] = field(default_factory=list)  # 章前散块


HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def parse_markdown(path: str) -> DocTree:
    """把结构化的 Markdown 指南解析成目录树（章=一级标题，节=二级标题）。"""
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    tree = DocTree(name=name, title=name)
    cur_ch: Optional[Section] = None
    cur_sec: Optional[Section] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf
        text = "\n".join(buf).strip()
        buf = []
        if not text:
            return
        blk = Block(text=text)
        if cur_sec is not None:
            cur_sec.blocks.append(blk)
        elif cur_ch is not None:
            cur_ch.blocks.append(blk)
        else:
            tree.orphan_blocks.append(blk)

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            if level <= 1:
                cur_ch = Section(title=title, level=1)
                tree.sections.append(cur_ch)
                cur_sec = None
            elif level <= 2:
                cur_sec = Section(title=title, level=2)
                (cur_ch.children if cur_ch else tree.sections).append(cur_sec)
            else:  # 更深标题并入当前节的标题文本
                if cur_sec is not None:
                    cur_sec.title += " · " + title
        else:
            buf.append(line)
    flush()

    # 分配 URI：viking://<doc>/<ch_idx>  或  viking://<doc>/<ch_idx>/<sec_idx>
    for ci, ch in enumerate(tree.sections):
        ch.uri = f"viking://{name}/ch{ci}"
        for si, sec in enumerate(ch.children):
            sec.uri = f"{ch.uri}/sec{si}"
        for bi, blk in enumerate(ch.blocks):
            blk.uri = f"{ch.uri}/blk{bi}"
        for si, sec in enumerate(ch.children):
            for bi, blk in enumerate(sec.blocks):
                blk.uri = f"{sec.uri}/blk{bi}"
    for bi, blk in enumerate(tree.orphan_blocks):
        blk.uri = f"viking://{name}/blk{bi}"
    return tree


# ---------------------------------------------------------------- 存储引擎

class Storage:
    """SQLite 持久化的层次化语义存储引擎。"""

    def __init__(self, db_path: str) -> None:
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        c = self.conn
        c.executescript("""
        CREATE TABLE IF NOT EXISTS nodes(
            uri TEXT PRIMARY KEY, kind TEXT, title TEXT,
            parent_uri TEXT, doc TEXT, level INT);
        CREATE TABLE IF NOT EXISTS objects(
            uri TEXT PRIMARY KEY, kind TEXT, node_uri TEXT, text TEXT, tokens INT);
        CREATE TABLE IF NOT EXISTS inverted(
            term TEXT, uri TEXT, tf REAL, PRIMARY KEY(term, uri));
        CREATE TABLE IF NOT EXISTS stats(key TEXT PRIMARY KEY, val REAL);
        CREATE TABLE IF NOT EXISTS experience_edges(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT, trace TEXT, answer TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP);
        """)
        c.commit()

    # ---------------- 写入 ----------------

    def _make_abstract(self, title: str, parts: List[str], is_leaf: bool) -> str:
        """自底向上生成摘要。有 LLM 用 LLM（质量高），离线用抽取式兜底。"""
        content = "\n".join(parts)
        if LLM.backend == "lmstudio":
            msgs = [{"role": "system", "content":
                     "你是医学文档摘要器。只输出一句话摘要，不超过60字，不要任何解释。"},
                    {"role": "user", "content":
                     f"请为以下章节生成一句话摘要，保留关键数值与标准：\n标题：{title}\n{content}"}]
            summary = LLM.chat(msgs, max_tokens=120).strip()
            if summary:
                return summary
        # 抽取式兜底：取含数字/标准/评分的关键句
        sents = re.split(r"[。；\n]", content)
        keys = [s.strip() for s in sents
                if re.search(r"[0-9≥≤]|SOFA|mmol|mmHg|评分|标准|小时内", s)
                and 6 < len(s.strip()) < 90]
        head = (keys[:2] if keys else [s.strip() for s in sents if s.strip()][:1])
        abs_text = "；".join(head)
        return f"{title}：{abs_text}"[:160]

    def add_document(self, tree: DocTree) -> Dict[str, int]:
        """整棵目录树落库：节点、块、摘要（自底向上）、倒排索引。"""
        n_nodes = n_objs = 0
        c = self.conn
        root_uri = f"viking://{tree.name}"
        c.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?,?)",
                  (root_uri, "doc", tree.title, None, tree.name, 0))
        n_nodes += 1

        def index_text(uri: str, text: str) -> None:
            toks = tokenize(text)
            if not toks:
                return
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            for term, f in tf.items():
                c.execute("INSERT OR REPLACE INTO inverted VALUES(?,?,?)",
                          (term, uri, f))

        def put_object(uri: str, kind: str, node_uri: str, text: str) -> None:
            nonlocal n_objs
            c.execute("INSERT OR REPLACE INTO objects VALUES(?,?,?,?,?)",
                      (uri, kind, node_uri, text, approx_tokens(text)))
            index_text(uri, text)
            n_objs += 1

        for blk in tree.orphan_blocks:
            put_object(blk.uri, "chunk", root_uri, blk.text)

        for ch in tree.sections:
            self._put_node(ch, root_uri, tree.name)
            n_nodes += 1
            # 先处理节（叶子），再自底向上汇总章摘要
            for sec in ch.children:
                self._put_node(sec, ch.uri, tree.name)
                n_nodes += 1
                for blk in sec.blocks:
                    put_object(blk.uri, "chunk", sec.uri, blk.text)
                if sec.blocks:
                    sec.abstract = self._make_abstract(
                        sec.title, [b.text for b in sec.blocks], is_leaf=True)
                    put_object(sec.uri + "#abs", "abstract", sec.uri, sec.abstract)
            for blk in ch.blocks:
                put_object(blk.uri, "chunk", ch.uri, blk.text)
            ch.abstract = self._make_abstract(
                ch.title, ([s.abstract for s in ch.children if s.abstract]
                           or [b.text for b in ch.blocks]), is_leaf=False)
            put_object(ch.uri + "#abs", "abstract", ch.uri, ch.abstract)
        self.conn.commit()
        return {"nodes": n_nodes, "objects": n_objs}

    def _put_node(self, sec: Section, parent: str, doc: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,?,?,?)",
                          (sec.uri, "dir", sec.title, parent, doc, sec.level))

    # ---------------- 读取 ----------------

    def get_object(self, uri: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM objects WHERE uri=?", (uri,)).fetchone()

    def list_children(self, uri: str) -> List[sqlite3.Row]:
        """List(u)：列出目录节点的直接子节点（局部结构，不加载全树）。"""
        rows = self.conn.execute(
            "SELECT uri, kind, title FROM nodes WHERE parent_uri=? ORDER BY uri",
            (uri,)).fetchall()
        if not rows:  # 兼容裸目录 URI（无子节点时看对象）
            rows = self.conn.execute(
                "SELECT uri, kind, node_uri AS title FROM objects WHERE node_uri=?",
                (uri,)).fetchall()
        return rows

    def _df(self, term: str) -> int:
        r = self.conn.execute(
            "SELECT COUNT(*) n FROM inverted WHERE term=?", (term,)).fetchone()
        return int(r["n"])

    def bm25_search(self, query: str, scope: Optional[str] = None,
                    top_k: int = 5) -> List[Tuple[sqlite3.Row, float]]:
        """Search(q,u,K)：BM25 检索块与摘要，可按 URI 前缀限定目录范围。"""
        terms = tokenize(query)
        if not terms:
            return []
        n_docs = max(1, self.conn.execute(
            "SELECT COUNT(*) n FROM objects").fetchone()["n"])
        scores: Dict[str, float] = {}
        for term in set(terms):
            rows = self.conn.execute(
                "SELECT uri, tf FROM inverted WHERE term=?", (term,)).fetchall()
            df = self._df(term)
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for r in rows:
                if scope and not r["uri"].startswith(scope):
                    continue
                scores[r["uri"]] = scores.get(r["uri"], 0.0) + idf * (r["tf"] * 1.2) / (
                    r["tf"] + 1.2 * (1 - 0.75 + 0.75 * r["tf"] / 8.0))
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        out = []
        for uri, score in ranked:
            obj = self.get_object(uri)
            if obj is not None:
                out.append((obj, score))
        return out

    def grep(self, pattern: str, scope: Optional[str] = None,
             max_hits: int = 10) -> List[Dict[str, str]]:
        """Grep(p,u)：在目录范围内的对象文本上做词法检索，返回命中行。"""
        rx = re.compile(re.escape(pattern), re.IGNORECASE)
        sql = "SELECT uri, text FROM objects"
        rows = self.conn.execute(sql).fetchall()
        hits = []
        for r in rows:
            if scope and not r["uri"].startswith(scope):
                continue
            for line in r["text"].splitlines():
                if rx.search(line):
                    hits.append({"uri": r["uri"], "line": line.strip()[:160]})
                    break
            if len(hits) >= max_hits:
                break
        return hits

    def full_catalog_tokens(self) -> int:
        """DeepRead 类基线：把完整目录树全部塞进 prompt 的 token 量。"""
        r = self.conn.execute(
            "SELECT COALESCE(SUM(tokens),0) t FROM objects").fetchone()
        node_titles = self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(title)),0) t FROM nodes").fetchone()
        return int(r["t"] + node_titles["t"] * 1.2)

    def close(self) -> None:
        self.conn.close()
