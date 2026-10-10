# -*- coding: utf-8 -*-
"""文档访问工具集 T = {Search, List, Grep, Read}（对应论文 3.1 节）。

四个工具共享同一 URI 空间，Agent 通过它们按需访问目录片段，
而不是把完整目录树序列化进 prompt —— 这是 VikingRAG 省 token 的关键。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from .storage import Storage


class ToolKit:
    def __init__(self, storage: Storage, top_k: int = 5) -> None:
        self.st = storage
        self.top_k = top_k

    def search(self, query: str, scope: Optional[str] = None, k: int = 0) -> str:
        """Search(q,u,K)：语义检索（BM25 实现），返回相关块/摘要 + URI + 预览。"""
        hits = self.st.bm25_search(query, scope=scope, top_k=k or self.top_k)
        if not hits:
            return "（无命中）"
        lines = []
        for obj, score in hits:
            preview = obj["text"][:110].replace("\n", " ")
            lines.append(f"[{score:.2f}] {obj['uri']} <{obj['kind']}>\n    {preview}…")
        return "\n".join(lines)

    def list_dir(self, uri: str) -> str:
        """List(u)：列出目录节点的直接子节点（章/节或块），暴露局部结构。"""
        rows = self.st.list_children(uri)
        if not rows:
            return f"（{uri} 下没有子节点，可尝试 Read）"
        return "\n".join(f"- {r['uri']}  {r['title']}" for r in rows)

    def grep(self, pattern: str, scope: Optional[str] = None) -> str:
        """Grep(p,u)：目录范围内的词法检索，适合精确数值/药名/标准命中。"""
        hits = self.st.grep(pattern, scope=scope)
        if not hits:
            return "（无命中）"
        return "\n".join(f"- {h['uri']}\n    {h['line']}" for h in hits)

    def read(self, uri: str) -> str:
        """Read(u)：读取 URI 对应对象（块/摘要）的完整内容——验证缺口的最后一步。"""
        obj = self.st.get_object(uri)
        if obj is not None:
            return f"{obj['uri']} <{obj['kind']}>\n{obj['text']}"
        # 目录 URI：退化为 List
        children = self.st.list_children(uri)
        if children:
            return self.list_dir(uri)
        return f"（找不到 {uri}：请先用 Search/List 发现有效 URI）"

    def dispatch(self, tool: str, args: Dict) -> str:
        t = tool.strip().lower()
        if t == "search":
            return self.search(args.get("query", ""), args.get("scope"), int(args.get("k", 0) or 0))
        if t == "list":
            return self.list_dir(args.get("uri", ""))
        if t == "grep":
            return self.grep(args.get("pattern", ""), args.get("scope"))
        if t == "read":
            return self.read(args.get("uri", ""))
        return f"未知工具 {tool}，可用：Search / List / Grep / Read"

    def catalog(self) -> List[str]:
        return ["Search(query, scope?, k?)", "List(uri)",
                "Grep(pattern, scope?)", "Read(uri)"]
