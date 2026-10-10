# -*- coding: utf-8 -*-
"""VikingRAG 迷你实现：面向结构化长文档（章/节/段）的 token 高效 RAG。

对应论文 arXiv:2609.11390 的三层机制：
- storage:  层次化语义存储（URI 寻址 + 摘要 + BM25 索引，SQLite 持久化）
- retriever: 证据缺口驱动的多轮检索（Search / List / Grep / Read 四工具）
- experience: 经验边复用（VikingRAG-E）与自适应升级（VikingRAG-E+）
"""
