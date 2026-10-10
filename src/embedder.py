# -*- coding: utf-8 -*-
"""向量化基线的嵌入层：LM Studio（nomic-embed-text）+ 离线确定性代理向量。

- 在线：走 LM Studio 的 OpenAI 兼容 /v1/embeddings 接口，
  使用已下载的 nomic-embed-text-v1.5（nomic 为非对称检索模型，
  文档加 "search_document: " 前缀、查询加 "search_query: " 前缀，见官方建议）；
- 离线：哈希签名随机投影（token 级，768 维，确定性、可复现），
  保证消融实验在无 GPU 时也能跑通——离线数字是各通道的下界。

候选模型名解析顺序：config.EMBEDDING_MODEL -> 探测 /v1/models 中含 embed 的模型。
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import List, Optional

import config


def _l2_normalize(v: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _hash_sign(s: str) -> int:
    """稳定的伪随机符号/权重：SHA256 取前 4 字节。"""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:4], "little")


class HashedEmbedding:
    """离线代理向量：token -> 哈希桶签名投影，余弦相似度近似词汇重叠。"""

    DIM = 768

    def __init__(self) -> None:
        self.backend = "hashed-surrogate"

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        out = []
        for text in texts:
            vec = [0.0] * self.DIM
            for tok in _tokens(text):
                for part in (f"w:{tok}", f"b:{tok[:2]}"):
                    h = _hash_sign(part)
                    vec[h % self.DIM] += 1.0 if (h >> 31) == 0 else -1.0
            out.append(_l2_normalize(vec))
        return out


def _tokens(text: str) -> List[str]:
    import re
    base = re.findall(r"[A-Za-z0-9_\.]+|[\u4e00-\u9fff]", text.lower())
    cjk = [t for t in base if len(t) == 1 and "一" <= t <= "鿿"]
    bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
    return [t for t in base if len(t) > 1 or t not in "的了是在和与或哪什么么"] + bigrams


class LMStudioEmbedder:
    """通过 OpenAI 兼容接口调用 LM Studio 的 embedding 模型。"""

    def __init__(self, model: str = None) -> None:
        from openai import OpenAI
        self.client = OpenAI(base_url=config.LMSTUDIO_BASE_URL,
                             api_key=config.LMSTUDIO_API_KEY, timeout=30)
        self._model = model or config.EMBEDDING_MODEL
        self.backend = "lmstudio-embedding"

    def _resolve_model(self) -> str:
        """确认模型在服务端可用（支持 /v1/models 探测 embed 模型）。"""
        try:
            models = [m.id for m in self.client.models.list().data]
        except Exception:
            models = []
        if self._model in models:
            return self._model
        for m in models:
            if "embed" in m.lower():
                return m
        if not models:
            return self._model  # 让调用报错再走降级
        raise RuntimeError(f"服务端无 embedding 模型（候选: {models}）")

    def embed(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        model = self._resolve_model()
        prefix = "search_query: " if is_query else "search_document: "
        resp = self.client.embeddings.create(
            model=model, input=[prefix + t for t in texts])
        return [_l2_normalize(d.embedding) for d in resp.data]


def build_embedder() -> object:
    """优先 LM Studio embedding，失败降级哈希代理（可复现）。"""
    try:
        emb = LMStudioEmbedder()
        emb.embed(["连通性探测"], is_query=True)
        return emb
    except Exception:
        return HashedEmbedding()


def cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
