# -*- coding: utf-8 -*-
"""LLM 接入层：三级降级链 + 离线规则兜底。

  LM Studio（主，Vulkan/GPU） -> Ollama（兜底，CPU） -> OfflineBrain（规则引擎）
任一通道不可达即自动降级；恢复后自动回升。所有 LLM 调用统一走这里，
Harness 的审计/校验层在其外侧。
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Any

import config


def approx_tokens(text: str) -> int:
    """粗略估算 token：CJK/全角约 1 token，英文词约 1.3，用于 token 统计。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if ord(ch) > 0x2E80)
    latin = len(re.findall(r"[A-Za-z0-9_\.]+", text))
    other = max(0, len(re.findall(r"[\x20-\x7E]", text)) - latin * 3)
    return cjk + int(latin * 1.3) + other // 4


class ChatLLM:
    """OpenAI 兼容客户端，带多通道降级。"""

    def __init__(self, slot: str = config.ACTIVE_MODEL) -> None:
        self.slot = slot
        info = config.MODEL_SLOTS.get(slot)
        if info is None:  # 槽位名不存在时回退到第一个启用的槽位（不绑定具体名字）
            info = next(v for v in config.MODEL_SLOTS.values()
                        if not v.get("disabled"))
        self.model = info["model"]
        self._backend = "offline"
        self._clients: Dict[str, Any] = {}
        if info.get("disabled"):
            return
        self._init_clients()
        self._probe()

    # ---------------- 通道管理 ----------------

    def _init_clients(self) -> None:
        try:
            from openai import OpenAI
            self._clients["lmstudio"] = OpenAI(
                base_url=config.LMSTUDIO_BASE_URL, api_key=config.LMSTUDIO_API_KEY,
                timeout=20, max_retries=0)
            self._clients["ollama"] = OpenAI(
                base_url=config.OLLAMA_BASE_URL, api_key="ollama",
                timeout=20, max_retries=0)
        except Exception:
            self._clients = {}

    def _try_chat(self, backend: str, messages, max_tokens, temperature,
                  timeout: int | None = None) -> str:
        client = self._clients[backend]
        model = self.model
        if timeout is None:
            # E+ 升级后最终合成的 prompt 可达 2~3 万 token，
            # 8B/Vulkan 预填充需 1~3 分钟，20s 默认超时会把它掐死 → 误降级。
            timeout = 600 if backend == "lmstudio" else 300
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
            timeout=timeout)
        return resp.choices[0].message.content or ""

    def _probe(self) -> None:
        for backend in ("lmstudio", "ollama"):
            if backend not in self._clients:
                continue
            try:
                self._try_chat(backend, [{"role": "user", "content": "ping"}], 1, 0.0,timeout=15)
                self._backend = backend
                return
            except Exception:
                continue

    @property
    def backend(self) -> str:
        return self._backend

    # ---------------- 对话 ----------------

    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 1024,
             temperature: float = 0.0) -> str:
        order = [self._backend] + [b for b in ("lmstudio", "ollama")
                                   if b != self._backend and b in self._clients]
        for backend in order:
            try:
                text = self._try_chat(backend, messages, max_tokens, temperature)
                self._backend = backend
                return text
            except Exception:
                continue
        return OfflineBrain.heuristic_reply(messages)

    def chat_json(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> Dict[str, Any]:
        raw = self.chat(messages, max_tokens=max_tokens).strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    pass
        return {}


class OfflineBrain:
    """离线规则引擎：LM Studio / Ollama 都不可用时兜底。
    保证检索轨迹、经验边、Harness 校验、token 统计全链路可演示。"""

    @staticmethod
    def _last_user(messages: List[Dict[str, str]]) -> str:
        for m in reversed(messages):
            if m["role"] == "user":
                return m["content"]
        return ""

    @classmethod
    def heuristic_reply(cls, messages: List[Dict[str, str]]) -> str:
        text = cls._last_user(messages)
        if "下一步动作" in text or "工具调用" in text:
            return cls._tool_decision(text)
        if "证据是否充分" in text:
            return json.dumps({"sufficient": True, "missing": [],
                               "reason": "离线模式：默认判为充分"}, ensure_ascii=False)
        if "摘要" in text and "生成" in text:
            return cls._summarize(text)
        return cls._final_answer(text)

    @staticmethod
    def _tool_decision(text: str) -> str:
        """模拟证据缺口驱动轨迹：Search 锚点 -> Read -> Read -> answer。
        Search 查询从问题中抽取（而非固定词），行为更贴近真模型。"""
        n_reads = text.count("【工具结果-Read】")
        n_search = text.count("【工具结果-Search】")
        if n_search == 0:
            qm = re.search(r"【问题】(.+?)\n\n【历史工具调用", text, re.DOTALL)
            query = (qm.group(1)[:80] if qm else "脓毒症 SOFA 评分 乳酸 定义 标准")
            return json.dumps({"action": "tool_call", "tool": "Search",
                               "args": {"query": query},
                               "think": "首轮：无证据，先语义粗定位"}, ensure_ascii=False)
        if n_reads < 2:
            return json.dumps({"action": "tool_call", "tool": "Read",
                               "args": {"uri": "auto-pick"},
                               "think": "已定位候选，Read 验证缺口"}, ensure_ascii=False)
        return json.dumps({"action": "answer", "think": "证据已充分"}, ensure_ascii=False)

    @staticmethod
    def _summarize(text: str) -> str:
        body = text.split("生成一句话摘要", 1)[-1]
        sents = re.split(r"[。；\n]", body)
        key = [s.strip() for s in sents
               if re.search(r"[0-9≥≤<>]|SOFA|脓毒|mmHg|mmol|评分", s)
               and 6 < len(s.strip()) < 80]
        return "。".join(key[:2]) or (sents[0].strip()[:60] if sents else "")

    @staticmethod
    def _final_answer(text: str) -> str:
        """离线最终答复：返回证据摘录（让评测的关键词命中取决于检索质量），
        附一句规则判断。LM Studio 在线时由真模型生成。"""
        ev_m = re.search(r"【证据】\n(.*?)\n\n请作答", text, re.DOTALL)
        evidence = ev_m.group(1) if ev_m else ""
        lines, seen = [], set()
        for ln in evidence.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("viking://") and ln not in seen:
                seen.add(ln)
                lines.append(ln)
        excerpt = "；".join(lines)[:700]
        low = text
        if re.search(r"去甲肾上腺素|乳酸[^\n]{0,20}[3-9]\.\d|≥ ?4", low):
            judge = "综合现有指标，存在脓毒症高风险征象（乳酸升高/循环支持加码），" \
                    "建议按指南时限尽快完成评估与处置。"
        else:
            judge = "依据现有指标，暂未满足脓毒症/脓毒性休克的 Sepsis-3 标准，" \
                    "建议继续监测乳酸与器官功能。"
        return f"【离线降级：证据摘录作答】\n{excerpt}\n{judge}"


LLM = ChatLLM()
