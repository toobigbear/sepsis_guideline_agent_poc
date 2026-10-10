# -*- coding: utf-8 -*-
"""确定性红线护栏（安全铁律层）。

铁律（对 Meta-Harness 冻结，不可搜索/修改）：
1. 不得输出诊断结论、治疗医嘱、预后判断；
2. 处方/病假/医嘱类请求一律拒绝并引导医师决策；
3. 对外文本必须携带免责声明；
4. URI 命名空间隔离，证据引用必须来自本次实际读取集合。
本模块全部为确定性代码，与 LLM 完全无关。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

import config
from vikingrag.prompts import ADVISORY_REFUSAL

# 违禁词字典：诊断结论 / 治疗医嘱 / 预后判断
# 注意：只收"对患者下结论/下指令"的措辞，不收指南转述语
# （指南原文含"推荐使用/不推荐常规使用"等，避免误伤证据摘录）。
FORBIDDEN_PATTERNS = [
    "可诊断", "诊断为", "明确诊断", "符合脓毒症诊断", "确诊脓毒症",
    "处方", "开具", "医嘱", "建议给予", "应给予", "予以下",
    "静滴", "静脉注射",
    "预后良好", "预计死亡", "存活率", "必然恶化", "预后判断",
]

# 越界请求识别（生成前第一层防线）
BOUNDARY_PATTERNS = [
    "处方", "病假", "假条", "医嘱", "帮我开", "给我开", "开具",
    "确诊为", "判断预后", "预后如何", "还能活",
]

DISCLAIMER = config.DISCLAIMER

CITATION_RE = re.compile(r"viking://[^\s\]\)）>\]\"']+")


@dataclass
class GuardrailReport:
    ok: bool
    forbidden_hits: List[str] = field(default_factory=list)
    has_disclaimer: bool = False
    boundary: bool = False

    def reasons(self) -> str:
        rs = []
        if self.forbidden_hits:
            rs.append("违禁词: " + ",".join(sorted(set(self.forbidden_hits))))
        if not self.has_disclaimer:
            rs.append("缺少免责声明")
        return "；".join(rs) or "通过"


class Guardrail:
    """三层防线中的第三层（写卡前确定性终检），以及生成前边界识别。"""

    @staticmethod
    def is_boundary_request(question: str) -> bool:
        return any(p in question for p in BOUNDARY_PATTERNS)

    @staticmethod
    def scan(text: str) -> List[str]:
        return [p for p in FORBIDDEN_PATTERNS if p in text]

    @classmethod
    def final_check(cls, text: str) -> GuardrailReport:
        hits = cls.scan(text)
        return GuardrailReport(ok=(not hits) and (DISCLAIMER in text),
                               forbidden_hits=hits,
                               has_disclaimer=DISCLAIMER in text)

    @staticmethod
    def refuse(question: str) -> str:
        return f"{ADVISORY_REFUSAL}\n（本次请求已按红线规则拒绝，事件已留痕。）\n{DISCLAIMER}"
