# -*- coding: utf-8 -*-
"""Meta-Harness：在医疗评测集上自动搜索/迭代 Harness 调度与校验参数。

思想来自 Meta-Harness（Stanford IRIS + MIT, arXiv:2603.28052）：
外层循环让 proposer 读取全部历史候选的 配置 + 分数 + 失败轨迹，
提出新候选，在搜索集上评估，维护 Pareto 前沿，最后才在测试集上验证。

本实现的务实裁剪（医院 POC 可落地版）：
1. 搜索对象 = HarnessConfig 的参数组合（不是自由代码）——安全红线天然隔离；
2. 安全不变量全程冻结：guardrails 违禁词、免责声明、引用校验、轮数硬上限
   不在 SEARCH_SPACE 中，Meta 层连读都不能改（治理可审计）；
3. proposer 两种：
   - systematic（默认）：对当前最优配置做坐标下降 + 全量历史（轨迹文件）反馈，
     离线可跑，结果确定可复现；
   - llm：用本地 Qwen3-7B 读历史提出下一个候选（解析失败自动回退 systematic）；
4. 搜索集/测试集严格隔离：调参只看 search split，最终报告 test split。
"""
from __future__ import annotations

import itertools
import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import config
from vikingrag.llm import LLM
from vikingrag.storage import Storage

from .audit import AuditLogger
from .eval_runner import EvalRunner, ItemRecord, load_qa_set, summarize
from .harness import HarnessConfig, ToolGateway


class MetaHarness:
    def __init__(self, storage: Storage, proposer: str = "systematic",
                 audit: Optional[AuditLogger] = None) -> None:
        self.st = storage
        self.proposer_name = proposer
        self.audit = audit or AuditLogger()
        items = load_qa_set()
        self.search_items = [it for it in items if it.get("split") == "search"]
        self.test_items = [it for it in items if it.get("split") == "test"]
        self.history: List[Dict[str, Any]] = []  # 全量历史：配置+分数+失败摘要
        self.trace_dir = os.path.join(config.META_DIR, "traces")
        os.makedirs(self.trace_dir, exist_ok=True)

    # ---------------- 评估 ----------------

    def evaluate(self, cfg: HarnessConfig,
                 items: List[Dict[str, Any]],
                 tag: str) -> Dict[str, Any]:
        runner = EvalRunner(self.st, cfg, audit=self.audit)
        result = runner.run(items)
        failures = [f"{r['item_id']}: recall={r['evidence_recall']}, "
                    f"kw={r['keyword_hit']}, blocked={r['blocked'] or '-'}"
                    for r in result["records"] if r["score"] < 0.7]
        entry = {"tag": tag, "cfg": cfg.as_dict(), "score": result["score"],
                 "tokens_avg": result["tokens_avg"],
                 "evidence_recall": result["evidence_recall"],
                 "keyword_hit": result["keyword_hit"],
                 "failures": failures}
        self.history.append(entry)
        with open(os.path.join(config.META_DIR, "iterations.jsonl"),
                  "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(os.path.join(self.trace_dir, f"{tag}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(result["records"], f, ensure_ascii=False, indent=1)
        return result

    # ---------------- proposer ----------------

    def propose(self, best_cfg: HarnessConfig,
                round_idx: int) -> List[HarnessConfig]:
        if self.proposer_name == "llm":
            cand = self._propose_by_llm(best_cfg, round_idx)
            if cand:
                return cand
        return self._propose_by_systematic(best_cfg, round_idx)

    def _propose_by_systematic(self, best_cfg: HarnessConfig,
                               round_idx: int) -> List[HarnessConfig]:
        """坐标下降：每次只动一个参数，取搜索空间的下一个候选值。"""
        cands = []
        base = best_cfg.as_dict()
        for param, values in config.SEARCH_SPACE.items():
            cur = base.get(param)
            idx = values.index(cur) if cur in values else -1
            nxt = values[(idx + 1) % len(values)]
            if nxt == cur:
                continue
            d = dict(base)
            d[param] = nxt
            cands.append(HarnessConfig.from_dict(d))
        return cands

    def _propose_by_llm(self, best_cfg: HarnessConfig,
                        round_idx: int) -> Optional[List[HarnessConfig]]:
        """LLM proposer：读取全量历史（分数+失败轨迹），提出下一个候选配置。"""
        hist = json.dumps(self.history[-8:], ensure_ascii=False, indent=1)[:6000]
        space = json.dumps(config.SEARCH_SPACE, ensure_ascii=False)
        prompt = (
            "你在优化一个医疗问答 Agent 的 Harness 参数。以下是历史候选的"
            "配置、得分与失败题目（只可调调度/检索/校验阈值参数，安全规则不可动）。\n"
            f"搜索空间：{space}\n\n历史：\n{hist}\n\n"
            "请提出 3 个不同的新候选配置（JSON 数组，每个元素是完整的参数对象），"
            "优先针对失败题目做有依据的修改，不要重复已有候选。")
        msgs = [{"role": "system",
                 "content": "你是 Harness 优化器，只输出 JSON 数组，不要其他文字。"},
                {"role": "user", "content": prompt}]
        resp = LLM.chat_json(msgs, max_tokens=1200)
        cands = []
        for d in (resp if isinstance(resp, list) else []):
            try:
                cfg = HarnessConfig.from_dict(d)
                if cfg.round_budget > ToolGateway.HARD_ROUND_CAP:
                    continue  # 冻结上限，Meta 无权突破
                cands.append(cfg)
            except Exception:
                continue
        return cands[:3] or None

    # ---------------- 主循环 ----------------

    @staticmethod
    def pareto_front(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """二维 Pareto：score 越大越好，tokens_avg 越小越好。"""
        front = []
        for h in history:
            dominated = any(
                (o["score"] >= h["score"] and o["tokens_avg"] <= h["tokens_avg"])
                and (o["score"] > h["score"] or o["tokens_avg"] < h["tokens_avg"])
                for o in history if o is not h)
            if not dominated:
                front.append(h)
        return sorted(front, key=lambda h: h["score"], reverse=True)

    def optimize(self, rounds: int = 2) -> Dict[str, Any]:
        open(os.path.join(config.META_DIR, "iterations.jsonl"), "w").close()
        print(f"[Meta-Harness] proposer={self.proposer_name}  "
              f"搜索集 {len(self.search_items)} 题 / 测试集 {len(self.test_items)} 题"
              f"（测试集调参全程不可见）")

        # 基线：人工默认配置
        base = HarnessConfig.from_dict(config.DEFAULT_HARNESS)
        best_res = self.evaluate(base, self.search_items, tag="iter0_baseline")
        best_cfg, best_score = base, best_res["score"]
        print(f"[iter 0] 人工基线 score={best_score:.3f} "
              f"tokens_avg={best_res['tokens_avg']}")

        for r in range(1, rounds + 1):
            cands = self.propose(best_cfg, r)
            for i, cfg in enumerate(cands):
                res = self.evaluate(cfg, self.search_items,
                                    tag=f"iter{r}_cand{i}")
                marker = ""
                if res["score"] > best_score:
                    best_cfg, best_score = cfg, res["score"]
                    marker = "  <- 新最优"
                print(f"[iter {r}.{i}] score={res['score']:.3f} "
                      f"tokens_avg={res['tokens_avg']} "
                      f"recall={res['evidence_recall']:.2f}{marker}")

        # Pareto 前沿（供报告展示"性能-成本"权衡）
        front = self.pareto_front(self.history)
        print(f"[Pareto] 前沿 {len(front)} 个候选（score vs tokens 互不支配）")

        # 最终：最优配置在测试集上验证（调参从未见过这些题）
        test_res = self.evaluate(best_cfg, self.test_items, tag="final_test")
        base_test = EvalRunner(self.st, HarnessConfig.from_dict(
            config.DEFAULT_HARNESS), audit=self.audit).run(self.test_items)

        summary = {
            "best_config": best_cfg.as_dict(),
            "search_score_best": best_score,
            "pareto_front": [{"cfg": h["cfg"], "score": h["score"],
                              "tokens_avg": h["tokens_avg"]} for h in front],
            "test": {
                "default_config": {k: base_test[k] for k in
                                   ("score", "evidence_recall", "keyword_hit",
                                    "citation_valid", "tokens_avg", "blocked_n")},
                "best_config": {k: test_res[k] for k in
                                ("score", "evidence_recall", "keyword_hit",
                                 "citation_valid", "tokens_avg", "blocked_n")},
            },
        }
        with open(os.path.join(config.META_DIR, "best_config.json"), "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        return summary
