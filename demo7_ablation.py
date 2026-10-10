# -*- coding: utf-8 -*-
"""Demo 7：消融实验 —— VikingRAG / STAIR / 普通向量 RAG 三通道对照。

同一脓毒症指南评测集（data/qa/qa_set.json）、同一评分函数、
同一护栏与引用校验，三条检索通道各跑一遍，输出对照表：

  evidence_recall  金牌证据 URI 命中率（结构定位能力）
  keyword_hit      金牌关键词覆盖率（答案质量）
  hallucination    幻觉/越界率 = 编造引用率 + 护栏拦截率（越低越好）
  tokens_avg       平均 token 消耗，并给出 DeepRead 全量精读基线的占比

用法：
  python main_poc.py --demo 7            # 三个通道跑 test 集（12 题）
  python main_poc.py --demo 7 --full     # 三个通道跑全部 25 题

结果落盘 output/ablation_report.json（含逐题记录，可评审回放）。
"""
from __future__ import annotations

import json
import os

import config
from vikingrag.storage import Storage

from src.eval_runner import EvalRunner, load_qa_set
from src.harness import HarnessConfig

CHANNELS = [
    ("vikingrag", "VikingRAG（结构化文档树 + 证据缺口多轮检索）"),
    ("stair",     "STAIR（ToC 目录寻址，零样本适配）"),
    ("plain_rag", "普通向量 RAG（无结构、单轮 top-k）"),
]

FULL_SPLIT_ITEMS = 25  # 含 3 道红线题（三通道同走护栏拒绝）


def run_ablation(full: bool = False) -> dict:
    if not os.path.exists(config.DB_PATH):  # 直接运行时自动入库（main_poc 已含同样逻辑）
        import subprocess, sys
        subprocess.run([sys.executable, os.path.join(config.ROOT, "ingest.py")],
                       cwd=config.ROOT)
    st = Storage(config.DB_PATH)
    # VikingRAG 通道用 Meta-Harness 在 search 集上调出的最优配置
    # （demo5 产物 output/meta_tuning/best_config.json；缺席时用默认配置）
    cfg = HarnessConfig.from_dict(config.DEFAULT_HARNESS)
    cfg_source = "DEFAULT_HARNESS（未经调优）"
    best_path = os.path.join(config.META_DIR, "best_config.json")
    if os.path.exists(best_path):
        try:
            best = json.load(open(best_path, encoding="utf-8"))
            params = (best.get("best_config") or best.get("config")
                      or best.get("harness") or best)
            cfg = HarnessConfig.from_dict({**config.DEFAULT_HARNESS,
                                           **{k: v for k, v in params.items()
                                              if k in config.DEFAULT_HARNESS}})
            cfg_source = "Meta-Harness 最优配置（demo5 产物）"
        except Exception:
            pass
    items = load_qa_set() if full else load_qa_set(split="test")
    baseline = max(1, st.full_catalog_tokens())  # DeepRead 全量精读基线

    report = {"split": "all" if full else "test",
              "n_items": len(items),
              "deepread_baseline_tokens": baseline,
              "vikingrag_config": cfg_source,
              "channels": {}}
    print(f"评测集: {report['split']}（{len(items)} 题） | "
          f"DeepRead 全量精读基线: {baseline} tokens")
    print(f"VikingRAG 通道配置: {cfg_source}")
    print("-" * 92)

    rows = []
    for channel, desc in CHANNELS:
        runner = EvalRunner(st, cfg, channel=channel)
        summary = runner.run(items)
        n = max(1, summary["n"])
        fab = sum(1 for r in summary["records"]
                  if "编造引用" in (r["blocked"] or "")) / n
        guard = sum(1 for r in summary["records"]
                    if r["blocked"] and "编造引用" not in r["blocked"]) / n
        halluc = round(fab + guard, 4)
        tok_pct = round(100.0 * summary["tokens_avg"] / baseline, 1)
        report["channels"][channel] = {
            "desc": desc, "summary": summary,
            "hallucination_rate": halluc,
            "fabrication_rate": round(fab, 4),
            "guardrail_block_rate": round(guard, 4),
            "tokens_pct_of_deepread": tok_pct,
        }
        rows.append((desc, summary, halluc, tok_pct))
        print(f"[{channel}] {desc}")
        print(f"  evidence_recall={summary['evidence_recall']:.4f}  "
              f"keyword_hit={summary['keyword_hit']:.4f}  "
              f"score={summary['score']:.4f}")
        print(f"  hallucination_rate={halluc:.4f}（编造引用 {fab:.4f} + "
              f"护栏拦截 {guard:.4f}）")
        print(f"  tokens_avg={summary['tokens_avg']}（DeepRead 的 {tok_pct}%）  "
              f"rounds_avg={summary['rounds_avg']}")
        print("-" * 92)

    path = os.path.join(config.OUTPUT_DIR, "ablation_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n对照表（幻觉率越低越好，token 占比越低越好）：")
    print(f"{'通道':<44}{'召回率':>8}{'幻觉率':>8}{'token%':>8}")
    for desc, summary, halluc, tok_pct in rows:
        print(f"{desc:<44}{summary['evidence_recall']:>8.4f}"
              f"{halluc:>8.4f}{tok_pct:>7.1f}%")
    print(f"\n逐题记录与完整指标已保存：{path}")
    return report


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="跑全部 25 题（默认只跑 test 集 12 题）")
    args = ap.parse_args()
    run_ablation(full=args.full)


if __name__ == "__main__":
    main()
