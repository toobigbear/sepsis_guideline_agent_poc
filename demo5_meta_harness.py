# -*- coding: utf-8 -*-
"""DEMO5 Meta-Harness：在医疗评测集上自动搜索 Harness 参数。

回答的问题：Harness 调参能不能不靠人工试错？
- proposer 读取全量历史（配置+分数+失败题目轨迹）提出新候选；
- 只在 search split（12 题）上迭代，test split（13 题）全程不可见；
- 安全红线（护栏/引用校验/轮数硬上限）冻结，Meta 层不可触碰；
- 输出 Pareto 前沿与测试集 before/after 对照。

用法：
  python demo5_meta_harness.py              # systematic proposer（默认，离线可跑）
  python demo5_meta_harness.py --proposer llm --rounds 2   # 本地 Qwen3-7B 当 proposer
"""
import argparse

import config
from vikingrag.llm import LLM
from vikingrag.storage import Storage

from src.audit import AuditLogger
from src.meta_harness import MetaHarness


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposer", choices=["systematic", "llm"],
                    default="systematic")
    ap.add_argument("--rounds", type=int, default=2)
    args = ap.parse_args()

    print("=" * 70)
    print(f"DEMO5 Meta-Harness 自动调参（proposer={args.proposer}）")
    print("=" * 70)
    st = Storage(config.DB_PATH)
    mh = MetaHarness(st, proposer=args.proposer, audit=AuditLogger())
    summary = mh.optimize(rounds=args.rounds)

    print("\n----- 测试集对照（调参从未见过这些题）-----")
    d, b = summary["test"]["default_config"], summary["test"]["best_config"]
    print(f"{'指标':16s}{'人工默认配置':>14s}{'Meta 最优':>14s}")
    for k in ("score", "evidence_recall", "keyword_hit", "citation_valid",
              "tokens_avg", "blocked_n"):
        print(f"{k:16s}{d[k]:>14}{b[k]:>14}")
    print(f"\n最优配置: {summary['best_config']}")
    print(f"Pareto 前沿 {len(summary['pareto_front'])} 个候选，"
          f"明细见 output/meta_tuning/（iterations.jsonl / traces/ / best_config.json）")
    print(f"LLM backend={LLM.backend}；全程安全红线冻结，变更仅限调度/检索/校验阈值。\n")
    st.close()


if __name__ == "__main__":
    main()
