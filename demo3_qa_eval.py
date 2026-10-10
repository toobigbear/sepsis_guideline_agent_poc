# -*- coding: utf-8 -*-
"""DEMO3 医疗指南问答批量评估（默认配置，全量 25 题）。

回答的问题：答得准吗、引得真吗、省 token 吗？
输出逐题明细 + 汇总指标，落盘 output/qa_eval_report.json。
"""
import json

import config
from vikingrag.llm import LLM
from vikingrag.storage import Storage

from src.audit import AuditLogger
from src.eval_runner import EvalRunner, load_qa_set
from src.harness import HarnessConfig


def main() -> None:
    print("=" * 70)
    print("DEMO3 医疗指南问答批量评估（默认 Harness 配置）")
    print("=" * 70)
    st = Storage(config.DB_PATH)
    items = load_qa_set()
    runner = EvalRunner(st, HarnessConfig.from_dict(config.DEFAULT_HARNESS),
                        audit=AuditLogger())
    result = runner.run(items)

    print(f"LLM backend={LLM.backend}  题数={result['n']}  "
          f"基线(DeepRead)={runner.baseline} token\n")
    print(f"{'题':5s}{'类别':14s}{'路径':26s}{'recall':>7s}{'kw':>6s}{'cit':>5s}"
          f"{'token':>7s}{'轮':>4s} 备注")
    for r in result["records"]:
        print(f"{r['item_id']:5s}{r['category']:14s}{r['path'][:25]:26s}"
              f"{r['evidence_recall']:>7.2f}{r['keyword_hit']:>6.2f}"
              f"{r['citation_valid']:>5.0f}{r['tokens']:>7d}{r['rounds']:>4d} "
              f"{r['blocked'][:24]}")
    print("-" * 70)
    print(f"汇总: score={result['score']:.3f}  evidence_recall={result['evidence_recall']:.3f}  "
          f"keyword_hit={result['keyword_hit']:.3f}  citation_valid={result['citation_valid']:.2f}  "
          f"refusal_correct={result['refusal_correct']:.2f}")
    print(f"      tokens_avg={result['tokens_avg']}（基线 {runner.baseline}，"
          f"{result['tokens_avg'] / runner.baseline * 100:.1f}%）  "
          f"rounds_avg={result['rounds_avg']}  blocked={result['blocked_n']}")

    out = {"llm_backend": LLM.backend, "baseline_tokens": runner.baseline,
           "harness": config.DEFAULT_HARNESS,
           "summary": {k: v for k, v in result.items() if k != "records"},
           "records": result["records"]}
    with open(config.OUTPUT_DIR + "/qa_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n报告已写入 output/qa_eval_report.json，审计留痕 {config.AUDIT_PATH}\n")
    st.close()


if __name__ == "__main__":
    main()
