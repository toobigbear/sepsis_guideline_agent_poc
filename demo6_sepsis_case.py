# -*- coding: utf-8 -*-
"""DEMO6 端到端脓毒症病例评估（4 个模拟患者）。

链路：VikingRAG-E+ 检索取证 -> LLM 作答 -> Harness 引用校验 ->
护栏终检（违禁词+免责声明）-> 报告落盘 output/case_report_P00X.md。
"""
import glob
import json
import os

import config
from vikingrag.llm import LLM
from vikingrag.prompts import build_question
from vikingrag.storage import Storage

from src.audit import AuditLogger
from src.guardrails import Guardrail
from src.harness import EvidencePolicy, HarnessConfig, ToolGateway
from vikingrag.experience import VikingRAGEPlus


def main() -> None:
    print("=" * 70)
    print("DEMO6 端到端脓毒症病例评估")
    print("=" * 70)
    st = Storage(config.DB_PATH)
    audit = AuditLogger()

    for path in sorted(glob.glob(os.path.join(config.PATIENT_DIR, "*.json"))):
        patient = json.load(open(path, encoding="utf-8"))
        question = build_question(patient)
        print(f"\n■ {patient['patient_id']}  {patient['chief_complaint']}")

        # 第一道防线：越界请求识别
        if Guardrail.is_boundary_request(question):
            ans = Guardrail.refuse(question)
            print("  [护栏] 越界请求，已拒绝")
            path_name = "护栏拒绝"
            tokens, uris = 0, []
        else:
            gateway = ToolGateway(st, audit=audit)
            agent = VikingRAGEPlus(st, params=HarnessConfig.from_dict(
                config.DEFAULT_HARNESS).as_dict(),
                executor=gateway.execute, audit=audit)
            ans, trace, path_name, verdict = agent.run(question)
            tokens, uris = trace.prompt_tokens, trace.read_uris()
            print(f"  [{path_name}] token={tokens} 证据{len(uris)}条 "
                  f"LLM调用{trace.llm_calls}次")

            # 第二道防线：编造引用拦截（一次"反馈-修订"机会，仍不通过才硬拦）
            violations = EvidencePolicy.verify_citations(ans, uris)
            if violations:
                audit.log("HARNESS_REPAIR_TRY", patient=patient["patient_id"],
                          reason=f"未取证引用 {violations}")
                fixed = LLM.chat([
                    {"role": "system", "content":
                     "你是答案修订器。给定评估答案与允许引用的证据 URI 白名单，"
                     "删除或改写所有不在白名单内的 viking:// 引用，"
                     "其余内容与结论保持不变，只输出修订后的完整答案。"},
                    {"role": "user", "content":
                     f"允许引用的 URI 白名单：{uris}\n\n"
                     f"不在白名单内的违规引用：{violations}\n\n"
                     f"原答案：\n{ans}\n\n请输出修订后的完整答案，不要解释过程。"},
                ], max_tokens=1800)
                if fixed and not EvidencePolicy.verify_citations(fixed, uris):
                    ans = fixed
                    audit.log("HARNESS_REPAIR_OK", patient=patient["patient_id"])
                else:
                    ans = "【Harness】答案含未取证引用，已拦截，请人工复核。"
                    audit.log("HARNESS_BLOCK", patient=patient["patient_id"],
                              reason=f"编造引用 {violations}")

        # 第三道防线：执行层先硬追加免责声明，再过违禁词 + 免责声明终检
        if config.DISCLAIMER not in ans:
            ans = ans + "\n" + config.DISCLAIMER
        gr = Guardrail.final_check(ans)
        if not gr.ok:
            audit.log("GUARDRAIL_BLOCK", patient=patient["patient_id"],
                      reason=gr.reasons())
            print(f"  [护栏拦截] {gr.reasons()}")
            ans = "【已拦截】生成内容越界，事件已留痕。" + config.DISCLAIMER

        report = (f"# 病例评估报告 {patient['patient_id']}\n\n"
                  f"- 路径: {path_name}\n- prompt token: {tokens}\n"
                  f"- 证据 URI: {uris}\n- 护栏: {'通过' if gr.ok else '拦截'}\n\n"
                  f"## 评估意见\n{ans}\n")
        out = os.path.join(config.OUTPUT_DIR,
                           f"case_report_{patient['patient_id']}.md")
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  报告: {out}")
        print(f"  结论: {ans.strip().splitlines()[0][:60]}…")

    print(f"\nLLM backend={LLM.backend}；审计: {config.AUDIT_PATH}\n")
    st.close()


if __name__ == "__main__":
    main()
