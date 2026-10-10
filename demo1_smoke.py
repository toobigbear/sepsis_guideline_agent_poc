# -*- coding: utf-8 -*-
"""DEMO1 冒烟测试：LLM 通道探测 + 知识库体检 + 工具注册表 + 护栏自检。

回答的问题：环境通吗？底座稳吗？（不带 AI 推理，全部确定性输出）
"""
import os

import config
from vikingrag.llm import LLM
from vikingrag.storage import Storage

from src.guardrails import FORBIDDEN_PATTERNS, DISCLAIMER, Guardrail


def main() -> None:
    print("=" * 70)
    print("DEMO1 冒烟测试")
    print("=" * 70)
    print(f"[1] LLM 通道: backend={LLM.backend}  model={LLM.model}  slot={LLM.slot}")
    if LLM.backend == "offline":
        print("    （LM Studio / Ollama 均未连通，已降级规则引擎，链路仍可演示）")

    if not os.path.exists(config.DB_PATH):
        raise SystemExit("请先运行 python ingest.py")
    st = Storage(config.DB_PATH)
    n_obj = st.conn.execute("SELECT COUNT(*) n FROM objects").fetchone()["n"]
    n_node = st.conn.execute("SELECT COUNT(*) n FROM nodes").fetchone()["n"]
    n_docs = st.conn.execute(
        "SELECT COUNT(DISTINCT doc) n FROM nodes").fetchone()["n"]
    print(f"[2] 知识库: {n_docs} 份文档, {n_node} 个结构节点, {n_obj} 个索引对象, "
          f"基线 {st.full_catalog_tokens()} token")

    from vikingrag.tools import ToolKit
    print(f"[3] 工具注册表: {ToolKit(st).catalog()}")

    sample = "该患者可诊断为脓毒症，建议使用头孢他啶静滴，预后不良"
    hits = Guardrail.scan(sample)
    print(f"[4] 护栏自检: 违禁词字典 {len(FORBIDDEN_PATTERNS)} 条; "
          f"对埋雷样本命中 {len(hits)} 个 -> {hits}")
    print(f"    免责声明: {DISCLAIMER[:30]}…")
    print("[5] 边界识别: ",
          "处方请求" if Guardrail.is_boundary_request("给我一张抗生素处方")
          else "漏检!", "|",
          "指南问题放行" if not Guardrail.is_boundary_request("SOFA 评分标准是什么")
          else "误伤!")
    st.close()
    print("\n结论: 冒烟测试通过（环境/底座/工具/护栏均就绪）\n")


if __name__ == "__main__":
    main()
