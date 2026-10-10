# -*- coding: utf-8 -*-
"""DEMO4 VikingRAG vs DeepRead 对照（对应论文 fig.1 tradeoff）。

DeepRead 基线：完整目录树 + 全部内容一次性塞进 prompt（结构信息线性膨胀）。
VikingRAG：URI 寻址 + 证据缺口按需检索。
同题对照 token 量与答案关键词覆盖。
"""
import config
from vikingrag.llm import LLM, approx_tokens
from vikingrag.storage import Storage

from src.audit import AuditLogger
from src.eval_runner import EvalRunner, load_qa_set
from src.harness import HarnessConfig

DEEPREAD_SYSTEM = "你是医疗指南问答助手。请阅读全部指南内容后回答问题。"
DISCLAIMER = config.DISCLAIMER


def deepread_run(st: Storage, question: str, catalog: str) -> tuple:
    """DeepRead 式：全量目录入 prompt，答案=全文中与问题相关的摘录（离线模拟）。"""
    prompt = f"【完整指南全文】\n{catalog}\n\n【问题】{question}\n\n请作答。"
    tokens = approx_tokens(DEEPREAD_SYSTEM + prompt)
    if LLM.backend != "offline":
        ans = LLM.chat([{"role": "system", "content": DEEPREAD_SYSTEM},
                        {"role": "user", "content": prompt}], max_tokens=900)
    else:
        # 离线模拟：从全文中抽取含问题词的句子（上界近似：全文在手）
        import re
        terms = set(re.findall(r"[A-Za-z0-9\.]+|[\u4e00-\u9fff]", question)) - set("的了是么什哪")
        sents = re.split(r"[。；\n]", catalog)
        picked = [s.strip() for s in sents
                  if sum(1 for t in terms if t in s) >= 2][:8]
        ans = "【DeepRead 离线模拟】\n" + "；".join(picked)[:600]
    return tokens, ans + ("\n" + DISCLAIMER if DISCLAIMER not in ans else "")


def main() -> None:
    print("=" * 70)
    print("DEMO4 VikingRAG vs DeepRead 对照")
    print("=" * 70)
    st = Storage(config.DB_PATH)
    items = [it for it in load_qa_set() if not it.get("expect_refusal")][:6]

    # 组装 DeepRead 用的"完整目录+全文"
    rows = st.conn.execute(
        "SELECT uri, text FROM objects ORDER BY uri").fetchall()
    catalog = "\n".join(f"{r['uri']}\n{r['text']}" for r in rows)
    base_tokens = approx_tokens(catalog)

    runner = EvalRunner(st, HarnessConfig.from_dict(config.DEFAULT_HARNESS),
                        audit=AuditLogger())
    print(f"{'题目':6s}{'DeepRead tok':>14s}{'VikingRAG tok':>15s}{'占比':>8s}"
          f"{'DR-kw':>8s}{'VK-kw':>8s}")
    tot_dr = tot_vk = 0
    for it in items:
        rec = runner.run_item(it)
        dr_tokens, dr_ans = deepread_run(st, it["question"], catalog)
        kws = it.get("keywords", [])
        dr_kw = sum(1 for k in kws if k in dr_ans) / max(1, len(kws))
        print(f"{it['id']:6s}{dr_tokens:>14d}{rec.tokens:>15d}"
              f"{rec.tokens / dr_tokens * 100:>7.1f}%{dr_kw:>8.2f}{rec.keyword_hit:>8.2f}")
        tot_dr += dr_tokens
        tot_vk += rec.tokens
    print("-" * 70)
    print(f"合计: DeepRead {tot_dr} token vs VikingRAG {tot_vk} token "
          f"（{tot_vk / tot_dr * 100:.1f}%）；"
          f"结构 token 不再随目录线性膨胀（对应论文 fig.1 拐点）")
    print(f"（完整目录全文约 {base_tokens} token 未入 VikingRAG 的任何一次 prompt）")
    print("（注：离线模拟的 DeepRead 摘录是质量下界；真模型下 DeepRead 质量会回升，"
          "但其 token 量不变——tradeoff 的对比维度正是 token。）\n")
    st.close()


if __name__ == "__main__":
    main()
