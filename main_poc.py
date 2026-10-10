# -*- coding: utf-8 -*-
"""一键入口：按顺序运行 DEMO1-6。

用法：
  python main_poc.py            # 全部
  python main_poc.py --demo 3   # 只跑 DEMO3
  python main_poc.py --demo 5 --proposer llm   # DEMO5 用本地模型当 proposer
"""
import argparse
import os
import subprocess
import sys

DEMOS = {
    1: "demo1_smoke.py",
    2: "demo2_harness_livefire.py",
    3: "demo3_qa_eval.py",
    4: "demo4_token_compare.py",
    5: "demo5_meta_harness.py",
    6: "demo6_sepsis_case.py",
    7: "demo7_ablation.py",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", type=int, choices=sorted(DEMOS), default=None)
    ap.add_argument("--proposer", choices=["systematic", "llm"],
                    default="systematic")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--full", action="store_true",
                    help="DEMO7 跑全部 25 题（默认 test 集）")
    args = ap.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(root, "viking_kb.sqlite3")):
        print("未检测到知识库，先执行 ingest.py …")
        subprocess.run([sys.executable, "ingest.py"], cwd=root)

    which = [args.demo] if args.demo else sorted(DEMOS)
    for d in which:
        cmd = [sys.executable, DEMOS[d]]
        if d == 5:
            cmd += ["--proposer", args.proposer, "--rounds", str(args.rounds)]
        if d == 7:
            cmd += ["--full"] if args.full else []
        print(f"\n\n>>> 运行 DEMO{d}: {DEMOS[d]}\n")
        subprocess.run(cmd, cwd=root)

    print("\n全部选定 DEMO 运行完毕。输出物：")
    for f in sorted(os.listdir(os.path.join(root, "output"))):
        print("  output/" + f)


if __name__ == "__main__":
    main()
