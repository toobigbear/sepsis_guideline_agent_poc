"""全局配置：模型服务通道、Harness 参数、路径。

模型通道（自动降级链）：
  1. LM Studio  http://localhost:1234/v1   —— 主通道，Vulkan 吃满 RX 6800M 12G
  2. Ollama     http://localhost:11434/v1  —— 兜底通道（纯 CPU 可用）
  3. OfflineBrain                          —— 规则引擎，保证链路可演示

ZGCM-1（中关村人工智能研究院 7B 开源基座）接入方式待定，
已在 MODEL_SLOTS 预留槽位：确定其 OpenAI 兼容地址后填入即可加入降级链。
"""
from __future__ import annotations

import os

# ---------------- 模型通道 ----------------
LMSTUDIO_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
LMSTUDIO_API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

# 模型槽位（名称需与服务端实际加载的模型标识一致）
MODEL_SLOTS = {
    "qwen3-8b": {          # 主评估模型（LM Studio，GPU）
        "provider": "lmstudio",
        "model": os.environ.get("LMSTUDIO_MODEL", "qwen3-8b"),
    },
    "qwen2.5-3b": {        # 联调/兜底小模型
        "provider": "lmstudio",
        "model": os.environ.get("LMSTUDIO_MODEL_SMALL", "qwen2.5-3b-instruct"),
    },
    "zgcm-1": {            # 预留槽位：待中关村智能研究院的基座模型 ZGCM-1  serving 方式确定后填写
        "provider": "lmstudio",
        "model": "zgcm-1",
        "disabled": True,
    },
}
ACTIVE_MODEL = os.environ.get("POC_MODEL", "qwen3-8b")# ← 把 "qwen3-8b" 改成 "qwen2.5-3b"

# ---------------- 向量模型（基线通道用） ----------------
# LM Studio 已加载的 embedding 模型标识（与 LM Studio 内显示一致）。
# nomic-embed-text 为非对称模型：查询/文档分别加 search_query: / search_document: 前缀。
EMBEDDING_MODEL = os.environ.get(
    "POC_EMBED_MODEL", "YorkieOH10/nomic-embed-text-v1.5-Q8_0-GGUF")
EMBEDDING_DIM = 768

# ---------------- 默认 Harness 参数（Meta-Harness 的搜索起点） ----------------
# 注意：这里是"调度与校验参数"，安全红线（guardrails.py）不在搜索空间内。
DEFAULT_HARNESS = {
    "top_k": 5,                # Search 返回条数
    "round_budget": 15,        # 证据缺口驱动检索的轮数预算 B（上限 25，冻结）
    "history_compress": 3500,  # 交互历史压缩阈值（字符）
    "experience_min_sim": 1.5, # 经验边复用相似度阈值
    "experience_min_overlap": 2,  # 经验边命中的最小共享词数
    "seed_reads": 6,           # 经验边种子最多读入条数
    "one_round_reads": 2,      # E+ 一轮路径的 Search 粗读条数
    "evidence_min_reads": 2,   # 充分性审查的最低证据条数（离线启发式）
    "numeric_threshold": 0.85, # JEV 数字一致度放行阈值（越高越严）
    "prompt_variant": "A",     # 检索提示词变体（A/B 两版供 Meta-Harness 搜索）
    "answer_temperature": 0.0,
}

# Meta-Harness 搜索空间（param -> 候选值列表；不在此表的参数不可被自动修改）
SEARCH_SPACE = {
    "top_k": [3, 5, 8, 12],
    "round_budget": [10, 15, 20],
    "history_compress": [2500, 3500, 5000],
    "experience_min_sim": [1.0, 1.5, 2.5],
    "experience_min_overlap": [2, 3],
    "seed_reads": [4, 6, 8],
    "one_round_reads": [1, 2, 3, 4],
    "evidence_min_reads": [1, 2, 3],
    "numeric_threshold": [0.8, 0.85, 0.9],
    "prompt_variant": ["A", "B"],
}

# ---------------- 路径 ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
KB_DIR = os.path.join(ROOT, "data", "kb")
QA_PATH = os.path.join(ROOT, "data", "qa", "qa_set.json")
PATIENT_DIR = os.path.join(ROOT, "data", "patients")
DB_PATH = os.path.join(ROOT, "viking_kb.sqlite3")
OUTPUT_DIR = os.path.join(ROOT, "output")
AUDIT_PATH = os.path.join(OUTPUT_DIR, "agent_trace.jsonl")
META_DIR = os.path.join(OUTPUT_DIR, "meta_tuning")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

DISCLAIMER = "本回答由 AI 辅助生成，仅供临床参考，不能替代医师判断。"
