# 脓毒症医疗指南问答 Agent POC：VikingRAG + STAIR/向量 RAG 消融 + Harness + Meta-Harness



📋【临床快速摘要（供科室主任、临床评审快速阅读）】

本项目是**脓毒症预警课题配套的回顾性科研POC原型，不用于实时患者诊疗**。
现有AI脓毒症预警大多只输出风险分数，缺少权威指南依据，误报、漏报发生后难以复盘质控。

本POC目标：

1. 对接预警输出，依据SSC、SOFA等脓毒症权威指南，完成病例与指南条文的证据匹配；
2. 对AI评估结果做证据溯源，识别证据缺口，生成可核查的评估报告；
3. 对历史漏报、误报病例做离线回顾复盘，服务于ICU临床科研与质控工作。

本原型重点解决：证据溯源、防止AI编造引用、输出越界内容拦截、可审计可复现；全部为离线历史病例仿真，数据本地运行。下面为详细技术说明与工程文档。



**项目简介：**

现有的医疗AI预警大多只输出风险分值，缺少指南依据溯源，误报漏报后难以自动化复盘质控。本POC为**面向ICU脓毒症辅助评估的回顾性科研原型**，旨在解决传统医疗大模型RAG系统普遍存在的几个现实痛点：**指南层级结构丢失、安全仅靠Prompt约束、Agent参数人工调参、大文档Token开销高、引用伪造难以拦截。**



**项目背景与要解决的业务问题：**

脓毒症是ICU高致死性急症，临床AI预警模型目前大多输出一个风险分数，但普遍存在短板：

1. 预警输出缺少对应的**指南条文证据溯源**，只给风险结论，医生无法快速核验AI判断依据；
2. 遇到复杂、边界病例，预警发生漏报/误报之后，缺少自动化手段回溯：到底是哪一条指南条款没有被充分参考；
3. 传统RAG方案处理临床指南时丢失文档层级结构，检索证据碎片化，容易编造虚假引用；
4. 安全约束依赖大模型Prompt，存在越界输出诊断、治疗建议的风险；
5. Agent多轮检索、评估的调度参数依赖人工试错调优，缺少可审计的自动化寻优方案；
6. 直接把全部指南全文灌入大模型，Token开销巨大，不利于多轮复杂病例证据核查。
   
   

**本POC的业务目标**：

**为脓毒症预警系统搭建一套指南证据核查与复盘底座**。当预警模型输出风险提示之后，本Agent可以依据SSC脓毒症指南、SOFA评分等权威文档，完成：病例信息‑指南条款的证据匹配、核验预警判断的依据、识别证据缺口、输出可溯源的评估记录；同时支持对历史误报、漏报病例做回顾复盘，为临床质控提供材料。



本原型参考 VikingRAG、Meta‑Harness、STAIR 等前沿推理架构思想，基于本地开源大模型构建完整闭环：

1. URI文档树存储临床指南，缺口驱动多轮证据检索，压缩上下文Token；
2. Harness确定性安全层：工具校验、路径防护、伪造引用拦截、临床红线护栏（安全规则冻结不可修改）；
3. Meta‑Harness元调参，搜索/测试集隔离，自动优化调度参数；
4. 多组消融对照实验，全量Agent轨迹审计留痕；
5. 纯本地运行，整套Demo可一键复现。

知识库为公开指南要点的学习性整理，患者为虚构数据；全部输出仅供 POC 验证，不构成医疗建议。

## 一、策划：这个例子回答医院的四个问题

| 问题                      | 机制                                                                                                          | DEMO                         |
| ----------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------- |
| 环境通吗、底座稳吗？              | 三级模型降级链 + 知识库体检 + 工具注册表                                                                                     | DEMO1 冒烟                     |
| **AI 管得住吗？**            | Harness：LLM 的每个工具调用过 schema 校验（白名单/参数/URI 命名空间/越界），答案引用必须来自实际读取的证据（编造引用拦截），无证据拒绝作答；护栏红线（诊断/医嘱/预后违禁词 + 免责声明） | DEMO2 实弹（14 用例含埋雷）           |
| **答得准、引得真、省 token 吗？**  | VikingRAG：URI 文档树 + Search/List/Grep/Read 证据缺口检索，结构 token 不随目录膨胀                                            | DEMO3 批量评估、DEMO4 对照 DeepRead |
| **Harness 调参能不靠人工试错吗？** | Meta-Harness：proposer 读全量历史（配置+分数+失败轨迹）提出候选，搜索集迭代，Pareto 前沿，测试集隔离验证，**安全红线全程冻结**                            | DEMO5 自动调参                   |
| 端到端像真的吗？                | 4 个模拟病例全流程：检索→作答→引用校验→护栏→报告落盘                                                                               | DEMO6                        |
| **比基线好多少？**             | 同一评测集三通道消融：证据召回率 / 幻觉率 / token 消耗（vs STAIR、普通向量 RAG）                                                        | DEMO7                        |
> 本节面向技术评审人员，说明本POC参考的前沿学术工作，临床阅读可以跳过。


## 二、与两项前沿工作的关系（诚实声明）

- **VikingRAG**（arXiv:2609.11390，RUC/Fudan）：本工程复现其三层机制——层次化 URI
  语义存储（结构搬出 prompt）、Search/List/Grep/Read 证据缺口驱动检索、
  经验边复用（E）与自适应升级（E+）。检索内核用 BM25 替代向量索引（零额外依赖，
  `storage.bm25_search()` 接口已隔离，可无缝换 sentence-transformers）。
- **Meta-Harness**（arXiv:2603.28052，Stanford IRIS + MIT）：复刻其外层循环思想
  （全量历史可见、搜索/测试集隔离、Pareto 前沿），但按医院场景做务实裁剪：
  搜索对象是**参数空间**（config.SEARCH_SPACE）而非自由代码——护栏、引用校验、
  轮数硬上限等安全不变量从机制上不可被 Meta 层修改（治理可审计）；
  proposer 默认 systematic（坐标下降，离线可复现），可选本地 Qwen3-7B 当 proposer。
- **STAIR**（arXiv:2609.03874，IBM）：原方法是微调 LLM 直接**生成叶章节标题**作为
  生成式检索（目录即寻址空间，Recall@1 82.6%，vs BM25 59.5% / DPR 68.7%）。
  本 POC 不做微调，做**零样本适配**：路由器（Qwen3-8B 读目录选叶节，离线用标题
  向量兜底）→ Read 叶节全部内容 → 作答。忠实保留其核心思想——目录结构是检索的
  第一公民——差异（未微调、未做 DSI 式生成打分）已在报告中声明，用于消融对照。

## 三、系统架构

```
提问 / 患者数据
   │
   ▼
┌─ 第一层（生成前）：越界请求识别 ── 处方/诊断/预后/病假 → 直接拒绝（红线）
│
├─ VikingRAG 检索层（被 Harness 包裹）
│    E+ 一轮路径：经验边种子 / Search 粗定位 → Read 验证（摘要必展开原文块）
│    └─ 充分性审查不过 → 升级：多轮证据缺口驱动检索（预算 B，工具调用全部过 Harness）
│
├─ 第二层（生成后）：Harness 校验
│    工具调用 schema 校验 / 无证据拒绝作答 / 引用编造拦截（确定性代码执行）
│
├─ 第三层（输出前）：护栏终检 ── 违禁词字典 + 必备免责声明
│
└─ 执行：报告落盘 output/ + 审计黑匣子 output/agent_trace.jsonl

外层：Meta-Harness 只调 SEARCH_SPACE 里的调度/检索/校验阈值，
      依据 = 搜索集分数 + 全量失败轨迹；测试集全程不可见。
```

模型通道（自动降级链）：**LM Studio（主，Vulkan 吃满 RX 6800M 12G）→
Ollama（兜底，CPU）→ 离线规则引擎（保链路可演示）**。
ZGCM-1（中关村人工智能研究院 7B 开源基座）槽位已在 `config.MODEL_SLOTS` 预留，
确定其 OpenAI 兼容 serving 方式后填入即可。

## 四、目录结构与模块清单

```
sepsis_guideline_agent_poc/
├── config.py                # 模型通道、默认 Harness 参数、SEARCH_SPACE、路径
├── ingest.py                # 知识库入库（data/kb/*.md -> SQLite）
├── main_poc.py              # 一键入口（--demo N 可单独跑）
├── demo1_smoke.py           # 冒烟测试
├── demo2_harness_livefire.py# Harness 校验实弹（14 用例）
├── demo3_qa_eval.py         # 指南问答批量评估（25 题）
├── demo4_token_compare.py   # VikingRAG vs DeepRead 对照
├── demo5_meta_harness.py    # Meta-Harness 自动调参
├── demo6_sepsis_case.py     # 端到端病例评估
├── demo7_ablation.py        # 三通道消融（VikingRAG / STAIR / 普通向量 RAG）
├── vikingrag/               # 检索引擎（storage/tools/retriever/experience/llm/prompts）
├── src/
│   ├── harness.py           # HarnessConfig + ToolCallValidator + EvidencePolicy + ToolGateway
│   ├── guardrails.py        # 红线护栏（冻结层）
│   ├── eval_runner.py       # 确定性评测器（recall/kw/引用/拒绝/token，支持三通道）
│   ├── baselines.py         # STAIR 通道 + 普通向量 RAG 通道（VectorIndex 带磁盘缓存）
│   ├── embedder.py          # 嵌入层：LM Studio nomic-embed-text / 离线哈希代理
│   ├── meta_harness.py      # Meta-Harness 外循环
│   └── audit.py             # 审计黑匣子（jsonl）
└── data/
    ├── kb/                  # 8 份结构化指南（Sepsis-3/SSC/SOFA/抗菌/液体/器官/通气/源控制）
    ├── qa/qa_set.json       # 25 题评测集（12 search + 13 test，含 3 道红线题）
    └── patients/            # 4 个虚构病例
```

## 五、快速开始（PyCharm，Python 3.10）

```bash
pip install -r requirements.txt
python ingest.py            # 入库（约 1 秒）
python main_poc.py          # 一键跑全部 DEMO（离线也能跑通）
```

模型接入（二选一，都不是必须的——离线兜底可演示全流程）：

- **LM Studio（推荐，GPU）**：加载 Qwen3-8B/2.5 3B，GPU Offload 拉满（Vulkan），
  打开 Local Server（默认 1234 端口），Context ≥ 8192；
- **Ollama（CPU 兜底）**：`ollama pull qwen3:8b` 后启动 `ollama serve`。

单跑：`python demo3_qa_eval.py` / `python demo5_meta_harness.py --proposer llm --rounds 2` /
`python demo7_ablation.py --full`（三通道全量消融）。

## 六、离线验证基准（本机实测，backend=offline 下界）

- DEMO2：14 个校验用例全部 PASS（含非法工具名、URI 越界/穿越、参数注入、
  无证据作答、编造引用、违禁词+免责声明）；

- DEMO3：25 题 score 0.881，evidence_recall 0.74，keyword_hit 0.74，
  citation_valid 1.00，blocked 0；tokens_avg ≈ 1276（DeepRead 基线 9173 的 13.9%）；

- DEMO5：Meta-Harness 一轮搜索即发现 `one_round_reads 2→3`，搜索集 0.869→0.989；
  测试集（全程不可见）score 0.894→0.926，token 1032→1413（Pareto 前沿如实展示
  性能-成本权衡）。产物：`output/meta_tuning/`（iterations.jsonl / traces/ / best_config.json）；

- DEMO7（三通道消融，全量 25 题，offline 下界；VikingRAG 通道用 DEMO5 调优配置）：
  
  | 通道            | evidence_recall | 幻觉率  | tokens_avg（DeepRead 9173 的占比） |
  | ------------- | --------------- | ---- | ----------------------------- |
  | VikingRAG     | 0.84            | 0.00 | 1692（18.4%）                   |
  | STAIR（ToC 寻址） | 0.88            | 0.00 | 1298（14.2%）                   |
  | 普通向量 RAG      | 0.84            | 0.00 | 1252（13.6%）                   |
  
  逐题记录：`output/ablation_report.json`。解读（诚实）：本知识库仅 48 个块、
  评测 25 题，三通道召回在此规模下趋同——STAIR 靠每题读约 10.6 个对象的
  "多读"换取召回，VikingRAG 的充分性升级/证据缺口多轮检索在离线启发式下
  发挥受限（Q07 长尾检索失败如实保留在报告中）；幻觉率三通道同为 0 的原因是
  离线作答=证据摘录+真实引用，编造引用拦截能力由 DEMO2 实弹用例单独证明。
  接入 Qwen3-8B 与 nomic-embed-text 在线后，充分性审查由 LLM 承担、检索决策
  由 LLM 提出针对性查询，三通道差距预期拉开——这正是 POC 要在医院环境
  复跑同一消融脚本（`python demo7_ablation.py --full`）的原因。

- 真模型（Qwen3-8B）在线后：答案质量、检索决策（LLM 提出针对性查询替代整段问题）
  与充分性审查（LLM 替代离线启发式）都会显著优于上述离线数字。

## 七、POC 验证报告大纲

1. 背景与业务问题（指南问答场景：章节/条款/分级，正是结构化文档树检索的主场）
2. 三条安全铁律 + 三层防线（生成前越界识别 / 生成后 Harness 双闸 / 输出前护栏终检）
3. 系统架构与模块清单（本文第二节图）
4. VikingRAG 机制对照表（URI 存储/四工具/证据缺口/经验边/E+）
5. 模型选型（LM Studio Vulkan 利用 6800M vs 上一版纯 CPU 的延迟对比；ZGCM-1 预留）
6. 实验设计与结果：DEMO1-6 逐一给出运行输出与逐行解读（本工程所有输出可复现）
7. Meta-Harness：调参自动化 + 安全不变量冻结的治理设计 + 搜索/测试集隔离证据
8. **消融实验（Ablation）**：同一脓毒症评测集三通道对照——VikingRAG /
   STAIR（ToC 目录寻址，零样本适配，注明与原文微调式生成检索的差异）/
   普通向量 RAG；指标 = 证据召回率、幻觉率（编造引用 + 护栏拦截）、token 消耗
   （对 DeepRead 全量精读归一化）；离线复跑脚本 + 医院环境在线复跑计划
9. 可观测性：agent_trace.jsonl 黑匣子 + meta_tuning 全量轨迹 + ablation_report.json
10. 数据安全：全本地、模拟数据、数据不出院
11. 局限性（诚实清单）：BM25 替代向量索引、离线数字是下界、评测集规模 25 题、
    知识库为要点整理非全文、STAIR 未微调、小规模知识库下三通道召回趋同、
    Q07 类长尾检索仍失败（已在测试集如实呈现）

## 八、已知边界

- 评测集 25 题是 POC 规模，结论不可外推为生产指标；扩充时保持 search/test 隔离。
- 离线规则引擎只保证链路可演示；一切质量数字以 Qwen3-8B 在线为准。
- 离线嵌入用哈希签名代理向量（`src/embedder.py`，可复现），仅保证消融流程可跑；
  真实对比用 LM Studio 加载的 nomic-embed-text-v1.5（非对称前缀已在代码中处理）。
- STAIR 通道为**零样本适配**（路由器选节而非微调生成式检索），用于机制级对照，
  不代表 STAIR 原文方法的上限。
- 本工程是测试性 POC，不构成任何医疗建议；接入真实指南全文前需完成合规审查。
