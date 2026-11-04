# Movie Agent Skeleton

当前实现是一套静态 LangGraph Plan-and-Execute 骨架：

~~~text
analyze_movie → controller
                   ├─ analyze_request
                   ├─ resolve_next
                   ├─ build_plan
                   ├─ execute_action
                   └─ finalize

resolve_next / plan step
        → execute_action → apply_result → controller
~~~

## 设计边界

- MovieAnalyzer 区分非电影、电影泛请求和具体作品请求，提取实际目标、参照作品及有原文依据的用户限定；名称提示只供检索。
- RequestAnalyzer 使用已选电影和澄清记录重述需求。
- PlanBuilder 只输出 PlanIntent；确定性编译器负责生成白名单 Action。
- Clarification 是 resolution 模式下的 Action，与普通执行共用 Registry 和 Executor。
- [架构 FigJam](https://www.figma.com/board/VR3oajHfyJKe5Z5pjLRZHB) 记录了此前的 RAG 架构；当前实现以本说明和代码为准。图 4–5 补充 LLM 调用 `ask_user` UI 工具的待实现设计。
- StatePatch 是 Action 修改 State 的唯一格式；Controller 根据 State 就绪程度路由，不保存 return_to。
- 默认注册 resolve_movie 和 collect_movie_reviews；传入 RagRuntime 后，再以扩展方式注册 build/query/aggregate 三个观点 Action。
- State 已使用 movie_targets 列表，但第一版只执行单目标；多个明确电影返回 unsupported。

## 公开入口

网页工作台入口：`python -u -m app.web --model gpt-5.4-mini`。前端启动、构建和兼容性约定见 [Web 工作台说明](../../frontend/README.md)。

~~~python
from app.agent.Agent import MovieAgent
from app.rag.runtime import build_openai_chat_model, build_openai_rag_runtime
from config import load_openai_api_key

api_key = load_openai_api_key()
llm = build_openai_chat_model(
    "YOUR_MODEL",
    api_key=api_key,
)
rag_runtime = build_openai_rag_runtime(llm, api_key=api_key)
agent = MovieAgent(llm, rag_runtime=rag_runtime)
state = await agent.run(user_query, thread_id="conversation-1")

if state.get("__interrupt__"):
    payload = state["__interrupt__"][0].value
    state = await agent.resume(
        {
            "issue_id": payload["issue_id"],
            "option_id": payload["options"][0]["option_id"],
        },
        thread_id="conversation-1",
    )
~~~

resume() 也兼容纯文本；运行时会将文本绑定到当前 issue_id。结构化回复可以同时提交 option_id 和补充文本。无效 issue 或 option 会在消费 interrupt 之前被拒绝。

analyze() 暂时是 run() 的兼容别名。默认最多允许三轮澄清，可通过 MovieAgent(..., max_clarification_rounds=N) 调整。

CLI 和 Web 入口会自动记录每次运行，默认保存到 `output/runs/<run_id>/events.jsonl` 和 `result.json`，可通过 `AGENT_DATA_DIR` 更改数据根目录。重试追加事件，最终结果原子替换。详细目录与恢复边界见 [Web 工作台说明](../../frontend/README.md#运行记录)。直接作为库调用时，可用同一上下文覆盖 `run` 和后续 `resume`：

~~~python
from app.run_log import record_run

with record_run(query=user_query, entrypoint="library") as recording:
    state = await agent.run(user_query, thread_id="conversation-1")
    # 如有澄清，继续在此上下文内调用 resume；完成后保存业务结果。
    if state.get("final_result") is not None:
        recording.write_result(state["final_result"])
~~~

## 计划与执行

Planner 的固定映射：

| Task | Actions |
|---|---|
| resolve_urls | resolve_movie |
| collect_reviews | resolve_movie → collect_movie_reviews |
| opinion_qa | resolve → collect → build → query → aggregate |
| platform_comparison | resolve → collect → build → query → aggregate |

识别后先在 resolution 模式查证电影，再进入需求理解和计划生成。Executor 根据 target_id 绑定工具参数；计划中的 resolve_movie 会复用已核实目标。

电影识别遵循以下黑盒合同：

- 非电影和未指向具体作品的电影泛请求返回 unsupported；想问某部作品但缺少线索时追问。
- 实际目标与参照作品分开，保留全部目标；明确多目标或整个系列时告知仅支持单部。
- 年份和部数必须带用户原文 quote，经代码校验后才成为过滤条件；模型记忆只能作为名称检索提示。
- Wikidata 支持中英文检索，读取系列成员、版本和序号，平台 ID 可缺失。搜索限额作用在实体层，避免多个上映日期挤掉其他电影。
- 仅裸片名即使只查到一项也先确认；明确 ID 或有外部数据支持的年份/部数可核实后继续。系列与首部同名时不自动选首部。
- 候选来自外部查询。澄清记录保留完整公开选项，支持“第二个”等明确选择；内部 patch 不进入历史消息。
- 查不到匹配作品时补充线索；服务故障抛出可重试异常。核实身份但缺少平台链接时，不伪装成作品不存在，采集阶段说明能力限制。

采集结果只在 State 中保存文件路径和数量。

未传入 RagRuntime 时，执行到观点 Action 仍会明确返回 not_implemented，便于离线测试或只启用采集能力。未预期的模型或工具异常不会伪装成业务状态。使用同一 thread_id 和原问题再次调用 run()，会从 checkpoint 的失败节点重试。

## 单电影临时 RAG

评论检索前，按已确认的 Wikidata QID 读取对应 Wikipedia 条目（优先英文，无英文条目时用中文）。
`app/rag/movie_context.py` 清理页面，保留简介、剧情、演员和制作章节，以及来源版本链接；
各章节独立限长，节选会明确标注。成功结果按 QID 缓存在进程中，不增加图节点、向量索引或 LLM 摘要调用。
这份 `movie_context` 用于查询改写、定向补检索、聚合和复核；全文背景随 RetrievalResult 留在进程内，State 仍只保存轻量检索结果。
读取失败记录 warning，并使用现有评论继续分析；无对应条目时不猜测页面。

Prompt 的分工是：查询改写用背景扩展人物/角色称呼，聚合联系评论细节、评价标准和角色处境，
复核分别检查电影事实、评论原意与分析推断。阿里云 reranker 根据原问题、明确方面和计划中的单条评论检索重点给候选原文排序。
背景不能补造观众立场或平台差异，也不进入评论引用和覆盖计数。
结合背景的解释需标明推断，并注明所用维基百科来源；引用仍须逐字存在于实际提供的评论阅读范围。

~~~text
collected JSON → normalize / deduplicate → paragraph / sentence passages + source offsets
    ↓
BM25 + normalized dense matrix (in memory)
    ↑
question → neutral queries in each platform's language
    ↓
BM25 + Dense → RRF → group hits by review → bounded source reading windows
    ↓
Aliyun reranker → select original review contexts by platform
    ↓
optional question-specific queries → at most one hybrid follow-up on unread text
    ↓
generate cited claims → audit with the same reading context → assemble answer
~~~

- 两个平台共用按实际长度决定的切分规则。默认片段最多 1800 字符，优先段落、其次句末，长度兜底；重叠最多约 200 字符，优先完整句子。完整评论不变，`start/end` 是原文中的半开区间。
- Embedding 仍使用标题与片段（最多 2000 字符）。BM25、精确向量搜索和 RRF 保留；索引只存在于本次运行时的内存中。
- 规划器参考维基百科条目中与问题相关的演员、角色、场景或制作信息，生成各平台的 `queries`（每平台最多两条）、单条评论的检索重点 `evidence_focus`，以及最多两个需要对照评论验证的 `analysis_questions`。
- 每个检索通道最多接纳 80 条独立评论的命中，融合后每平台最多阅读 80 条评论。多次命中不累加为一条评论的额外票数，按其最高片段融合分排序。
- 同条评论的命中片段优先合并，再用剩余预算补充相邻句子。每条评论的阅读范围合计最多 3600 字符；短评直接读全文，长评的远处命中保持分开的范围，不伪造成连续原文。
- 阿里云 `qwen3.7-text-rerank` 对每个平台的候选原文单独排序；两个平台可并发请求。没有逐批 LLM 观点抽取，也不预生成 `target/subaspect/stance/reason` 标签。
- 每平台最多保留 12 条独立评论的阅读范围，供生成与复核使用。同一评论不会因为命中多个片段而占用多个来源名额。
- `coverage.source_counts` 只统计独立评论，`sufficient` 表示达到配置中的最低来源数（默认 6），不承诺语义完整、证据正确或代表平台总体。
- 若仍有未读文本和检索预算，规划器结合首轮计划、百科资料和选中评论判断具体缺口。即使无需补检索，也可根据已读原文细化分析问题，供生成与复核共同使用。
- 补检索沿用 BM25 + Dense + RRF，最多一次。只有已选中供生成模型阅读的范围会被排除；仅参与重排但未选中的评论仍可被新的查询命中。
- 重排分数只在同一次请求内用于比较。同一平台的已选来源与新候选合并后重新请求重排，再选最终来源，不直接混用两轮分数。
- 生成模型一次性输出判断及其逐字引文；复核读取相同原文范围、电影背景和判断，检查评价对象、含义、条件、推断边界与反证关系。
- 代码检查来源 ID、引文是否同时存在于完整原文和一个连续阅读范围、支持与反证是否重复、跨平台判断是否具有双方支持，以及 insight 是否包含解释与限制。不会把重排分数当作观点可信度。
- 固定一次生成和一次复核，采用 `keep/revise/drop`；修订只能使用原判断已有的来源与引文，可收窄为单平台 finding。通过复核的判断直接组装报告，不再自由改写。
- 复核后只按实际引用且仍有效的独立评论重算来源数量。同一原评论的不同引文可以形成不同 EvidenceCard，但不会增加独立评论数。
- EvidenceCard 保留完整原文、逐字引文、阅读范围、原评论 ID 和稳定引用编号。LangGraph State 只存索引/检索 ID 和最终产物，完整运行时数据在任务结束时释放。

## 配置与运行

`OPENAI_API_KEY` 从项目根目录 `.env.local` 读取，用于向量化、查询规划、生成和复核。
`DASHSCOPE_API_KEY` 优先从进程环境读取，其次读取 `.env.local`，只用于阿里云重排；密钥缺失会在实际重排时明确报错。
IMDb/豆瓣 Cookie 从 `config/source_cookies.json` 读取。源码和测试不会打印凭据。

`config.py` 的默认 `rerank_model` 为 `qwen3.7-text-rerank`，`rerank_endpoint` 为：

~~~text
https://cn-beijing.maas.qianwenaiapi.com/api/v1/services/rerank/text-rerank/text-rerank
~~~

请求按平台发送 `input.query/documents`，`top_n` 等于该平台候选数，关闭原文回传；响应的索引必须唯一、完整且合法，相关性分数必须为 0–1 的有限数。单请求超时 60 秒，网络错误、429 与 5xx 最多重试一次；不会把接口错误伪装为空证据。

CLI 将阶段、向量批次、重排、聚合和复核的开始/完成及耗时写入 stderr，最终答案仍写入 stdout。
`Collected 400/400` 只是豆瓣采集进度，随后还有向量化、相关性重排、报告生成和复核。
`config.py` 中 `API_TIMEOUT_SECONDS=180`、`API_MAX_RETRIES=1` 同时控制默认聊天模型和向量接口；
这是单次请求的等待设置，整个多批流程可能更长。调用 `build_openai_chat_model` 时可通过 `timeout` / `request_timeout`、`max_retries` 显式覆盖。

CLI 与回放共用 `build_openai_chat_model`：`gpt-5.4-mini`（包括日期版本）显式使用 `reasoning_effort="medium"`，可用 `--reasoning-effort none` 关闭或自行调整；其他模型保持各自 API 默认值。原因是该 mini 模型的 [API 默认推理预算为 none](https://developers.openai.com/api/docs/models/gpt-5.4-mini)，不宜把未显式配置的两次生成当成充分语义复核。启用推理会增加输出 token 消耗和等待时间，具体质量仍需回放验证。调用者直接传入自己的 LLM 时，其参数由调用者控制。

~~~bash
conda run --no-capture-output -n Agent \
  python -m app.agent --model MODEL_NAME \
  '查找《小鬼当家4》的 IMDb 和豆瓣链接'
~~~

## Offline unittest

下列命令适用于保留本地开发资料的工作目录。测试、回放报告和采集数据不随仓库发布；新克隆的项目请按根目录 README 启动应用。

~~~bash
conda run -n Agent python -m unittest discover -s tests -v
~~~

测试使用受控 LLM 回复和 fake Tool，覆盖 schema、计划编译、Registry 权限、Patch 失效、interrupt/resume、线程隔离、失败重试，以及链接解析和评论采集的完整编排。测试不会访问 OpenAI、阿里云、Wikidata、Wikipedia、IMDb 或豆瓣，也不会写正式评论文件。

真实模型的局部回放可复用 CLI 保存的 JSON（也支持带终端前后缀的日志），避免重新采集：

~~~bash
python -m tests.replay_opinion_answer saved-output.txt \
  --model MODEL_NAME --question '原始问题' --aspect '评价方面' \
  --output reports/answer-replay.json
~~~

该命令调用阿里云重排和 OpenAI 生成/复核，只使用已保存引文附近的原文；不验证完整召回率、补检索或采集效果。阶段结果立即落盘，包含重排分数、原文来源、复核前覆盖、初稿、逐条复核及最终答案。只有 `status=completed` 表示回放完整结束。可通过 `--reuse-retrieval reports/answer-replay.json` 复用相同问题、来源和阅读范围的新版重排结果，只重跑生成与复核。旧的观点抽取结果不能复用。

回答质量诊断与回放记录保存在本地 `reports/` 中，不随源码发布。
