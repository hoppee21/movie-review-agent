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
- [架构 FigJam](https://www.figma.com/board/VR3oajHfyJKe5Z5pjLRZHB) 已同步当前 RAG 流程；图 4–5 补充 LLM 调用 `ask_user` UI 工具的待实现设计。
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
这份 `movie_context` 贯穿查询改写、每批观点抽取、定向补检索、聚合和复核；全文背景随 RetrievalResult 留在进程内，State 仍只保存轻量检索结果。
读取失败记录 warning，并使用现有评论继续分析；无对应条目时不猜测页面。

Prompt 的分工是：查询改写用背景扩展人物/角色称呼，重排核对指代与表演语境，
聚合联系评论细节、评价标准和角色处境，复核分别检查电影事实、评论原意与分析推断。
背景不能补造 `reason`、观众立场或平台差异，也不进入评论引用和覆盖计数。
结合背景的解释需标明推断，并注明所用维基百科来源；评论的逐字引文与理由检查保持不变。

~~~text
collected JSON → normalize / deduplicate → paragraph / sentence passages + source offsets
    ↓
BM25 + normalized dense matrix (in memory)
    ↑
question → neutral queries in each platform's language
    ↓
BM25 + Dense → RRF → group hits by review → bounded source reading windows
    ↓
extract observed opinions / reasons / quotes → select writing evidence
    ↓
optional question-specific queries → at most one hybrid follow-up on unread text
    ↓
generate cited claims → audit with the same reading context → assemble answer
~~~

- 两个平台共用按实际长度决定的切分规则。默认片段最多 1800 字符，优先段落、其次句末，长度兜底；重叠最多约 200 字符，优先完整句子。完整评论不变，`start/end` 是原文中的半开区间。
- Embedding 仍使用标题与片段（最多 2000 字符）。BM25、精确向量搜索和 RRF 保留；索引只存在于本次运行时的内存中。
- 查询计划只有 `queries[{platform,text}]`，每平台通常一条、最多两条。删除 `purpose` 和计划中的 `subaspects`；用户明确指定的对象、方面、关系与限制始终传给后续模型。
- 每个检索通道最多接纳 80 条独立评论的命中，融合后每平台最多阅读 80 条评论。多次命中不累加为一条评论的额外票数，按其最高片段融合分排序。
- 同条评论的命中片段优先合并，再用剩余预算补充相邻句子。每条评论的阅读范围合计最多 3600 字符；短评直接读全文，长评的远处命中保持分开的范围，不伪造成连续原文。
- 抽取模型收到标题和 `contexts`；每条候选最多提取六个 `OpinionUnit`。`subaspect` 是从实际观点归纳的标签，既可复用已有同义标签，也可新增。后续批次仅接收已经通过原文校验的对象和方面。
- 每条引文必须位于一个连续阅读范围内，同时存在于完整原文；`reason` 必须原样出现在引文中。不能因为提供了更多上下文，就借用另一对象或另一观点的细节。
- 最终每平台最多 12 个观点，优先保证独立来源，保留实际发现的不同立场及带理由的证据；没有分类填空或强制正反配额。重叠片段中的重复观点去重，同一评论的不同观点可以共存。
- `coverage` 只报告独立评论数量与立场计数，`sufficient` 表示达到配置中的最低来源数（默认 6），不承诺语义上回答完整。不同观点属于同一原评论时只计一次，冲突立场记为 mixed；这些计数不能代表平台总体分布。
- 首轮后，若仍有未阅读文本且有预算，复用查询生成器阅读已有引文与理由，判断用户问题的哪些具体要求仍缺证据；足够则返回空查询。不会因为未达到数量阈值、缺少模型预设维度或偶然出现的新演员而强制补检索。
- 补检索使用普通查询和同样的 BM25 + Dense + RRF，最多一次，不再生成 HyDE 假设评论。排除已完整阅读的片段，同一长评未读部分仍可检索；合并两轮观点后重新选择最终证据。
- 生成和复核读取相同的原文上下文、引文和电影背景。实际发现的方面用来组织回答，不作为白名单。`both_platforms` 只说明该对象有双方来源，不意味着所有观点可比。
- 代码校验引用存在、支持/反证不重叠、跨平台引用具有双方同对象证据，以及 insight 具有理由、解释和限制。评价含义是否对应、反证是否相关由已有复核调用根据原文判定，不用自由生成的方面标签做严格字符串配对。
- 固定一次生成和一次复核，判断采用 `keep/revise/drop`；修订只能使用原判断已有引用，可收窄为单平台 finding。复核后重算来源计数，不启动新一轮检索。
- EvidenceCard 保留完整原文、逐字引文、阅读范围和稳定的引用编号；旧报告没有阅读范围时仍可展示。LangGraph State 只存索引/检索 ID 和最终产物，完整运行时数据在任务结束时释放。

## 配置与运行

OpenAI Key 从项目根目录 .env.local 读取；IMDb/豆瓣 Cookie 从 config/source_cookies.json 读取。源码和测试不会打印凭据。

CLI 将阶段、向量批次、观点抽取批次、聚合和复核的开始/完成及耗时写入 stderr，最终答案仍写入 stdout。
`Collected 400/400` 只是豆瓣采集进度，随后还有向量化、分批模型分析和复核。
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

测试使用受控 LLM 回复和 fake Tool，覆盖 schema、计划编译、Registry 权限、Patch 失效、interrupt/resume、线程隔离、失败重试，以及链接解析和评论采集的完整编排。测试不会访问 OpenAI、Wikidata、Wikipedia、IMDb 或豆瓣，也不会写正式评论文件。

真实模型的局部回放可复用 CLI 保存的 JSON（也支持带终端前后缀的日志），避免重新采集：

~~~bash
python -m tests.replay_opinion_answer saved-output.txt \
  --model MODEL_NAME --question '原始问题' --aspect '评价方面' \
  --output reports/answer-replay.json
~~~

该命令使用 `.env.local` 调用模型，只在已保存引文附近的原文范围内重跑查询改写、抽取与聚合；不验证完整召回率、补检索或采集效果。每个已完成阶段立即落盘，包含查询计划、原始重排、观点短编号映射、复核前覆盖、初稿、逐条复核及最终答案。只有 `status=completed` 表示回放完整结束。可通过 `--reuse-retrieval reports/answer-replay.json` 复用相同问题的新版观点提取结果，只重跑生成与复核。

回答质量诊断与回放记录保存在本地 `reports/` 中，不随源码发布。
