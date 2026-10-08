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

- MovieAnalyzer 只提取电影 mentions 和候选，不理解任务。
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

## 计划与执行

Planner 的固定映射：

| Task | Actions |
|---|---|
| resolve_urls | resolve_movie |
| collect_reviews | resolve_movie → collect_movie_reviews |
| opinion_qa | resolve → collect → build → query → aggregate |
| platform_comparison | resolve → collect → build → query → aggregate |

Executor 根据 target_id 从 State 绑定工具参数。Wikidata 返回唯一结果时更新目标与平台 URL；返回多个结果时进入同一 structured clarification 流程；采集结果只在 State 中保存文件路径和数量。

未传入 RagRuntime 时，执行到观点 Action 仍会明确返回 not_implemented，便于离线测试或只启用采集能力。未预期的模型或工具异常不会伪装成业务状态。使用同一 thread_id 和原问题再次调用 run()，会从 checkpoint 的失败节点重试。

## 单电影临时 RAG

评论检索前，按已确认的 Wikidata QID 读取对应 Wikipedia 条目（优先英文，无英文条目时用中文）。
`app/rag/movie_context.py` 清理页面，保留简介、剧情、演员和制作章节，以及来源版本链接；
各章节独立限长，节选会明确标注。成功结果按 QID 缓存在进程中，不增加图节点、向量索引或 LLM 摘要调用。
这份 `movie_context` 贯穿查询改写、每批观点抽取、HyDE、聚合和复核；全文背景随 RetrievalResult 留在进程内，State 仍只保存轻量检索结果。
读取失败记录 warning，并使用现有评论继续分析；无对应条目时不猜测页面。

Prompt 的分工是：查询改写用背景扩展人物/角色称呼，重排核对指代与表演语境，
聚合联系评论细节、评价标准和角色处境，复核分别检查电影事实、评论原意与分析推断。
背景不能补造 `reason`、观众立场或平台差异，也不进入评论引用和覆盖计数。
结合背景的解释需标明推断，并注明所用维基百科来源；评论的逐字引文与理由检查保持不变。

~~~text
collected JSON
    ↓ normalize + deduplicate
Douban: one comment/one chunk
IMDb: paragraph-aware long-review chunks
    ↓
BM25 index + normalized dense matrix (in memory)
    ↓
platform-specific multilingual query rewrites
    ↓
BM25 + Dense → RRF → LLM rerank / multiple opinion units per chunk
    ↓
select final evidence → coverage gaps ─→ one targeted Dense-only HyDE retry
    ↓
group evidence by target/dimension → cited claims
    ↓
one claim audit (keep / revise / drop) → code checks → assemble cited answer
~~~

- 输入文件只是采集适配层；内部统一为 ReviewDocument / ReviewChunk，不依赖 IMDb 或豆瓣 JSON 的具体字段布局。
- 原始评论全文始终保留用于证据审计，只有 embed_text 按 RagConfig 截断。
- 两个平台分别召回、限额和检查覆盖，平台比较不会让数量或篇幅更大的来源淹没另一来源。
- LLM 只负责查询语义、相关性/立场标注和证据内总结；BM25、Dense、RRF、候选数与 HyDE 次数全部由代码配置。第一版不启用 SPLADE。
- HyDE 文本只用于补充 dense query，不会进入 EvidenceCard，也不会被当作真实评论。
- 索引和完整候选池只存在于当前进程的 EphemeralRagStore；LangGraph State 仅保存轻量 ID、覆盖结果和最终答案，不创建长期向量数据库。
- RAG 不创建额外索引文件；服务端会话结束后可调用 rag_runtime.clear() 立即释放内存，CLI 退出时由进程直接回收。
- 候选预算按 chunk 计算，同一长评的多个命中片段可以进入重排；不再在理解内容之前按 review_id 只留一个片段。
- 重排在一次调用中提取 `OpinionUnit`：`target / subaspect / opinion / stance / reason / evidence_span`。每条观点只绑定一个对象和维度，一个片段最多六条；不同演员的相反评价分别保留，条件式混合评价保留其转折。后续批次和补检索接收已出现的对象名称，优先沿用相同对象的名称。
- 引文必须逐字出现在来源片段和全文中；`reason` 必须是引文中明示的理由或条件，否则清空。泛泛褒贬可作为观点，但不能满足理由覆盖。
- `evidence.py` 统一负责观点去重、最终证据选择和覆盖检查。在维度和最低独立评论要求之后，优先保留实际存在的不同立场，再兼顾可比对象与解释性证据，避免更多演员的正评挤掉已有负评。重叠片段中的同一观点不重复保留，同一评论的不同观点可以共存。
- `stance_counts` 只统计最终写作证据涉及的独立评论。同一评论有相反立场时归入 mixed，每篇评论只计一次；这些数量不代表全部采集评论或平台用户分布。
- `coverage.gaps` 是补检索直接使用的缺口列表，每项包含平台、对象、维度和类型：`reviews` 独立评论不足、`subaspect` 缺直接观点、`reason` 缺原文明示的理由、`comparison` 缺同一对象/维度的跨平台对应证据。选择、覆盖和聚合共用 `opinion_scope`，忽略姓名大小写、空格及常见标点差异；中英文别名仍由重排统一，模糊指代不强行合并。
- 覆盖在最终证据选择之后计算。HyDE 段落绑定缺口编号，由代码确定目标平台；第二轮只召回并重排未处理片段，保留首轮观点池。数据耗尽或两轮预算用完后保留未解决缺口，不为凑齐反例生成观点。
- 删除了重排后再截取 top-N 的中间预算。只有每轮候选 chunk 预算和最终观点预算；聚合直接读取这批已检查的证据，不再次裁剪。
- 聚合按同一对象和维度组织观点、评论 ID 和完整观点引文，明确标出可比组，生成一组带引用的 `AnswerClaim`：直接观点 `finding`、解释性判断 `insight`、平台对照 `comparison`。平台内有根据的条件和分歧也可以形成 insight；可比组只约束跨平台判断。
- 草稿绑定已提取观点的引文；复核再读取原始 chunk 上下文，帮助理解指代、讽刺和转折，不能借同一 chunk 里另一演员或另一观点的细节来支持当前引用。结论从通过复核的判断组装，不单独生成无引用总论。
- 固定一次生成、一次逐条复核。复核为每个判断返回且只返回一次 `keep / revise / drop` 和理由；修订只能使用本条原有引用，不能加入新论点。解释不足的 insight 可降为 finding，没有支持的判断整体删除。
- 代码进一步检查支持/反证引用存在且不重叠，反证与支持证据的对象和维度对应；insight 必须有解释、限制和原文明示理由。finding 只支持单个平台内的观点；任何跨平台判断引用的每个对象/维度都须有双方证据，换一个 kind 不能绕过检查。具体引用错误会交给复核模型，模型说 keep 不能绕过这些条件。语义上的支持关系仍由模型核验，这些检查不保证消除全部误判。
- 复核也可以剔除被错误归类的观点证据；若观点相关，但所谓 reason 只是褒贬重复，可通过 `unsupported_reason_ids` 单独清空理由。随后按原 `coverage.minimum_reviews` 重算覆盖与立场计数，缺口继续保留在答案中。这一步不启动额外检索循环。
- `insights` 包含判断、解释、支持证据、反证和限制，证据薄弱时允许为空。结论和平台对照分别通过 `conclusion_evidence_ids`、`platform_comparison_evidence_ids` 提供引用，普通观点组也保留 `counterevidence_ids`。
- 没有比较通过复核时，答案明确说明尚无可核验的平台对照；如双方仍有直接观点，结论各保留一条，避免被单个平台的多个观点占满。
- 重排使用批内短编号映射原始 chunk ID，必须完整返回每个候选（无直接观点则 opinions=[]）；漏项、重复或未知编号会报错。
- 最终 EvidenceCard 包含 `chunk_id` 和观点字段；每个观点有独立 `evidence_id`，并通过 `citation_id` 保留本次 e1/e2 短编号。剔除证据不重排编号，正文中出现的短编号仍能找到对应卡片。旧的重排结果缺少观点字段，需要重新提取；回放脚本仍可读取旧版答案的原始引文。
- 聚合使用本次短编号 e1/e2 与复核编号 c1/c2，输出时由代码还原为来源观点 ID。复核漏项、重复项、未知 c 编号或未知被剔除证据编号会报错；未通过引用和支持条件检查的判断不会写入答案。
- 观点任务的公开结果位于 state["final_result"].data["answer"]；索引 ID、检索 ID 和采集路径只留在 state["artifacts"] 供调试与审计。

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

该命令使用 `.env.local` 调用模型，只重跑旧答案已引用 chunk 的查询规划、重排与聚合；不验证完整召回率、HyDE 或采集效果。每个已完成阶段立即落盘，包含查询计划、原始重排、观点短编号映射、复核前覆盖、初稿、逐条复核及最终答案。只有 `status=completed` 表示回放完整结束。可通过 `--reuse-retrieval reports/answer-replay.json` 复用相同问题的新版观点提取结果，只重跑生成与复核。

回答质量诊断与回放记录保存在本地 `reports/` 中，不随源码发布。
