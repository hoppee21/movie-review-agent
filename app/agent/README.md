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
- StatePatch 是 Action 修改 State 的唯一格式；Controller 根据 State 就绪程度路由，不保存 return_to。
- 默认注册 resolve_movie 和 collect_movie_reviews；传入 RagRuntime 后，再以扩展方式注册 build/query/aggregate 三个观点 Action。
- State 已使用 movie_targets 列表，但第一版只执行单目标；多个明确电影返回 unsupported。

## 公开入口

~~~python
from langchain_openai import ChatOpenAI

from app.agent.Agent import MovieAgent
from app.rag.runtime import build_openai_rag_runtime
from config import load_openai_api_key

api_key = load_openai_api_key()
llm = ChatOpenAI(
    model="YOUR_MODEL",
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
BM25 + Dense → RRF → LLM rerank/stance/evidence span
    ↓
coverage check ─ insufficient → one Dense-only HyDE retry
    ↓
diverse evidence selection → grounded aggregation → cited answer
~~~

- 输入文件只是采集适配层；内部统一为 ReviewDocument / ReviewChunk，不依赖 IMDb 或豆瓣 JSON 的具体字段布局。
- 原始评论全文始终保留用于证据审计，只有 embed_text 按 RagConfig 截断。
- 两个平台分别召回、限额和检查覆盖，平台比较不会让数量或篇幅更大的来源淹没另一来源。
- LLM 只负责查询语义、相关性/立场标注和证据内总结；BM25、Dense、RRF、候选数与 HyDE 次数全部由代码配置。第一版不启用 SPLADE。
- HyDE 文本只用于补充 dense query，不会进入 EvidenceCard，也不会被当作真实评论。
- 索引和完整候选池只存在于当前进程的 EphemeralRagStore；LangGraph State 仅保存轻量 ID、覆盖结果和最终答案，不创建长期向量数据库。
- RAG 不创建额外索引文件；服务端会话结束后可调用 rag_runtime.clear() 立即释放内存，CLI 退出时由进程直接回收。
- stance_counts 统计的是被检索并判为直接相关的样本，不是全量平台统计；答案固定说明热门/高赞采样偏差。
- 观点任务的公开结果位于 state["final_result"].data["answer"]；索引 ID、检索 ID 和采集路径只留在 state["artifacts"] 供调试与审计。

## 配置与运行

OpenAI Key 从项目根目录 .env.local 读取；IMDb/豆瓣 Cookie 从 config/source_cookies.json 读取。源码和测试不会打印凭据。

~~~bash
conda run --no-capture-output -n Agent \
  python -m app.agent --model MODEL_NAME \
  '查找《小鬼当家4》的 IMDb 和豆瓣链接'
~~~

## Offline unittest

~~~bash
conda run -n Agent python -m unittest discover -s tests -v
~~~

测试使用受控 LLM 回复和 fake Tool，覆盖 schema、计划编译、Registry 权限、Patch 失效、interrupt/resume、线程隔离、失败重试，以及链接解析和评论采集的完整编排。测试不会访问 OpenAI、Wikidata、IMDb 或豆瓣，也不会写正式评论文件。
