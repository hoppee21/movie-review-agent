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
- 真实注册 resolve_movie 和 collect_movie_reviews。观点 RAG Action 已进入计划契约但尚未注册，运行到该步骤会明确返回 not_implemented。
- State 已使用 movie_targets 列表，但第一版只执行单目标；多个明确电影返回 unsupported。

## 公开入口

~~~python
from langchain_openai import ChatOpenAI

from app.agent.Agent import MovieAgent
from config import load_openai_api_key

llm = ChatOpenAI(
    model="YOUR_MODEL",
    api_key=load_openai_api_key(),
)
agent = MovieAgent(llm)
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

未预期的模型或工具异常不会伪装成业务状态。使用同一 thread_id 和原问题再次调用 run()，会从 checkpoint 的失败节点重试。

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
