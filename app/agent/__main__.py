"""运行：python -m app.agent --model <model> '你的电影问题'。"""

import argparse
import asyncio
import json
import logging

from app.agent.Agent import MovieAgent
from app.rag.runtime import build_openai_chat_model, build_openai_rag_runtime
from config import API_MAX_RETRIES, API_TIMEOUT_SECONDS, load_openai_api_key


async def main() -> None:
    parser = argparse.ArgumentParser(description="电影 Plan-and-Execute Agent。")
    parser.add_argument("query", help="用户的原始问题")
    parser.add_argument(
        "--model",
        required=True,
        help="当前 OpenAI 项目可调用的模型名称",
    )
    parser.add_argument(
        "--reasoning-effort", choices=["none", "low", "medium", "high", "xhigh"],
        help="推理预算；gpt-5.4-mini 默认 medium，其他模型保留 API 默认值。",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger("app").setLevel(logging.INFO)
    logging.getLogger("app").info(
        "开始处理问题；模型 %s，单次 API 超时 %.0f 秒，最多重试 %d 次",
        args.model, API_TIMEOUT_SECONDS, API_MAX_RETRIES,
    )
    api_key = load_openai_api_key()
    llm = build_openai_chat_model(args.model, api_key=api_key, reasoning_effort=args.reasoning_effort)
    agent = MovieAgent(
        llm,
        rag_runtime=build_openai_rag_runtime(llm, api_key=api_key),
    )
    state = await agent.run(args.query, thread_id="cli")

    while pending := state.get("__interrupt__"):
        print(json.dumps(pending[0].value, ensure_ascii=False, indent=2))
        reply = input("补充信息：")
        while not reply.strip():
            reply = input("请输入补充信息：")
        state = await agent.resume(reply, thread_id="cli")

    print(json.dumps(
        state["final_result"].model_dump(mode="json"),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    asyncio.run(main())
