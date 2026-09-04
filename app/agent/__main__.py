"""运行：python -m app.agent --model <model> '你的电影问题'。"""

import argparse
import asyncio
import json

from langchain_openai import ChatOpenAI

from app.agent.Agent import MovieAgent
from config import load_openai_api_key


async def main() -> None:
    parser = argparse.ArgumentParser(description="电影 Plan-and-Execute Agent。")
    parser.add_argument("query", help="用户的原始问题")
    parser.add_argument(
        "--model",
        required=True,
        help="当前 OpenAI 项目可调用的模型名称",
    )
    args = parser.parse_args()

    llm = ChatOpenAI(model=args.model, api_key=load_openai_api_key())
    agent = MovieAgent(llm)
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
