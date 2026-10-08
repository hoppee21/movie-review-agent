"""Run the local API: python -m app.web --port 8000."""

import argparse
import logging
import os

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Movie Evidence Agent Web 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=os.getenv("AGENT_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    os.environ["AGENT_MODEL"] = args.model
    logging.basicConfig(level=logging.WARNING, format="[%(asctime)s] %(message)s")
    logging.getLogger("app").setLevel(logging.INFO)
    uvicorn.run("app.web.server:app", host=args.host, port=args.port, reload=args.reload, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
