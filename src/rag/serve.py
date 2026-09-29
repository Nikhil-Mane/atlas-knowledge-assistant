"""Run the API: `python -m rag.serve [--host 0.0.0.0] [--port 8000]`.

On Linux/Docker this is equivalent to `uvicorn rag.api.app:app`. On Windows
it is required: uvicorn creates a Proactor event loop there, which psycopg's
async driver (conversation memory in Postgres) cannot use, so this runs the
server on a selector loop instead. One process per container; scale with
replicas (see deploy/k8s).
"""
import argparse
import asyncio
import sys

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log-level", default="info")
    args = ap.parse_args()

    config = uvicorn.Config("rag.api.app:app", host=args.host, port=args.port,
                            log_level=args.log_level, proxy_headers=True, forwarded_allow_ips="*",
                            timeout_keep_alive=30)
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(server.serve())


if __name__ == "__main__":
    main()
