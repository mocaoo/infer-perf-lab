"""Exercise the complete HTTP-to-report path locally; no model or GPU is used."""

import argparse
import asyncio
import json
from pathlib import Path
import socket

from aiohttp import web

from inferbench.cli import execute, parser


async def demo(run_dir):
    generated = 0

    async def complete(request):
        nonlocal generated
        body = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        n = body["max_tokens"]
        for i in range(n):
            payload = {"choices": [{"index": 0, "text": "x", "token_ids": [100 + i]}]}
            await response.write(("data: " + json.dumps(payload) + "\n\n").encode())
            await asyncio.sleep(0.005)
        generated += n
        payload = {
            "choices": [{"index": 0, "text": "", "finish_reason": "length"}],
            "usage": {"prompt_tokens": len(body["prompt"]), "completion_tokens": n},
        }
        await response.write(("data: " + json.dumps(payload) + "\n\ndata: [DONE]\n\n").encode())
        await response.write_eof()
        return response

    async def metrics(_):
        return web.Response(text=f'vllm:generation_tokens_total{{model_name="mock"}} {generated}\n')

    app = web.Application()
    app.router.add_post("/v1/completions", complete)
    app.router.add_get("/metrics", metrics)
    runner = web.AppRunner(app)
    await runner.setup()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    site = web.SockSite(runner, sock)
    try:
        await site.start()
        args = parser().parse_args(
            [
                "bench",
                "--manifest",
                str(Path(__file__).resolve().parents[1] / "configs/prompts.example.json"),
                "--run-dir",
                str(run_dir),
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--model",
                "mock",
                "--tag",
                "mock",
                "--concurrency",
                "2",
                "--repeats",
                "2",
                "--output-tokens",
                "4",
            ]
        )
        await execute(args)
        print("Mock transport demo only. These latency numbers are not GPU inference results.")
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", type=Path, default=Path("results/mock-demo"))
    asyncio.run(demo(cli.parse_args().run_dir))
