import asyncio
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import aiohttp
from aiohttp import web

from inferbench import core
from inferbench.analysis import percentile, summarize
from inferbench.cli import execute, manifest, parser
from inferbench.io import read, save
from inferbench.metrics import counter, counter_delta, parse_metrics
from inferbench.quality import isolate_code, math_correct
from inferbench.server import Server, ServerPlan

FIXTURE = Path(__file__).resolve().parents[1] / "configs/prompts.example.json"


class MetricsTests(unittest.TestCase):
    def test_labels_timestamps_and_exact_names(self):
        values = parse_metrics("""# comment
vllm:generation_tokens_total{model_name="a b"} 10 12345
vllm:generation_tokens_total{model_name="c"} 4
vllm:generation_tokens_total_extra 999
vllm:broken NaN
""")
        self.assertEqual(counter(values, "generation_tokens_total"), 14)
        self.assertNotIn("vllm:broken", values)

    def test_reset_or_disappearing_series_rejected(self):
        for after in [{"vllm:generation_tokens_total": 2}, {}]:
            with self.assertRaises(RuntimeError):
                counter_delta({"vllm:generation_tokens_total": 10}, after)

    def test_percentile_and_quality_semantics(self):
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.99), 3.97)
        result = math_correct("The answer is 42", "#### 42")
        self.assertFalse(result["correct"])
        self.assertTrue(result["fallback_correct"])
        with patch.dict(os.environ, {"INFERBENCH_BWRAP": "/missing/sandbox"}):
            self.assertEqual(
                isolate_code("def f(): return 1", "", "f")["status"], "sandbox_unavailable"
            )

    def test_split_overlap_rejected_even_if_hashes_differ(self):
        data = read(FIXTURE)
        data["rows"][1]["token_ids"] = data["rows"][0]["token_ids"]
        data["rows"][1]["prompt_sha256"] = data["rows"][0]["prompt_sha256"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            save(path, data)
            with self.assertRaisesRegex(ValueError, "overlap"):
                manifest(path)

    def test_invalid_case_cannot_be_reported(self):
        with self.assertRaises(ValueError):
            summarize({"counter_boundary_valid": False})

    def test_atomic_save_does_not_replace_valid_json_with_nan(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "result.json"
            save(path, {"ok": True})
            with self.assertRaises(ValueError):
                save(path, {"bad": float("nan")})
            self.assertEqual(read(path), {"ok": True})
            self.assertEqual(len(list(Path(folder).iterdir())), 1)


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old = (core.ROOT, core.MANIFEST, core.BASE, core.MODEL, core.SEED)
        self.generated = 0
        self.active = 0
        self.peak = 0
        self.mode = "normal"
        self.calls = []
        app = web.Application()
        app.router.add_post("/v1/completions", self.complete)
        app.router.add_get("/metrics", self.metrics)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        self.site = web.SockSite(self.runner, sock)
        await self.site.start()
        core.ROOT = Path(self.temp.name)
        core.MANIFEST = FIXTURE
        core.BASE = f"http://127.0.0.1:{self.port}"
        self.prompt = read(FIXTURE)["rows"][1]

    async def asyncTearDown(self):
        await self.runner.cleanup()
        core.ROOT, core.MANIFEST, core.BASE, core.MODEL, core.SEED = self.old
        self.temp.cleanup()

    async def metrics(self, _):
        return web.Response(
            text=f'vllm:generation_tokens_total{{model_name="test model"}} {self.generated}\n'
        )

    async def complete(self, request):
        body = await request.json()
        self.calls.append(body)
        if self.mode == "error":
            return web.Response(status=503, text="test failure")
        self.active += 1
        self.peak = max(self.peak, self.active)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        try:
            n = body["max_tokens"]
            ids = list(range(100, 100 + n))
            chunks = [
                {"choices": [{"index": 0, "text": "你", "token_ids": ids[:1]}]},
                {
                    "choices": [
                        {"index": 0, "text": "好", "token_ids": ids[1:], "finish_reason": "length"}
                    ]
                },
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": len(body["prompt"]),
                        "completion_tokens": n + (self.mode == "bad_usage"),
                    },
                },
            ]
            for obj in chunks:
                # Split UTF-8 text and SSE framing across transport writes.
                payload = ("data: " + json.dumps(obj, ensure_ascii=False) + "\r\n\r\n").encode()
                for offset in range(0, len(payload), 7):
                    await response.write(payload[offset : offset + 7])
                    await asyncio.sleep(0)
                await asyncio.sleep(0.01)
            if self.mode != "truncated":
                await response.write(b"data: [DONE]\n\n")
            self.generated += n
            await response.write_eof()
        finally:
            self.active -= 1
        return response

    async def test_fragmented_sse_and_multi_token_chunk(self):
        async with aiohttp.ClientSession() as session:
            result = await core.request(session, self.prompt, output=4)
        self.assertEqual(result["text"], "你好")
        self.assertEqual(result["token_ids"], [100, 101, 102, 103])
        self.assertEqual(result["chunk_count"], 2)
        self.assertAlmostEqual(result["tpot_ms"], result["chunk_gaps_ms"][0] / 3)
        self.assertGreaterEqual(result["e2e_ms"], result["ttft_ms"])

    async def test_usage_and_truncated_stream_rejected(self):
        async with aiohttp.ClientSession() as session:
            for mode in ["bad_usage", "truncated", "error"]:
                self.mode = mode
                with self.assertRaises(RuntimeError):
                    await core.request(session, self.prompt, output=4)

    async def test_bounded_concurrency(self):
        async with aiohttp.ClientSession() as session:
            rows, _ = await core.batch(session, [self.prompt] * 5, 2, 4)
        self.assertEqual(len(rows), 5)
        self.assertEqual(self.peak, 2)

    async def test_batch_failure_cancels_sibling_requests(self):
        cancelled = asyncio.Event()

        async def fake_request(session, prompt, *args):
            if prompt["id"] == "bad":
                await asyncio.sleep(0.01)
                raise RuntimeError("failed")
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()

        with patch.object(core, "request", side_effect=fake_request):
            with self.assertRaises(RuntimeError):
                await core.batch(None, [{"id": "slow"}, {"id": "bad"}], 2, 4)
        self.assertTrue(cancelled.is_set())

    async def test_complete_cli_case_and_refuse_overwrite(self):
        run_dir = Path(self.temp.name) / "run"
        args = parser().parse_args(
            [
                "bench",
                "--manifest",
                str(FIXTURE),
                "--run-dir",
                str(run_dir),
                "--base-url",
                core.BASE,
                "--output-tokens",
                "4",
                "--repeats",
                "2",
                "--concurrency",
                "2",
            ]
        )
        result = await execute(args)
        self.assertTrue(result["counter_boundary_valid"])
        self.assertEqual(result["expected_generation_tokens"], 16)
        self.assertEqual(result["observed_generation_tokens"], 16)
        self.assertEqual(self.calls[0]["prompt"], [1, 2])  # tune-only warmup
        self.assertEqual(self.calls[0]["max_tokens"], 4)
        self.assertTrue((run_dir / "report.md").exists())
        self.assertEqual(read(run_dir / "status.json")["state"], "complete")
        with self.assertRaises(FileExistsError):
            await execute(args)

    async def test_mixed_burst_bypasses_short_semaphore(self):
        from inferbench.mixed import run

        rows = read(FIXTURE)["rows"]
        result = await run(
            [rows[4]] * 3, [rows[-1]], 1, 0.005, 4, Path(self.temp.name) / "mixed.json"
        )
        # Warmup then short+long overlap, although short concurrency is one.
        self.assertEqual(self.peak, 2)
        self.assertEqual(result["long_requests"], 1)
        burst = result["actual_burst_sent_seconds"][0]
        self.assertLess(
            burst, max(r["sent_seconds"] for r in result["requests"] if r["group"] == "short")
        )


class PlanTests(unittest.TestCase):
    def test_config_expansion_and_fixed_capacity(self):
        with patch.dict(os.environ, {"INFERBENCH_MODELS": "/models"}):
            plan = ServerPlan.load(FIXTURE.parent / "dflash.fixed.json")
            self.assertIn("/models/Qwen3-8B", plan.command())
            self.assertIn("--kv-cache-memory-bytes", plan.command())
            self.assertEqual(plan.speculative_config["num_speculative_tokens"], 16)
            capacity = ServerPlan.load(FIXTURE.parent / "fp8-kv.capacity.json")
            self.assertNotIn("--kv-cache-memory-bytes", capacity.command())

    @unittest.skipUnless(os.name == "posix", "Managed server is Linux/WSL only")
    def test_start_failure_cleans_child_and_handle(self):
        plan = ServerPlan(model="unused")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            plan.port = sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as folder:
            server = Server(plan, folder, startup_timeout=0.1)
            with patch.object(
                plan, "command", return_value=[sys.executable, "-c", "import time; time.sleep(30)"]
            ):
                with self.assertRaises(TimeoutError):
                    server.start()
            self.assertIsNone(server.proc)
            self.assertIsNone(server.handle)

    @unittest.skipUnless(os.name == "posix", "Managed server is Linux/WSL only")
    def test_occupied_port_is_not_modified(self):
        with socket.socket() as sock, tempfile.TemporaryDirectory() as folder:
            sock.bind(("127.0.0.1", 0))
            sock.listen()
            server = Server(ServerPlan(model="unused", port=sock.getsockname()[1]), folder)
            with self.assertRaisesRegex(RuntimeError, "Port occupied"):
                server.start()
            self.assertEqual(sock.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN), 1)


if __name__ == "__main__":
    unittest.main()
