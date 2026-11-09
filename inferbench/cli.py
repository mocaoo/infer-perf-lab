"""A small command surface for existing experiments, without a service platform."""

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import shlex
import sys

from . import core
from .analysis import report
from .io import read, save, stamp
from .server import Server, ServerPlan


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def manifest(path):
    data = read(path)
    rows = data["rows"]
    if not rows:
        raise ValueError("Empty manifest")
    seen = set()
    for r in rows:
        if r["id"] in seen or not re.fullmatch(r"[a-zA-Z0-9_-]+", r["task"]):
            raise ValueError("Manifest needs unique IDs and safe task names")
        seen.add(r["id"])
        if r["split"] not in ("tune", "heldout"):
            raise ValueError("split must be tune or heldout")
        if not r["token_ids"] or any(type(t) is not int or t < 0 for t in r["token_ids"]):
            raise ValueError("token_ids must be a nonempty list of nonnegative integers")
        if not re.fullmatch(r"[0-9a-f]{64}", r["prompt_sha256"]):
            raise ValueError("prompt_sha256 must be a lowercase SHA256 provenance fingerprint")
        expected = hashlib.sha256(json.dumps(r["token_ids"]).encode()).hexdigest()
        if r["prompt_sha256"] != expected:
            raise ValueError("prompt_sha256 does not match the token IDs")
    # Split identity is checked on token IDs, regardless of external text hashes.
    tune = {tuple(r["token_ids"]) for r in rows if r["split"] == "tune"}
    heldout = {tuple(r["token_ids"]) for r in rows if r["split"] == "heldout"}
    if tune & heldout:
        raise ValueError("Tuning and heldout token sequences overlap")
    return rows


def parser():
    p = argparse.ArgumentParser(description="单卡 vLLM 性能实验与结果核验")
    subs = p.add_subparsers(dest="command", required=True)
    for name in ["bench", "mixed"]:
        sub = subs.add_parser(name)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--run-dir", type=Path, required=True, help="Must not already exist")
        sub.add_argument("--base-url", default="http://127.0.0.1:8023")
        sub.add_argument("--model", default="Qwen3-8B")
        sub.add_argument("--seed", type=int, default=20261103)
        sub.add_argument(
            "--server-config", type=Path, help="Start and stop an owned vLLM child (Linux/WSL)"
        )
        sub.add_argument("--concurrency", type=positive, default=1 if name == "bench" else 8)
        sub.add_argument("--output-tokens", type=positive, default=256)
        if name == "bench":
            sub.add_argument("--task", default="chat")
            sub.add_argument("--repeats", type=positive, default=3)
            sub.add_argument(
                "--requests", type=positive, help="First N heldout rows; no implicit repetition"
            )
            sub.add_argument("--natural-eos", action="store_true")
            sub.add_argument("--tag", default="baseline")
        else:
            sub.add_argument("--short-task", default="short")
            sub.add_argument("--long-task", default="long")
            sub.add_argument("--burst-delay", type=float, default=3)
    summary = subs.add_parser("report")
    summary.add_argument("run_dir", type=Path)
    plan = subs.add_parser("plan", help="Print a vLLM command without starting a server")
    plan.add_argument("config", type=Path)
    return p


async def execute(args):
    rows = manifest(args.manifest)
    core.ROOT, core.MANIFEST = args.run_dir, args.manifest
    core.BASE, core.MODEL, core.SEED = args.base_url.rstrip("/"), args.model, args.seed
    plan = ServerPlan.load(args.server_config) if args.server_config else None
    if plan:
        core.BASE, core.MODEL, core.SEED = plan.base_url, plan.served_model, plan.seed
    if args.command == "bench":
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.tag):
            raise ValueError("tag may contain only letters, digits, underscore and hyphen")
        selected = [r for r in rows if r["task"] == args.task and r["split"] == "heldout"]
        if args.requests:
            if len(selected) < args.requests:
                raise ValueError(
                    "Requested more distinct rows than available; replicate explicitly in the manifest if intended"
                )
            selected = selected[: args.requests]
        if not selected or not core.pools(args.task, "tune"):
            raise ValueError("Task needs both tune and heldout rows")
    else:
        short = [r for r in rows if r["task"] == args.short_task and r["split"] == "heldout"]
        long = [r for r in rows if r["task"] == args.long_task and r["split"] == "heldout"]
        if not short or not long or not core.pools(args.short_task, "tune") or args.burst_delay < 0:
            raise ValueError(
                "Mixed load needs short/long heldout pools, short warmup rows and nonnegative delay"
            )
    # Creating a fresh directory is the resume policy: no stale-cache ambiguity.
    args.run_dir.mkdir(parents=True, exist_ok=False)
    save(
        args.run_dir / "protocol.json",
        {
            "at": stamp(),
            "command": args.command,
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "effective_model": core.MODEL,
            "effective_base_url": core.BASE,
            "effective_seed": core.SEED,
            "python": sys.version,
            "platform": platform.platform(),
            "aiohttp": importlib.metadata.version("aiohttp"),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "server_plan": asdict(plan) if plan else None,
        },
    )
    # Freeze inputs alongside results; source manifests can change after the run.
    save(args.run_dir / "manifest.json", read(args.manifest))
    core.MANIFEST = args.run_dir / "manifest.json"
    server = Server(plan, args.run_dir / "server") if plan else None
    try:
        if server:
            # No measurement tasks exist during synchronous startup. Keeping this
            # on the owning thread makes startup errors and cleanup sequential.
            server.start()
        if args.command == "bench":
            result = await core.measure(
                args.tag,
                args.task,
                selected,
                args.concurrency,
                args.repeats,
                args.output_tokens,
                args.natural_eos,
            )
            report(args.run_dir)
        else:
            from .mixed import run

            result = await run(
                short,
                long,
                args.concurrency,
                args.burst_delay,
                args.output_tokens,
                args.run_dir / "mixed.json",
            )
        save(args.run_dir / "status.json", {"state": "complete", "at": stamp()})
        print(f"Saved: {args.run_dir.resolve()}")
        return result
    except BaseException as error:
        save(args.run_dir / "status.json", {"state": "failed", "at": stamp(), "error": repr(error)})
        raise
    finally:
        if server:
            server.stop()


def main():
    args = parser().parse_args()
    try:
        if args.command == "plan":
            print(shlex.join(ServerPlan.load(args.config).command()))
        elif args.command == "report":
            print(json.dumps(report(args.run_dir), ensure_ascii=False, indent=2))
        else:
            asyncio.run(execute(args))
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
