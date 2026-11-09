"""Build explicit vLLM commands and manage only the child we create (Linux/WSL)."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time
import urllib.request

from .io import read, save, stamp


def expand(value):
    """Expand portable configuration; fail on an unset environment variable."""
    if isinstance(value, str):
        for name in re.findall(r"\$\{([A-Za-z_][A-Za-z_0-9]*)\}", value):
            if name not in os.environ:
                raise ValueError(f"Set environment variable {name} before using this config")
        return os.path.expandvars(os.path.expanduser(value))
    if isinstance(value, list):
        return [expand(x) for x in value]
    if isinstance(value, dict):
        return {key: expand(x) for key, x in value.items()}
    return value


@dataclass
class ServerPlan:
    model: str
    tokenizer: str | None = None
    served_model: str = "Qwen3-8B"
    executable: str = "vllm"
    port: int = 8023
    dtype: str = "bfloat16"
    attention_backend: str = "FLASH_ATTN"
    max_model_len: int = 8192
    max_num_seqs: int = 64
    max_num_batched_tokens: int = 2048
    gpu_memory_utilization: float = 0.8
    kv_cache_memory_bytes: int | None = 3221225472
    quantization: str | None = None
    kv_cache_dtype: str | None = None
    speculative_config: dict | None = None
    chunked_prefill: bool = True
    seed: int = 20261103
    env: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1)")
        if min(self.max_model_len, self.max_num_seqs, self.max_num_batched_tokens) < 1:
            raise ValueError("Server capacities must be positive")
        if self.kv_cache_memory_bytes is not None and self.kv_cache_memory_bytes < 1:
            raise ValueError("KV cache bytes must be positive or null for automatic allocation")
        if self.speculative_config and self.speculative_config.get("num_speculative_tokens", 0) < 1:
            raise ValueError("num_speculative_tokens must be positive")
        if not self.chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("Without chunked prefill, token budget must cover max_model_len")

    @classmethod
    def load(cls, path):
        return cls(**expand(read(path)))

    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def command(self):
        command = [
            self.executable,
            "serve",
            self.model,
            "--tokenizer",
            self.tokenizer or self.model,
            "--served-model-name",
            self.served_model,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--max-num-seqs",
            str(self.max_num_seqs),
            "--max-num-batched-tokens",
            str(self.max_num_batched_tokens),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--no-enable-prefix-caching",
            "--attention-backend",
            self.attention_backend,
            "--seed",
            str(self.seed),
            "--enable-chunked-prefill" if self.chunked_prefill else "--no-enable-chunked-prefill",
        ]
        for flag, value in [
            ("--kv-cache-memory-bytes", self.kv_cache_memory_bytes),
            ("--quantization", self.quantization),
            ("--kv-cache-dtype", self.kv_cache_dtype),
        ]:
            if value is not None:
                command += [flag, str(value)]
        if self.speculative_config:
            command += ["--speculative-config", json.dumps(self.speculative_config)]
        return command


class Server:
    def __init__(self, plan, log_dir, startup_timeout=600):
        self.plan = plan
        self.log_dir = Path(log_dir)
        self.startup_timeout = startup_timeout
        self.proc = None
        self.handle = None

    def start(self):
        if os.name != "posix":
            raise RuntimeError(
                "Managed vLLM startup requires Linux/WSL; external HTTP mode also works on Windows"
            )
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", self.plan.port)) == 0:
                raise RuntimeError("Port occupied; refusing to touch an existing process")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        path = self.log_dir / "server.log"
        env = dict(os.environ, **self.plan.env)
        save(
            self.log_dir / "launch.json",
            {"at": stamp(), "command": self.plan.command(), "env_overrides": self.plan.env},
        )
        try:
            self.handle = path.open("w", encoding="utf-8")
            self.proc = subprocess.Popen(
                self.plan.command(),
                stdout=self.handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
            deadline = time.monotonic() + self.startup_timeout
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"vLLM exited with {self.proc.returncode}; inspect {path}")
                try:
                    with urllib.request.urlopen(
                        self.plan.base_url + "/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            return self
                except OSError:
                    pass
                time.sleep(1)
            raise TimeoutError(f"vLLM startup timed out; inspect {path}")
        except BaseException:
            self.stop()
            raise

    def stop(self):
        """Clean the owned process group even if its leader has already exited."""
        proc = self.proc
        try:
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    # Wait for workers too: the group can outlive its leader.
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        proc.poll()  # reap the leader so a zombie doesn't keep the group alive
                        try:
                            os.killpg(proc.pid, 0)
                        except ProcessLookupError:
                            break
                        time.sleep(0.1)
                    else:
                        os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
        finally:
            self.proc = None
            if self.handle:
                self.handle.close()
                self.handle = None

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
