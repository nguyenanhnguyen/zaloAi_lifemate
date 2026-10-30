"""Qwen3.5 Q8 tool calling through a private, locally managed llama.cpp server."""
from __future__ import annotations

import atexit
import json
import logging
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASR_REPO = "Qualcomm-AI-Research/PhoASR-whisper-small"
ASR_REVISION = "da2d9ae0ac0a6a6faba20add61c2e980ccd85037"
LLM_REPO = "unsloth/Qwen3.5-4B-GGUF"
LLM_REVISION = "e87f176479d0855a907a41277aca2f8ee7a09523"
LLM_FILE = "Qwen3.5-4B-Q8_0.gguf"


def model_path(repo: str, *, filename: str | None = None, download: bool = False) -> str:
    """Reuse C/D caches; reserve 10 GiB plus staging space before a new download."""
    from huggingface_hub import hf_hub_download, snapshot_download
    revision = {ASR_REPO: ASR_REVISION, LLM_REPO: LLM_REVISION,
                "vinai/PhoWhisper-large": "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"}.get(repo, "main")
    default = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    roots = list(dict.fromkeys([default, Path.home() / ".cache/huggingface/hub",
                               PROJECT_ROOT / "models/hf-cache/hub"]))

    def fetch(root: Path, local: bool) -> str:
        kwargs = dict(repo_id=repo, revision=revision, cache_dir=str(root), local_files_only=local)
        if filename:
            return hf_hub_download(filename=filename, **kwargs)
        path = snapshot_download(allow_patterns=["*.json", "*.txt", "*.safetensors", "*.model", "pytorch_model.bin"],
                                 max_workers=2, **kwargs)
        if not any(Path(path).glob("*.safetensors")) and not (Path(path) / "pytorch_model.bin").is_file():
            raise FileNotFoundError(f"Weights missing from {path}")
        return path

    for root in roots:
        try:
            return fetch(root, True)
        except (OSError, ValueError):
            pass
    if not download:
        raise FileNotFoundError(f"Missing {repo}. Run scripts/infer.py --prepare-only first.")
    expected = 4.2 if filename else (5.76 if repo == "vinai/PhoWhisper-large" else 1.6)
    root = default
    probe = root
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < (10 + 2 * expected + 2) * 1024**3:
        root = PROJECT_ROOT / "models/hf-cache/hub"
        if shutil.disk_usage(PROJECT_ROOT).free < (2 * expected + 2) * 1024**3:
            raise OSError("Insufficient space in default cache and project drive")
    logger.info("Downloading %s to %s", repo, root)
    return fetch(root, False)


def server_path() -> Path:
    if os.environ.get("LLAMA_SERVER"):
        return Path(os.environ["LLAMA_SERVER"])
    name = "llama-server.exe" if os.name == "nt" else "llama-server"
    matches = list((PROJECT_ROOT / "models/llama.cpp").rglob(name))
    return matches[0] if matches else Path(shutil.which(name) or PROJECT_ROOT / "models/llama.cpp" / name)


class LLMAgent:
    """Own one local server and reuse prompt cache between tool turns."""

    def __init__(self, model_name: str | None = None, *, context_size: int = 12288,
                 max_new_tokens: int = 1024, thinking_tokens: int = 512,
                 threads: int = 8, gpu_layers: int | None = None) -> None:
        self.process = None
        self.log_handle = None
        self.max_new_tokens = max_new_tokens
        self.thinking_tokens = thinking_tokens
        self.last_usage = {}
        model = Path(model_name) if model_name else Path(model_path(LLM_REPO, filename=LLM_FILE))
        binary = server_path()
        if not binary.is_file() or not model.is_file():
            raise FileNotFoundError("Missing GGUF or llama-server. See README installation instructions.")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        log_path = PROJECT_ROOT / "models/llama.cpp/server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = log_path.open("w", encoding="utf-8")
        command = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
                   "--ctx-size", str(context_size), "--parallel", "1", "--jinja",
                   "--threads", str(threads), "--threads-batch", str(threads),
                   "--batch-size", "512", "--ubatch-size", "128", "--flash-attn", "on",
                   "--fit", "on", "--fit-target", "384", "--cache-ram", "512",
                   "--reasoning-budget", str(thinking_tokens), "--reasoning-format", "deepseek",
                   "--no-context-shift", "--no-webui"]
        if gpu_layers is not None:
            command += ["--n-gpu-layers", str(gpu_layers)]
        logger.info("Loading Qwen3.5 Q8; fit GPU/CPU placement to free VRAM. Log: %s", log_path)
        try:
            self.process = subprocess.Popen(command, stdout=self.log_handle, stderr=subprocess.STDOUT,
                                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            atexit.register(self.close)
            started = time.perf_counter()
            next_log = started + 15
            while time.perf_counter() - started < 300:
                if self.process.poll() is not None:
                    raise RuntimeError(f"llama-server exited ({self.process.returncode}); see {log_path}")
                try:
                    with self.http.open(self.base_url + "/health", timeout=2) as response:
                        if response.status == 200:
                            logger.info("LLM ready after %.1f s", time.perf_counter() - started)
                            return
                except (OSError, urllib.error.URLError):
                    pass
                if time.perf_counter() >= next_log:
                    logger.info("Still loading/warming up LLM (%.0f s)", time.perf_counter() - started)
                    next_log += 15
                time.sleep(0.25)
            raise TimeoutError(f"llama-server startup exceeded 300 s; see {log_path}")
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log_handle is not None:
            self.log_handle.close()

    def generate(self, messages: list[dict], *, tools: list[dict], deadline: float) -> dict:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Sample time budget exhausted before LLM")
        payload = {"messages": messages, "tools": tools, "tool_choice": "auto",
                   "parallel_tool_calls": False, "max_tokens": self.max_new_tokens,
                   "temperature": 0, "seed": 42, "cache_prompt": True,
                   "chat_template_kwargs": {"enable_thinking": self.thinking_tokens > 0}}
        request = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        try:
            with self.http.open(request, timeout=remaining) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(exc.read().decode("utf-8", errors="replace")[:1000]) from exc
        self.last_usage = result.get("usage", {})
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise ValueError("LLM output hit token/context limit; refusing a truncated action")
        return choice["message"]
