"""Qwen3-4B-Instruct-2507 tool calling via transformers; GPU-first, KV cache reused per sample.

Replaces the earlier llama.cpp/GGUF backend: that path ran the LLM on CPU (787 MB VRAM only,
50-470 s/sample locally — see EXPERIMENTS.md) and could never fit the competition's 30 s/sample
budget. Running the full-precision model directly on GPU (as on the organisers' A30) is both
simpler (no server subprocess) and fast enough.
"""
from __future__ import annotations

import copy
import json
import logging
import re
import time
import unicodedata

from .models import model_path

logger = logging.getLogger(__name__)
LLM_REPO = "Qwen/Qwen3-4B-Instruct-2507"
LLM_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"


def _compact_schema(value):
    """Drop prose from a JSON Schema; required/enum/x-ref/x-one-of stay intact."""
    if isinstance(value, dict):
        return {key: _compact_schema(val) for key, val in value.items()
                if key not in {"description", "title", "examples", "default"}}
    if isinstance(value, list):
        return [_compact_schema(item) for item in value]
    return value


def _keywords(value: str) -> set[str]:
    value = unicodedata.normalize("NFD", str(value).lower())
    value = "".join(c for c in value if unicodedata.category(c) != "Mn").replace("đ", "d")
    return set(re.findall(r"[a-z0-9]+", value))


class InvalidDecision(ValueError):
    """The model's output could not be parsed into DONE or a well-formed tool call."""


def _parse_decision(text: str) -> dict:
    """Parse <tool_call>{...}</tool_call> blocks; DONE or plain text means no tool calls."""
    text = text.strip()
    if not text or text.upper().rstrip(".") == "DONE":
        return {"tool_calls": [], "content": "DONE"}
    calls = []
    for index, match in enumerate(re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S)):
        try:
            value = json.loads(match.group(1))
        except ValueError as exc:
            raise InvalidDecision(f"Malformed tool call: {match.group(1)[:160]}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise InvalidDecision("Tool call missing function name")
        args = value.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError as exc:
                raise InvalidDecision("Invalid serialized arguments") from exc
        calls.append({"id": f"call_{index}", "type": "function",
                      "function": {"name": value["name"], "arguments": json.dumps(args, ensure_ascii=False)}})
    if calls:
        return {"tool_calls": calls, "content": ""}
    if "<tool_call>" in text:
        raise InvalidDecision("Unclosed tool call")
    return {"tool_calls": [], "content": text}


class LLMAgent:
    """Own one Qwen3 model on GPU; reuse the KV-cache prefix across tool turns of one sample."""

    def __init__(self, model_name: str | None = None, *, context_size: int = 8192,
                 max_new_tokens: int = 384, vram_4bit_threshold_gib: float = 10.0) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.context_size = context_size
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = model_name or model_path(LLM_REPO, LLM_REVISION)
        # The A30 (24 GiB) runs full bf16; a dev GPU too small for the 4B model in bf16
        # (<10 GiB total, e.g. a 4 GB laptop card) auto-falls to 4-bit NF4 instead of OOM-ing
        # or silently spilling into slow/unstable shared GPU memory under load.
        quantization_config = None
        if self.device == "cuda":
            total_vram_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if total_vram_gib < vram_4bit_threshold_gib:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
                )
                logger.warning("GPU has %.1f GiB VRAM (<%.0f GiB): loading LLM in 4-bit NF4. "
                               "The frozen Docker solution for the A30 must use full bf16.",
                               total_vram_gib, vram_4bit_threshold_gib)
        logger.info("Loading LLM %s on %s (dtype=%s, 4bit=%s)", model, self.device, self.dtype,
                    quantization_config is not None)
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model, dtype=self.dtype, low_cpu_mem_usage=True, local_files_only=True,
            attn_implementation="sdpa", quantization_config=quantization_config,
            device_map={"": 0} if quantization_config is not None else None,
        )
        if quantization_config is None:
            self.llm = self.llm.to(self.device)
        self.llm.eval()
        self.cache = None
        self.cached_ids: list[int] = []

    def _prompt_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        encoded = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        return encoded["input_ids"].shape[1]

    def fit_tools(self, messages: list[dict], tools: list[dict], request_text: str = "") -> list[dict]:
        """Bound the rendered prompt before it reaches the GPU (see EXPERIMENTS.md: a 47-tool
        catalog alone renders to ~19 KB / several thousand tokens uncompacted, which both wastes
        the 30 s/sample budget and can OOM a small GPU during attention). Tool schemas kept are
        always complete; only prose and, as a last resort, less relevant tools are dropped.
        """
        limit = self.context_size - self.max_new_tokens
        if self._prompt_tokens(messages, tools) <= limit:
            return tools
        compact = copy.deepcopy(tools)
        for tool in compact:
            tool["function"]["parameters"] = _compact_schema(tool["function"].get("parameters", {}))
            tool["function"]["description"] = tool["function"].get("description", "")[:100]
        if self._prompt_tokens(messages, compact) <= limit:
            logger.info("Tool prompt compacted to fit %d tokens", limit)
            return compact
        query = _keywords(request_text)
        ranked = sorted(range(len(compact)), key=lambda i: (
            compact[i]["function"]["name"] == "sys.ask_missing_info",
            len(_keywords(compact[i]["function"]["name"].replace("_", " ") + " "
                          + compact[i]["function"]["description"]) & query),
        ), reverse=True)
        mandatory = {i for i, tool in enumerate(compact) if tool["function"]["name"] == "sys.ask_missing_info"}

        def choose(count: int) -> list[dict]:
            keep = set(ranked[:count]) | mandatory
            return [tool for i, tool in enumerate(compact) if i in keep]

        low, high = 0, len(compact)
        while low < high:
            middle = (low + high + 1) // 2
            if self._prompt_tokens(messages, choose(middle)) <= limit:
                low = middle
            else:
                high = middle - 1
        selected = choose(low)
        if not selected or self._prompt_tokens(messages, selected) > limit:
            raise ValueError(f"Cannot fit tool schemas within {limit} tokens even after routing")
        logger.warning("Tool prompt routed to %d/%d tools to fit %d tokens",
                       len(selected), len(tools), limit)
        return selected

    def reset(self) -> None:
        """Call once per sample: a new conversation shares no KV-cache prefix with the last one."""
        self.cache = None
        self.cached_ids = []

    def close(self) -> None:
        pass  # No subprocess to tear down; kept for interface parity with the old backend.

    def generate(self, messages: list[dict], *, tools: list[dict], deadline: float) -> dict:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopAtDeadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return time.perf_counter() >= deadline

        if time.perf_counter() >= deadline:
            raise TimeoutError("Sample time budget exhausted before LLM")
        encoded = self.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        ids = encoded["input_ids"][0].tolist()
        if len(ids) + self.max_new_tokens > self.context_size:
            raise ValueError(f"{len(ids)} input + {self.max_new_tokens} output > {self.context_size}")
        # Reuse the KV-cache prefix shared with the previous turn's prompt (same sample only).
        prefix = 0
        if self.cache is not None:
            limit = min(len(ids) - 1, len(self.cached_ids), self.cache.get_seq_length())
            while prefix < limit and ids[prefix] == self.cached_ids[prefix]:
                prefix += 1
            if prefix:
                self.cache.crop(prefix)
            else:
                self.reset()
        inputs = encoded.to(self.device)
        if prefix:
            inputs["input_ids"] = inputs["input_ids"][:, prefix:]
        uncached_length = inputs["input_ids"].shape[1]
        with self.torch.inference_mode():
            output = self.llm.generate(
                **inputs, past_key_values=self.cache, use_cache=True,
                return_dict_in_generate=True, max_new_tokens=self.max_new_tokens,
                do_sample=False, num_beams=1, pad_token_id=self.tokenizer.eos_token_id,
                tokenizer=self.tokenizer, stop_strings=["</tool_call>"],
                stopping_criteria=StoppingCriteriaList([StopAtDeadline()]),
            )
        tokens = output.sequences[0, uncached_length:]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        self.cache = output.past_key_values
        if self.cache is not None:
            self.cached_ids = (ids[:prefix] + output.sequences[0].tolist())[:self.cache.get_seq_length()]
        if time.perf_counter() >= deadline:
            raise TimeoutError("LLM exceeded sample budget")
        eos_ids = {self.tokenizer.eos_token_id} if isinstance(self.tokenizer.eos_token_id, int) \
            else set(self.tokenizer.eos_token_id or [])
        if len(tokens) >= self.max_new_tokens and "</tool_call>" not in text and not (set(tokens.tolist()) & eos_ids):
            raise ValueError("LLM output hit token limit; refusing a truncated action")
        return _parse_decision(text)
