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


def _fold(value: str) -> str:
    """Lowercase, strip Vietnamese accents, collapse to letters/digits/spaces."""
    value = unicodedata.normalize("NFD", str(value).lower()).replace("đ", "d").replace("Đ", "d")
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"[a-z0-9]+", value))


def _trigrams(value: str) -> set[str]:
    """Character 3-grams of folded text. Robust to the spelling errors Vietnamese ASR makes:
    'bao thich' vs 'bao thuc' still share most trigrams, where token equality shares nothing."""
    folded = f" {_fold(value)} "
    return {folded[i:i + 3] for i in range(max(0, len(folded) - 2))}


def _relevance(request: set[str], text: str) -> float:
    """Dice overlap between the request's trigrams and a candidate's, in [0, 1]."""
    other = _trigrams(text)
    if not request or not other:
        return 0.0
    return 2 * len(request & other) / (len(request) + len(other))


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
    # The model sometimes emits a call as bare prose, e.g. `sys.ask_missing_info({"tool_name": ...})`
    # or a raw JSON object. Treating that as a final answer silently submits zero tool calls and
    # scores 0, so recover the intent instead of discarding it.
    loose = re.search(r"([a-z_]+\.[a-z_]+)\s*\(\s*(\{.*\})\s*\)", text, re.S | re.I)
    if loose:
        try:
            args = json.loads(loose.group(2))
        except ValueError as exc:
            raise InvalidDecision(f"Unparsable loose tool call: {text[:160]}") from exc
        if isinstance(args, dict):
            return {"tool_calls": [{"id": "call_0", "type": "function",
                                    "function": {"name": loose.group(1),
                                                 "arguments": json.dumps(args, ensure_ascii=False)}}],
                    "content": ""}
    if re.match(r"\s*[\[{]", text) or '"tool_name"' in text or '"arguments"' in text:
        raise InvalidDecision(f"Looks like a malformed tool call, not a final answer: {text[:160]}")
    return {"tool_calls": [], "content": text}


class LLMAgent:
    """Own one Qwen3 model on GPU; reuse the KV-cache prefix across tool turns of one sample."""

    def __init__(self, model_name: str | None = None, *, context_size: int = 4096,
                 max_new_tokens: int = 256, precision: str = "bf16",
                 prompt_token_budget: int = 1800) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if precision not in {"bf16", "4bit"}:
            raise ValueError("precision must be 'bf16' or '4bit'")
        self.torch = torch
        self.context_size = context_size
        self.max_new_tokens = max_new_tokens
        self.precision = precision
        self.prompt_token_budget = prompt_token_budget
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        model = model_name or model_path(LLM_REPO, LLM_REVISION)
        # Precision decides whether the weights fit in VRAM, which dominates speed far more than
        # raw GPU power: bf16 is 8.0 GB for this 4B model, so on a 4 GB card ~4 GB spills into
        # system RAM and every generated token streams it back over PCIe (measured: ~900 s per
        # LLM turn). 4-bit NF4 is ~2.3 GB and stays resident (measured: see EXPERIMENTS.md).
        # bf16 is what the organisers' A30 runs; 4bit is the dev-speed option for small GPUs.
        quantization_config = None
        if precision == "4bit":
            if self.device != "cuda":
                raise ValueError("4-bit quantization requires a CUDA GPU")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
        logger.info("Loading LLM %s on %s (precision=%s)", model, self.device, precision)
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

    def fit_tools(self, messages: list[dict], tools: list[dict], request_text: str = "",
                  app_hints: dict[str, str] | None = None) -> list[dict]:
        """Keep the rendered prompt near `prompt_token_budget`, not merely inside the context.

        Prefill cost explodes with prompt length on a small GPU because this PyTorch build has no
        flash-attention kernel, so attention peaks well past VRAM and spills. Measured on an RTX
        2050 4 GB (4-bit Qwen3-4B, 32 new tokens): 1k prompt 5.5 s / 3.05 GiB peak, 2k 28.3 s /
        4.11 GiB, 3.9k 252.5 s / 7.54 GiB. Staying near ~1.4k tokens is therefore worth far more
        than any other optimisation here. Tool schemas kept are always complete; only prose and,
        as a last resort, less relevant tools are dropped.
        """
        limit = min(self.prompt_token_budget, self.context_size - self.max_new_tokens)
        if self._prompt_tokens(messages, tools) <= limit:
            return tools
        compact = copy.deepcopy(tools)
        for tool in compact:
            tool["function"]["parameters"] = _compact_schema(tool["function"].get("parameters", {}))
            # Drop the "Returns: ..." tail and keep the description short: every token saved here
            # lets another whole app stay in the prompt, and app coverage matters far more for
            # correctness than per-tool prose (a wrongly routed app loses the sample outright).
            summary = re.split(r"\s*Returns:", tool["function"].get("description", ""))[0]
            tool["function"]["description"] = summary[:60]
        if self._prompt_tokens(messages, compact) <= limit:
            logger.info("Tool prompt compacted to fit %d tokens", limit)
            return compact
        # Route by app, not by individual tool: a task needs an app's read AND write tools together
        # (smart_home.get_status to find a device_id, then smart_home.light_set to use it), so
        # dropping half an app is worse than dropping a whole irrelevant one. Rank apps against the
        # Vietnamese request using the catalog's own Vietnamese names/descriptions, which `app_hints`
        # supplies; English tool identifiers share almost nothing with a Vietnamese transcript.
        request = _trigrams(request_text)
        hints = app_hints or {}
        by_app: dict[str, list[int]] = {}
        for index, tool in enumerate(compact):
            by_app.setdefault(tool["function"]["name"].split(".")[0], []).append(index)
        mandatory = {i for i, tool in enumerate(compact)
                     if tool["function"]["name"].startswith("sys.")}
        ranked_apps = sorted(
            (app for app in by_app if app != "sys"),
            key=lambda app: _relevance(request, hints.get(app, app) + " " + " ".join(
                compact[i]["function"]["description"] for i in by_app[app])),
            reverse=True)

        def choose(app_count: int) -> list[dict]:
            keep = set(mandatory)
            for app in ranked_apps[:app_count]:
                keep.update(by_app[app])
            return [tool for i, tool in enumerate(compact) if i in keep]

        low, high = 0, len(ranked_apps)
        while low < high:
            middle = (low + high + 1) // 2
            if self._prompt_tokens(messages, choose(middle)) <= limit:
                low = middle
            else:
                high = middle - 1
        selected = choose(max(low, 1))  # never route down to sys-only: that guarantees a wrong answer
        if not selected:
            raise ValueError(f"Cannot fit any tool schema within {limit} tokens")
        logger.warning("Tool prompt routed to %d/%d tools from apps %s (budget %d tokens)",
                       len(selected), len(tools), ranked_apps[:max(low, 1)], limit)
        return selected

    def reset(self) -> None:
        """Call once per sample: a new conversation shares no KV-cache prefix with the last one."""
        self.cache = None
        self.cached_ids = []

    def close(self) -> None:
        pass  # No subprocess to tear down; kept for interface parity with the old backend.

    def generate(self, messages: list[dict], *, tools: list[dict], deadline: float) -> dict:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from transformers import StoppingCriteria, StoppingCriteriaList

        _SDPA_BACKENDS = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

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
        # This PyTorch build ships no flash-attention kernel, and the math backend materialises the
        # full attention matrix (measured 7.54 GiB peak for a 3.9k prompt). Prefer mem-efficient.
        with self.torch.inference_mode(), sdpa_kernel(_SDPA_BACKENDS):
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
