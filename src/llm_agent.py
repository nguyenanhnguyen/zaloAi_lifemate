"""Deterministic local LLM for a single JSON tool decision."""

from __future__ import annotations

import json
import logging
import re
import time

logger = logging.getLogger(__name__)


class LLMAgent:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct", load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available():
            raise RuntimeError(
                "PyTorch hiện là bản CPU; LLM 4B không thể chạy kịp trên máy này. "
                "Hãy cài PyTorch CUDA rồi kiểm tra torch.cuda.is_available()."
            )
        quantization = None
        if load_in_4bit and torch.cuda.is_available():
            quantization = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16,
            )
        logger.info("Loading LLM %s", model_name)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization, device_map={"": 0},
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            attn_implementation="sdpa",
        )
        self.model.eval()
        for flag in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, flag, None)

    def generate(self, prompt: str, max_new_tokens: int = 160, *, deadline: float | None = None) -> str:
        from transformers import StoppingCriteria, StoppingCriteriaList
        messages = [
            {"role": "system", "content": "Reply with one JSON object only."},
            {"role": "user", "content": prompt},
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        prompt_length = inputs["input_ids"].shape[1]
        if prompt_length + max_new_tokens > 4096:
            raise ValueError(f"Prompt too long: {prompt_length} tokens; limit is 4096 including output")
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Sample time budget exhausted before LLM generation")
        tokenizer = self.tokenizer

        class StopDecision(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if deadline is not None and time.perf_counter() >= deadline:
                    return True
                text = tokenizer.decode(input_ids[0][prompt_length:], skip_special_tokens=True).strip()
                try:
                    data = json.loads(text)
                    return isinstance(data, dict) and "action" in data
                except ValueError:
                    return False

        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([StopDecision()]),
            )
        tokens = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(tokens, skip_special_tokens=True).strip()

    @staticmethod
    def parse_decision(response: str) -> dict | None:
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE)
        try:
            data, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
        except (ValueError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None
