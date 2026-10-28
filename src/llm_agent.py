"""Deterministic local LLM for a single JSON tool decision."""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)


class LLMAgent:
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Instruct-2507", load_in_4bit: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
            model_name, quantization_config=quantization, device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        self.model.eval()

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [
            {"role": "system", "content": "Reply with one JSON object only."},
            {"role": "user", "content": prompt},
        ]
        chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(chat, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
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
