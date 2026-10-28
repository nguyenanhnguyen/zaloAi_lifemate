"""LLM Agent dùng Qwen3-4B với BitsAndBytes 4-bit quantization."""

from __future__ import annotations

import json
import logging
import re

import torch

logger = logging.getLogger(__name__)

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{[^{}]*\"name\"\s*:[^{}]*\"arguments\"\s*:[^{}]*\}", re.DOTALL)


class LLMAgent:
    """LLM Agent ReAct với Qwen3-4B-Instruct 4-bit quantized."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B-Instruct",
        load_in_4bit: bool = True,
    ) -> None:
        """Khởi tạo model và tokenizer với BitsAndBytesConfig 4-bit.

        Args:
            model_name: HuggingFace model ID.
            load_in_4bit: Dùng 4-bit quantization để tiết kiệm VRAM.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        logger.info("Loading LLM: %s (4bit=%s)", model_name, load_in_4bit)

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model.eval()
        logger.info("LLM loaded successfully.")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Sinh text từ prompt, greedy decoding (temperature=0).

        Args:
            prompt: Chuỗi prompt đầy đủ (system + user).
            max_new_tokens: Số token tối đa sinh thêm.

        Returns:
            Chuỗi response của model (chỉ phần mới sinh).
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,          # greedy
                temperature=None,         # tắt sampling
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Chỉ lấy phần token mới sinh (bỏ prompt)
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def parse_tool_call(self, response: str) -> tuple[str, dict] | None:
        """Parse JSON tool call từ response LLM.

        Hỗ trợ:
          - ```json {"name": ..., "arguments": {...}} ```
          - {"name": ..., "arguments": {...}} (bare JSON)

        Args:
            response: Chuỗi response thô từ model.

        Returns:
            (tool_name, arguments) hoặc None nếu không parse được.
        """
        # Thử tìm trong code block trước
        match = _JSON_BLOCK_RE.search(response)
        raw_json = match.group(1) if match else None

        # Fallback: tìm bare JSON object
        if raw_json is None:
            match = _BARE_JSON_RE.search(response)
            if match:
                raw_json = match.group(0)

        if raw_json is None:
            return None

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            # Thử sửa JSON đơn giản: trailing comma
            try:
                data = json.loads(re.sub(r",\s*([}\]])", r"\1", raw_json))
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from response: %s", raw_json[:200])
                return None

        name = data.get("name") or data.get("tool") or data.get("function")
        arguments = data.get("arguments") or data.get("args") or data.get("parameters") or {}

        if not name:
            return None

        return str(name), dict(arguments)
