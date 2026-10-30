"""Vietnamese ASR; CPU on 4 GB cards leaves dedicated VRAM for the agent."""
from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class WhisperASR:
    def __init__(self, model_name: str, *, beams: int = 5) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        self.beams = beams
        free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
        self.device = "cuda" if free > 10 * 1024**3 else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading ASR %s on %s (beam=%d)", model_name, self.device, beams)
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        for flag in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, flag, None)

    def transcribe(self, audio_path: str | Path, *, deadline: float | None = None) -> str:
        import soundfile as sf
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopAtDeadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return deadline is not None and time.perf_counter() >= deadline

        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Sample budget exhausted before ASR")
        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz: {audio_path}")
        long_audio = len(audio) > 30 * sample_rate
        options = {"truncation": False, "padding": "longest"} if long_audio else {}
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt",
                                return_attention_mask=True, **options)
        with self.torch.inference_mode():
            token_ids = self.model.generate(
                inputs.input_features.to(self.device, dtype=self.model.dtype),
                attention_mask=inputs.attention_mask.to(self.device),
                language="vi", task="transcribe", do_sample=False,
                num_beams=self.beams, max_new_tokens=256, return_timestamps=long_audio,
                stopping_criteria=StoppingCriteriaList([StopAtDeadline()]),
            )
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("ASR exceeded sample budget; incomplete transcript discarded")
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
