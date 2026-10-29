"""Vietnamese speech recognition with the original Whisper Turbo checkpoint."""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class WhisperASR:
    def __init__(self, model_name: str = "openai/whisper-small") -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.torch = torch
        # A 4 GB GPU needs its VRAM for the quantized LLM. Whisper Turbo on CPU
        # is slower, but avoids loading two large models onto that GPU together.
        gpu_memory = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        self.device = "cuda" if gpu_memory > 6 * 1024**3 else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading ASR %s on %s", model_name, self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

    def transcribe(self, audio_path: str | Path, *, deadline: float | None = None) -> str:
        import soundfile as sf
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopAtDeadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return deadline is not None and time.perf_counter() >= deadline

        audio, sample_rate = sf.read(str(audio_path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz: {audio_path}")
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True
        )
        input_features = inputs.input_features.to(self.device, dtype=self.model.dtype)
        attention_mask = inputs.attention_mask.to(self.device)
        with self.torch.inference_mode():
            token_ids = self.model.generate(
                input_features, attention_mask=attention_mask, language="vi", task="transcribe",
                do_sample=False, num_beams=1, max_new_tokens=128,
                stopping_criteria=StoppingCriteriaList([StopAtDeadline()]),
            )
        return self.processor.batch_decode(token_ids, skip_special_tokens=True)[0].strip()
