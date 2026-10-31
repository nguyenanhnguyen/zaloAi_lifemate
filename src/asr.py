"""Vietnamese ASR via vinai/PhoWhisper-large: batches one sample's audio in a single GPU call.

Research (2026-10-31, see EXPERIMENTS.md): an external benchmark (TSPC, arXiv 2509.05983) found
PhoWhisper-base (74M, Vietnamese-specific) scores far lower WER than Qwen3-ASR-0.6B (782M, general
multilingual) on Vietnamese test data — 14.05 vs 22.47. PhoWhisper's own paper reports its -large
checkpoint as SOTA across Vietnamese ASR benchmarks. Switching the whole family in (not just -base)
spends more of the <6B system budget on the model that research shows actually moves WER, while
staying on a long-established architecture (Whisper via transformers) instead of the very new,
narrowly-tested Qwen3ASR API.

GPU-first, no manual CPU/GPU split: .to("cuda") and let the driver fall back to shared/system
memory if VRAM is tight (slower, but no complex layer configuration). On a GPU too small to hold
both this and the LLM (<~6 GiB free when this loads first), falls back to CPU instead of starving
the LLM's activation memory later.
"""
from __future__ import annotations

import logging
import time
import unicodedata
from pathlib import Path

from .models import model_path

logger = logging.getLogger(__name__)
ASR_REPO = "vinai/PhoWhisper-large"
ASR_REVISION = "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb"


class PhoWhisperASR:
    def __init__(self, model_name: str, *, max_new_tokens: int = 256, beams: int = 1,
                 device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.beams = beams
        # ASR is only ~1% of a sample's time (measured 15-17 s/sample on GPU, ~25 s on CPU), so on
        # a small card it is worth running here on CPU to leave every byte of VRAM to the LLM,
        # whose speed collapses the moment its weights spill out of VRAM. See EXPERIMENTS.md.
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info("Loading ASR %s on %s (dtype=%s)", model_name, self.device, self.dtype)
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name, dtype=self.dtype, low_cpu_mem_usage=True, local_files_only=True,
            attn_implementation="sdpa",
        ).to(self.device)
        self.model.eval()
        for flag in ("temperature", "top_p", "top_k"):
            setattr(self.model.generation_config, flag, None)

    def transcribe_batch(self, audio_paths: list[str | Path], *, deadline: float | None = None) -> list[str]:
        """One padded batch forward pass for every audio file of a sample (request + history)."""
        import soundfile as sf
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopAtDeadline(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return deadline is not None and time.perf_counter() >= deadline

        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("Sample budget exhausted before ASR")
        audios = []
        for path in audio_paths:
            audio, sample_rate = sf.read(str(path), dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sample_rate != 16000:
                raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz: {path}")
            audios.append(audio)
        long_audio = any(len(audio) > 30 * 16000 for audio in audios)
        options = {"truncation": False, "padding": "longest"} if long_audio else {}
        inputs = self.processor(audios, sampling_rate=16000, return_tensors="pt",
                                return_attention_mask=True, **options)
        with self.torch.inference_mode():
            token_ids = self.model.generate(
                inputs.input_features.to(self.device, dtype=self.model.dtype),
                attention_mask=inputs.attention_mask.to(self.device),
                language="vi", task="transcribe", do_sample=False,
                num_beams=self.beams, max_new_tokens=self.max_new_tokens, return_timestamps=long_audio,
                stopping_criteria=StoppingCriteriaList([StopAtDeadline()]),
            )
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError("ASR exceeded sample budget; incomplete transcript discarded")
        texts = self.processor.batch_decode(token_ids, skip_special_tokens=True)
        return [unicodedata.normalize("NFC", text).strip() for text in texts]
