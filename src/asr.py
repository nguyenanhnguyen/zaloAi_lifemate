"""Vietnamese ASR on the Whisper encoder-decoder architecture, one batched pass per sample.

Every checkpoint in ASR_MODELS speaks the same `transformers` API (`AutoModelForSpeechSeq2Seq` +
`AutoProcessor`), so swapping one for another needs no new dependency and no code path of its own —
which is the point: a frozen Docker submission cannot afford a per-model runtime.

Why a choice of checkpoint rather than one fixed model (2026-10-31, see EXPERIMENTS.md):

- `vinai/PhoWhisper-large` (1.543B, BSD-3) is the Vietnamese-specific reference. An external
  benchmark (TSPC, arXiv 2509.05983) puts even PhoWhisper-base at 14.05 WER against Qwen3-ASR-0.6B's
  22.47 on Vietnamese, and its own paper reports -large as SOTA on Vietnamese benchmarks at release.
- `erax-ai/EraX-WoW-Turbo-V1.1` (809M, MIT) is a Vietnamese fine-tune of whisper-large-v3-turbo. It
  keeps the SAME 32-layer encoder (d_model 1280) but has a **4-layer decoder instead of 32**. The
  encoder cost per sample is therefore unchanged while autoregressive decoding gets ~8x cheaper —
  and decoding is what dominates on a CPU. 734M fewer parameters also frees system budget.
- `openai/whisper-large-v3-turbo` (809M, MIT) is the same turbo architecture WITHOUT Vietnamese
  fine-tuning: the control that separates "newer base + shallow decoder" from "Vietnamese training".

Measured cost is roughly FIXED per audio file, not per second of speech (~16 s/file on this CPU for
PhoWhisper-large), because Whisper always pads the mel spectrogram to 30 s. A sample with two
history audio turns therefore costs about three times one without.
"""
from __future__ import annotations

import logging
import time
import unicodedata
from pathlib import Path

from .models import model_path

logger = logging.getLogger(__name__)
# repo id -> pinned revision. Every entry must predate the competition's 2026-10-26 cutoff and
# carry a permissive licence (BSD-3 / MIT); NonCommercial checkpoints are disqualified.
ASR_MODELS = {
    "vinai/PhoWhisper-large": "b9136a44b5f2ca664bd0b8f74baecf1715f6eeeb",
    "erax-ai/EraX-WoW-Turbo-V1.1": "1b3a8daa1758490b19b20a63b09cdf8572a88a2c",
    "openai/whisper-large-v3-turbo": "41f01f3fe87f28c78e2fbf8b568835947dd65ed9",
}
# Default stays the Vietnamese-specific model, so a bare run matches the Kaggle notebook and the
# submission. Laptop experiments select another checkpoint explicitly; see README.md.
ASR_REPO = "vinai/PhoWhisper-large"
ASR_REVISION = ASR_MODELS[ASR_REPO]


def asr_revision(repo: str) -> str:
    """Pinned revision for a known repo id; empty string for a local directory or unknown repo."""
    return ASR_MODELS.get(repo, "")


class WhisperASR:
    def __init__(self, model_name: str, *, max_new_tokens: int = 256, beams: int = 1,
                 device: str = "cuda") -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if device not in {"cuda", "cpu"}:
            raise ValueError("device must be 'cuda' or 'cpu'")
        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.beams = beams
        # Measured after the LLM was optimised: ASR is 40% of total runtime (357 s of 893 s over 15
        # samples), not the ~1% it was when a single LLM turn cost 900 s. On a small card it still
        # runs on CPU by default to leave every byte of VRAM to the LLM, whose speed collapses once
        # its weights spill out of VRAM -- but that trade is now expensive enough to measure both
        # ways rather than assume. See EXPERIMENTS.md.
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

    def close(self) -> None:
        """Release the weights so the next model gets the whole card (see CachedASR)."""
        self.model = self.processor = None
        if self.device == "cuda":
            self.torch.cuda.empty_cache()


class CachedASR:
    """Replays transcripts produced by an earlier ASR phase; same interface as WhisperASR.

    Why a separate phase exists: on a 4 GB card the ASR and the LLM cannot both be resident, and
    whichever one spills runs ~8x slower. Measured on this laptop, PhoWhisper-large costs 24.5 s per
    audio file on CPU against 3.1 s on GPU for an identical transcript (WER 23.59% either way). So
    the laptop transcribes everything first with the card to itself, frees the weights, then loads
    the LLM. An A30 holds both at once and needs none of this.
    """

    def __init__(self, transcripts: dict[str, str]) -> None:
        self.transcripts = transcripts

    def transcribe_batch(self, audio_paths: list[str | Path], *, deadline: float | None = None) -> list[str]:
        texts = []
        for path in audio_paths:
            key = str(path)
            if key not in self.transcripts:
                raise KeyError(f"No cached transcript for {key}; rerun the ASR phase")
            texts.append(self.transcripts[key])
        return texts

    def close(self) -> None:
        return None
