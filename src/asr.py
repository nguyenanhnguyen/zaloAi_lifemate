"""ASR module dùng faster-whisper để transcribe audio."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


def _load_model(model_name: str = "openai/whisper-large-v3-turbo"):
    """Tải faster-whisper model, fallback CPU nếu CUDA không khả dụng."""
    from faster_whisper import WhisperModel  # type: ignore

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    compute_type = "int8_float16" if device == "cuda" else "int8"
    logger.info("Loading Whisper model on %s with compute_type=%s", device, compute_type)
    return WhisperModel(model_name, device=device, compute_type=compute_type)


def transcribe(
    audio_path: str | Path,
    hotwords: list[str],
    *,
    model=None,
    beam_size: int = 1,
) -> str:
    """Transcribe audio file, trả về text thuần.

    Args:
        audio_path: Đường dẫn file audio.
        hotwords: Danh sách từ khoá ưu tiên (tên app, tool, địa chỉ...).
        model: WhisperModel đã load sẵn (nếu None sẽ load mới).
        beam_size: 1 = greedy decoding (deterministic, nhanh hơn).

    Returns:
        Chuỗi transcript đã ghép các segment.
    """
    if model is None:
        model = _load_model()

    # Dùng initial_prompt để boost hotwords vào decoder
    initial_prompt = ", ".join(hotwords) if hotwords else None

    segments, _info = model.transcribe(
        str(audio_path),
        beam_size=beam_size,
        initial_prompt=initial_prompt,
        language="vi",  # tiếng Việt
    )
    return " ".join(seg.text.strip() for seg in segments)
