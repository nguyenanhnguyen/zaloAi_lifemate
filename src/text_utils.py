"""Normalize fuzzy Vietnamese text và trích hotwords từ context."""

from __future__ import annotations

import re
import unicodedata


# Từ đệm phổ biến đầu/cuối câu tiếng Việt cần loại bỏ
_FILLER_HEAD = re.compile(
    r"^(hãy|làm ơn|vui lòng|giúp tôi|giùm tôi|hộ tôi|ơi|hey|ê|à|ừ)\s+",
    re.IGNORECASE,
)
_FILLER_TAIL = re.compile(
    r"\s+(nhé|nha|đi|giùm|hộ tôi|giúp tôi|với|được không|không|ạ|nhá|đó|thôi)$",
    re.IGNORECASE,
)


def normalize_fuzzy(text: str) -> str:
    """NFC normalize, lowercase, cắt từ đệm đầu/cuối câu.

    Args:
        text: Chuỗi thô từ ASR hoặc user.

    Returns:
        Chuỗi đã được chuẩn hóa.
    """
    # NFC normalize (hợp nhất dấu tổ hợp)
    text = unicodedata.normalize("NFC", text)
    text = text.strip().lower()

    # Cắt từ đệm đầu (lặp nhiều lần để loại hết)
    for _ in range(3):
        new = _FILLER_HEAD.sub("", text)
        if new == text:
            break
        text = new.strip()

    # Cắt từ đệm cuối (lặp nhiều lần)
    for _ in range(3):
        new = _FILLER_TAIL.sub("", text)
        if new == text:
            break
        text = new.strip()

    return text


def extract_hotwords(item: dict) -> list[str]:
    """Lấy tên app, tool, địa chỉ từ installed_apps và history.

    Args:
        item: Dict chứa context và history của request.

    Returns:
        Danh sách hotwords duy nhất.
    """
    hotwords: set[str] = set()
    context: dict = item.get("context", {})

    # Tên app đã cài
    installed_apps: dict = context.get("installed_apps", {})
    for app_name, app_info in installed_apps.items():
        if app_name:
            hotwords.add(app_name)
        # Tên hiển thị nếu có
        if isinstance(app_info, dict):
            display = app_info.get("display_name") or app_info.get("name")
            if display:
                hotwords.add(display)
            # Tên tools trong app
            for tool in app_info.get("tools", []):
                if isinstance(tool, dict) and tool.get("name"):
                    hotwords.add(tool["name"])

    # Địa chỉ, tên người từ history
    history: list[dict] = item.get("history", [])
    for turn in history:
        content = turn.get("content", "")
        if isinstance(content, str) and len(content) < 200:
            # Lấy token >= 4 ký tự, viết hoa chữ đầu (entity-like)
            tokens = content.split()
            for tok in tokens:
                tok_clean = re.sub(r"[^\w]", "", tok)
                if len(tok_clean) >= 4:
                    hotwords.add(tok_clean)

    return [w for w in hotwords if w]
