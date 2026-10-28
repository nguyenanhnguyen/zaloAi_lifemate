"""Pipeline chính: ghép ASR → Schema Pruning → ReAct Loop → Normalize."""

from __future__ import annotations

import hashlib
import json
import logging
import random

import torch

from .asr import transcribe
from .llm_agent import LLMAgent
from .schema_utils import (
    format_tools_for_prompt,
    get_xref_tools,
    has_missing_required,
    prune_catalog,
)
from .text_utils import extract_hotwords, normalize_fuzzy

logger = logging.getLogger(__name__)

# Seed cố định cho reproducibility
random.seed(42)
torch.manual_seed(42)
try:
    import numpy as np  # type: ignore
    np.random.seed(42)
except ImportError:
    pass

MAX_REACT_STEPS = 10


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    """Hướng dẫn ReAct loop và output format JSON cho LLM."""
    return (
        "Bạn là LifeMate AI assistant. Hãy phân tích yêu cầu của người dùng "
        "và chọn tool phù hợp để thực thi.\n\n"
        "## Quy tắc:\n"
        "1. Nếu yêu cầu nằm ngoài phạm vi tools hoặc app chưa cài: "
        'trả về {"action": "out_of_scope", "reason": "<lý do>"}\n'
        "2. Nếu tool cần dữ liệu từ read-tool (x-ref): "
        'trả về {"action": "call_read", "name": "<tool>", "arguments": {}}\n'
        "3. Nếu thiếu tham số bắt buộc: "
        'trả về {"action": "ask_missing", "name": "sys.ask_missing_info", '
        '"arguments": {"missing_fields": [...], "question": "<câu hỏi>"}}\n'
        "4. Nếu đủ thông tin, gọi tool chính: "
        '{"action": "call", "name": "<tool>", "arguments": {...}}\n\n'
        "## Output format:\n"
        "Chỉ trả về một JSON object duy nhất, không giải thích thêm.\n"
        '```json\n{"action": "...", "name": "...", "arguments": {...}}\n```'
    )


def build_user_prompt(
    transcript: str,
    context: dict,
    pruned_tools: list[dict],
    history: list[dict],
    observations: list[str],
) -> str:
    """Tạo user prompt với đầy đủ context cho LLM.

    Args:
        transcript: Văn bản từ ASR.
        context: current_time, user_profile, installed_apps.
        pruned_tools: Danh sách tool schemas đã lọc.
        history: Lịch sử hội thoại.
        observations: Kết quả các read-tool đã gọi.

    Returns:
        Chuỗi prompt đầy đủ.
    """
    # Tóm tắt history (chỉ lấy 3 lượt cuối)
    recent_history = history[-3:] if len(history) > 3 else history
    history_str = json.dumps(recent_history, ensure_ascii=False)

    # Context gọn: chỉ giữ tên app
    ctx_summary = {
        "current_time": context.get("current_time", ""),
        "user": context.get("user_profile", {}).get("name", ""),
        "installed_apps": list(context.get("installed_apps", {}).keys()),
    }

    tools_str = format_tools_for_prompt(pruned_tools)
    obs_str = "\n".join(f"- {o}" for o in observations) if observations else "Chưa có."

    return (
        f"## Yêu cầu người dùng:\n{transcript}\n\n"
        f"## Context:\n{json.dumps(ctx_summary, ensure_ascii=False)}\n\n"
        f"## Tools khả dụng:\n{tools_str}\n\n"
        f"## Lịch sử gần đây:\n{history_str}\n\n"
        f"## Observations từ read-tools:\n{obs_str}\n\n"
        "Hãy chọn action phù hợp và trả về JSON:"
    )


# ---------------------------------------------------------------------------
# Tool schema lookup helper
# ---------------------------------------------------------------------------

def _find_tool_schema(pruned_tools: list[dict], tool_name: str) -> dict | None:
    """Tìm schema của tool theo tên."""
    for t in pruned_tools:
        if t.get("name") == tool_name:
            return t
    return None


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------

def _make_hash(item: dict, transcript: str) -> str:
    """Tạo hash ngắn từ item id + transcript để làm request fingerprint."""
    raw = f"{item.get('id', '')}{transcript}"
    return hashlib.md5(raw.encode()).hexdigest()[:8]


def infer(
    item: dict,
    env,
    asr_model,
    llm_agent: LLMAgent,
    tools_catalog: dict,
) -> tuple[list[dict], str]:
    """Pipeline chính: ASR → Prune → ReAct → Normalize.

    Args:
        item: Dict chứa audio_path, context, history.
        env: Environment object có .validate(name, args) và .call(name, args).
        asr_model: WhisperModel đã load.
        llm_agent: LLMAgent đã khởi tạo.
        tools_catalog: Dict app_name -> list[tool_schema].

    Returns:
        (actions: list[dict], request_hash: str)
          actions = [] nếu out_of_scope
          mỗi action = {"name": ..., "arguments": {...}}
    """
    context: dict = item.get("context", {})
    history: list[dict] = item.get("history", [])
    audio_path: str = item.get("audio_path", "")

    # ------------------------------------------------------------------
    # Bước 1: ASR
    # ------------------------------------------------------------------
    hotwords = extract_hotwords(item)
    transcript = transcribe(audio_path, hotwords, model=asr_model)
    logger.info("[ASR] transcript: %s", transcript)

    req_hash = _make_hash(item, transcript)

    # ------------------------------------------------------------------
    # Bước 2: Schema Pruning
    # ------------------------------------------------------------------
    installed_apps: dict = context.get("installed_apps", {})
    pruned_tools = prune_catalog(tools_catalog, installed_apps)
    logger.info("[Prune] %d tools kept", len(pruned_tools))

    # ------------------------------------------------------------------
    # Bước 3: ReAct Loop
    # ------------------------------------------------------------------
    observations: list[str] = []
    actions: list[dict] = []
    system_prompt = build_system_prompt()

    for step in range(MAX_REACT_STEPS):
        logger.info("[ReAct] step %d/%d", step + 1, MAX_REACT_STEPS)

        user_prompt = build_user_prompt(
            transcript, context, pruned_tools, history, observations
        )
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        response = llm_agent.generate(full_prompt)
        logger.debug("[LLM] response: %s", response[:300])

        parsed = llm_agent.parse_tool_call(response)
        if parsed is None:
            logger.warning("[ReAct] Không parse được JSON, thử lại...")
            continue

        action_type_raw, args = parsed
        # Tách "action" và "name" từ parsed (parse_tool_call trả về name=action field)
        # Cần parse lại nếu format {"action": ..., "name": ..., "arguments": ...}
        try:
            data = json.loads(
                response[response.find("{") : response.rfind("}") + 1]
            )
        except Exception:
            data = {}

        action_type: str = data.get("action", action_type_raw)
        tool_name: str = data.get("name", "")
        arguments: dict = data.get("arguments", args)

        # ---- Luật 1: out_of_scope ----
        if action_type == "out_of_scope":
            logger.info("[ReAct] Out of scope: %s", data.get("reason", ""))
            return [], req_hash

        # ---- Luật 2: x-ref read tool ----
        if action_type == "call_read":
            tool_schema = _find_tool_schema(pruned_tools, tool_name)
            if tool_schema:
                xref_tools = get_xref_tools(tool_schema)
                for rt in xref_tools:
                    valid = env.validate(rt, {})
                    if valid:
                        obs = env.call(rt, {})
                        observations.append(f"{rt}: {json.dumps(obs, ensure_ascii=False)}")
            else:
                # Gọi thẳng read-tool nếu không tìm được schema
                valid = env.validate(tool_name, arguments)
                if valid:
                    obs = env.call(tool_name, arguments)
                    observations.append(f"{tool_name}: {json.dumps(obs, ensure_ascii=False)}")
            continue  # tiếp tục ReAct loop với observations mới

        # ---- Luật 3: ask_missing_info ----
        if action_type == "ask_missing":
            # Kiểm tra lại tool schema nếu có tên tool chính
            main_tool_schema = _find_tool_schema(pruned_tools, tool_name)
            if main_tool_schema:
                has_missing, missing_fields = has_missing_required(main_tool_schema, arguments)
                if not has_missing:
                    # LLM nhầm, không thực sự thiếu -> bỏ qua ask
                    logger.info("[ReAct] LLM claimed missing but schema OK, skipping ask.")
                    continue
            # Gọi sys.ask_missing_info
            ask_args = arguments if arguments else {"missing_fields": [], "question": ""}
            actions.append({"name": "sys.ask_missing_info", "arguments": ask_args})
            return actions, req_hash

        # ---- Luật 4: call tool (validate → call, retry 1 lần nếu fail) ----
        if action_type == "call" and tool_name:
            tool_schema = _find_tool_schema(pruned_tools, tool_name)

            # Normalize fuzzy text trong string arguments
            for k, v in arguments.items():
                if isinstance(v, str):
                    arguments[k] = normalize_fuzzy(v)

            # Validate
            valid = env.validate(tool_name, arguments)
            if not valid:
                logger.warning("[ReAct] validate failed for %s, LLM sẽ tự sửa...", tool_name)
                observations.append(
                    f"validate_error: tool={tool_name} args={json.dumps(arguments)}"
                )
                continue  # cho LLM tự sửa 1 lần (vẫn còn step)

            result = env.call(tool_name, arguments)
            actions.append({"name": tool_name, "arguments": arguments, "result": result})
            logger.info("[ReAct] called %s -> %s", tool_name, str(result)[:100])
            return actions, req_hash

    logger.warning("[ReAct] Hết %d steps, trả về actions tích lũy.", MAX_REACT_STEPS)
    return actions, req_hash
