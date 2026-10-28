"""LifeMate inference: speech -> tool decisions -> ToolEnv calls."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

from .asr import WhisperASR
from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)
MAX_STEPS = 12


def _available_tools(catalog: dict, installed_apps: dict) -> list[dict]:
    allowed = {name for app in installed_apps.values() for name in app.get("tools", [])}
    allowed.add("sys.ask_missing_info")
    return [
        tool for app in catalog.get("apps", []) for tool in app.get("tools", [])
        if tool.get("name") in allowed
    ]


def _prompt(item: dict, transcript: str, tools: list[dict], observations: list[dict]) -> str:
    compact_tools = [
        {"name": t["name"], "kind": t.get("kind"), "description": t.get("description"),
         "parameters": t.get("parameters", {})} for t in tools
    ]
    context = item.get("context", {})
    useful_context = {key: context.get(key) for key in ("current_time", "timezone", "location", "user_profile")}
    useful_context["installed_apps"] = list(context.get("installed_apps", {}))
    return (
        "Bạn là LifeMate. Chuyển yêu cầu nói thành các tool calls hợp lệ. "
        "Chỉ trả về đúng một JSON object, không markdown.\n"
        'Dạng: {"action":"call|done","name":"app.tool","arguments":{...}}. '
        "Dùng done khi đã hoàn tất hoặc yêu cầu ngoài phạm vi. "
        "Mỗi lượt chỉ chọn một tool; sau khi gọi sẽ nhận kết quả rồi quyết định tiếp. "
        "Chỉ dùng tool trong danh sách. Đọc bằng read tool khi cần ID hoặc trạng thái. "
        "Nếu thiếu thông tin cần thiết, gọi sys.ask_missing_info đúng schema. "
        "Không đoán ID hay thông tin người dùng. Giữ nguyên chữ hoa và nội dung tự do khi cần.\n"
        f"Yêu cầu: {transcript}\n"
        f"Context: {json.dumps(useful_context, ensure_ascii=False)}\n"
        f"History: {json.dumps(item.get('history', []), ensure_ascii=False)}\n"
        f"Tools: {json.dumps(compact_tools, ensure_ascii=False)}\n"
        f"Kết quả các bước trước: {json.dumps(observations, ensure_ascii=False)}\n"
        "JSON:"
    )


class Pipeline:
    """Load models once; the runner creates one ToolEnv per sample."""

    def __init__(self, asr_model: WhisperASR, llm_agent: LLMAgent | None = None) -> None:
        self.asr_model = asr_model
        self.llm_agent = llm_agent or LLMAgent()

    def infer(self, item: dict, env, *, catalog: dict, audio_path: Path) -> tuple[list[dict], str]:
        installed = item.get("context", {}).get("installed_apps", {})
        tools = _available_tools(catalog, installed)
        names = {tool["name"] for tool in tools}
        transcript = self.asr_model.transcribe(audio_path)
        transcript = unicodedata.normalize("NFC", transcript).strip()
        logger.info("%s transcript: %s", item["id"], transcript)

        observations: list[dict] = []
        for step in range(MAX_STEPS):
            response = self.llm_agent.generate(_prompt(item, transcript, tools, observations))
            decision = self.llm_agent.parse_decision(response)
            if decision is None:
                logger.warning("%s step %d: invalid LLM JSON: %s", item["id"], step + 1, response[:200])
                observations.append({"error": "Output must be one valid JSON object."})
                continue
            logger.info("%s step %d decision: %s", item["id"], step + 1, decision)
            if decision.get("action") == "done":
                break
            name = decision.get("name")
            args = decision.get("arguments", {})
            if decision.get("action") != "call" or name not in names or not isinstance(args, dict):
                observations.append({"error": "Unknown action or unavailable tool", "decision": decision})
                continue
            validation = env.validate(name, args)
            if not validation.get("ok"):
                logger.warning("%s invalid call %s: %s", item["id"], name, validation.get("error"))
                observations.append({"tool": name, "arguments": args, "error": validation.get("error")})
                continue
            if any(call["name"] == name and call["arguments"] == args for call in env.get_tool_calls()):
                observations.append({"error": "This tool call was already made. Choose next step or done."})
                continue
            result = env.call(name, args)
            observations.append({"tool": name, "arguments": args, "result": result})
            if name == "sys.ask_missing_info":
                break

        return env.get_tool_calls(), env.get_state_hash()
