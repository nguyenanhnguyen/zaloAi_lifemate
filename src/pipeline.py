"""LifeMate inference: speech -> tool decisions -> ToolEnv calls."""

from __future__ import annotations

import json
import logging
import time
import unicodedata
from pathlib import Path

from .asr import WhisperASR
from .llm_agent import LLMAgent

logger = logging.getLogger(__name__)
MAX_STEPS = 3


def _available_tools(catalog: dict, installed_apps: dict) -> list[dict]:
    allowed = {name for app in installed_apps.values() for name in app.get("tools", [])}
    allowed.add("sys.ask_missing_info")
    return [
        tool for app in catalog.get("apps", []) for tool in app.get("tools", [])
        if tool.get("name") in allowed
    ]


def _prompt(item: dict, transcript: str, tools: list[dict], observations: list[dict]) -> str:
    tool_lines = []
    for tool in tools:
        params = tool.get("parameters", {})
        properties = []
        for name, spec in params.get("properties", {}).items():
            detail = f"{name}:{spec.get('type', 'any')}"
            if name in params.get("required", []):
                detail += "!"
            if spec.get("enum"):
                detail += f"={','.join(map(str, spec['enum']))}"
            if spec.get("x-ref"):
                detail += f"[ref:{spec['x-ref']}]"
            if spec.get("x-match"):
                detail += f"[match:{spec['x-match']}]"
            properties.append(detail)
        one_of = f" one-of:{params['x-one-of']}" if params.get("x-one-of") else ""
        tool_lines.append(
            f"{tool['name']} [{tool.get('kind', '')}]: "
            f"{tool.get('description', '')[:60]} | {', '.join(properties)}{one_of}"
        )
    context = item.get("context", {})
    useful_context = {key: context.get(key) for key in ("current_time", "timezone", "location", "user_profile")}
    useful_context["installed_apps"] = list(context.get("installed_apps", {}))
    return (
        "Bạn là LifeMate. Chuyển yêu cầu nói thành các tool calls hợp lệ. "
        "Chỉ trả về đúng một JSON object, không markdown.\n"
        'Khi gọi tool, trả về {"action":"call","name":"app.tool","arguments":{...}}. '
        'Khi đã xong hoặc ngoài phạm vi, trả về {"action":"done"}. '
        "Mỗi lượt chỉ chọn một tool; sau khi gọi sẽ nhận kết quả rồi quyết định tiếp. "
        "Chỉ dùng tool trong danh sách. Đọc bằng read tool khi cần ID hoặc trạng thái. "
        "Nếu thiếu thông tin cần thiết, gọi sys.ask_missing_info đúng schema. "
        "Không đoán ID, địa điểm hoặc thông tin người dùng. Nếu âm thanh không rõ điểm đến, hãy hỏi thiếu thông tin. "
        "Nếu người dùng nói đặt ngay, đừng điền schedule_time. Giữ nguyên chữ hoa và nội dung tự do khi cần.\n"
        f"Yêu cầu: {transcript}\n"
        f"Context: {json.dumps(useful_context, ensure_ascii=False)}\n"
        f"History: {json.dumps(item.get('history', []), ensure_ascii=False)}\n"
        f"Tools (! = required):\n" + "\n".join(tool_lines) + "\n"
        f"Kết quả các bước trước: {json.dumps(observations, ensure_ascii=False)}\n"
        "JSON:"
    )


class Pipeline:
    """Load models once; the runner creates one ToolEnv per sample."""

    def __init__(self, asr_model: WhisperASR, llm_agent: LLMAgent | None = None,
                 *, max_steps: int = MAX_STEPS, max_seconds: float = 28) -> None:
        self.asr_model = asr_model
        self.llm_agent = llm_agent or LLMAgent()
        self.last_asr_seconds = 0.0
        self.last_llm_seconds = 0.0
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.stop_reason = "not_started"

    def infer(self, item: dict, env, *, catalog: dict, audio_path: Path) -> tuple[list[dict], str]:
        deadline = time.perf_counter() + self.max_seconds
        self.stop_reason = "step_limit"
        installed = item.get("context", {}).get("installed_apps", {})
        tools = _available_tools(catalog, installed)
        names = {tool["name"] for tool in tools}
        self.last_asr_seconds = 0.0
        self.last_llm_seconds = 0.0
        asr_started = time.perf_counter()
        transcript = self.asr_model.transcribe(audio_path, deadline=deadline)
        self.last_asr_seconds = time.perf_counter() - asr_started
        transcript = unicodedata.normalize("NFC", transcript).strip()
        logger.info("%s transcript: %s", item["id"], transcript)

        observations: list[dict] = []
        previous_rejected = None
        for step in range(self.max_steps):
            if time.perf_counter() >= deadline:
                self.stop_reason = "time_limit"
                break
            llm_started = time.perf_counter()
            response = self.llm_agent.generate(_prompt(item, transcript, tools, observations), deadline=deadline)
            self.last_llm_seconds += time.perf_counter() - llm_started
            if time.perf_counter() >= deadline:
                self.stop_reason = "time_limit"
                break
            decision = self.llm_agent.parse_decision(response)
            if decision is None:
                logger.warning("%s step %d: invalid LLM JSON: %s", item["id"], step + 1, response[:200])
                self.stop_reason = "invalid_json"
                break
            logger.info("%s step %d decision: %s", item["id"], step + 1, decision)
            if decision.get("action") == "call|done":
                # Recover from the ambiguous prompt used by older checkpoints/runs.
                decision["action"] = "call" if decision.get("name") else "done"
            if decision.get("action") == "done":
                self.stop_reason = "done"
                break
            name = decision.get("name")
            args = decision.get("arguments", {})
            signature = json.dumps(decision, ensure_ascii=False, sort_keys=True)
            if decision.get("action") != "call" or name not in names or not isinstance(args, dict):
                if signature == previous_rejected:
                    logger.warning("%s repeated invalid decision; stopping", item["id"])
                    break
                previous_rejected = signature
                observations.append({"error": "Unknown action or unavailable tool", "decision": decision})
                continue
            validation = env.validate(name, args)
            if not validation.get("ok"):
                if signature == previous_rejected:
                    logger.warning("%s repeated invalid arguments; stopping", item["id"])
                    break
                previous_rejected = signature
                logger.warning("%s invalid call %s: %s", item["id"], name, validation.get("error"))
                observations.append({"tool": name, "arguments": args, "error": validation.get("error")})
                continue
            if any(call["name"] == name and call["arguments"] == args for call in env.get_tool_calls()):
                logger.warning("%s repeated tool call; stopping", item["id"])
                self.stop_reason = "repeated_call"
                break
            previous_rejected = None
            result = env.call(name, args)
            observations.append({"tool": name, "arguments": args, "result": result})
            if name == "sys.ask_missing_info":
                self.stop_reason = "asked_missing"
                break

        logger.info("%s stopped: %s", item["id"], self.stop_reason)
        return env.get_tool_calls(), env.get_state_hash()
