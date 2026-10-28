"""Schema pruning & validation helpers cho tools catalog."""

from __future__ import annotations

import json


def prune_catalog(tools_catalog: dict, installed_apps: dict) -> list[dict]:
    """Chỉ giữ tools thuộc installed_apps + sys.ask_missing_info.

    Args:
        tools_catalog: Dict mapping app_name -> list[tool_schema].
        installed_apps: Dict các app đã cài (key = app_name).

    Returns:
        Danh sách tool schema đã lọc.
    """
    pruned: list[dict] = []
    allowed_apps = set(installed_apps.keys())

    for app_name, tools in tools_catalog.items():
        # Luôn giữ system tools
        if app_name == "sys" or app_name in allowed_apps:
            if isinstance(tools, list):
                pruned.extend(tools)
            else:
                pruned.append(tools)

    return pruned


def get_xref_tools(tool_schema: dict) -> list[str]:
    """Trả về read-tools cần gọi trước theo x-ref params.

    Các param có extension 'x-ref' cần được điền bằng kết quả
    của read-tool tương ứng.

    Args:
        tool_schema: Schema của tool cần gọi.

    Returns:
        Danh sách tên read-tools cần gọi trước.
    """
    xref_tools: list[str] = []
    params = tool_schema.get("parameters", {})
    properties = params.get("properties", {})

    for _param_name, param_def in properties.items():
        ref_tool = param_def.get("x-ref")
        if ref_tool and ref_tool not in xref_tools:
            xref_tools.append(ref_tool)

    return xref_tools


def has_missing_required(
    tool_schema: dict, args: dict
) -> tuple[bool, list[str]]:
    """Kiểm tra required params & x-one-of còn thiếu.

    Args:
        tool_schema: Schema tool cần kiểm tra.
        args: Arguments LLM đã cung cấp.

    Returns:
        (has_missing: bool, missing_fields: list[str])
    """
    params = tool_schema.get("parameters", {})
    required_fields: list[str] = params.get("required", [])
    properties: dict = params.get("properties", {})
    missing: list[str] = []

    # Kiểm tra required fields
    for field in required_fields:
        if field not in args or args[field] is None or args[field] == "":
            missing.append(field)

    # Kiểm tra x-one-of groups (ít nhất 1 trong nhóm phải có)
    one_of_groups: list[list[str]] = params.get("x-one-of", [])
    for group in one_of_groups:
        if not any(args.get(f) for f in group):
            # Thêm toàn bộ group dưới dạng "a|b|c"
            missing.append("|".join(group))

    return bool(missing), missing


def format_tools_for_prompt(pruned_tools: list[dict]) -> str:
    """Serialise danh sách tools thành JSON ngắn gọn cho LLM prompt.

    Args:
        pruned_tools: Danh sách tool schemas đã prune.

    Returns:
        Chuỗi JSON compact.
    """
    compact = []
    for tool in pruned_tools:
        compact.append({
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        })
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
