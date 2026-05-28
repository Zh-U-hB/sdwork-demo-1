"""Apply parameter mappings to a template IDF file."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def apply_mappings(
    template_path: Path,
    params: dict[str, Any],
    mappings: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Generate IDF by applying configured mappings to a template."""
    text = template_path.read_text(encoding="utf-8", errors="replace")
    linesep = "\r\n" if "\r\n" in text else "\n"

    for rule in mappings:
        param = rule.get("parameter")
        if param not in params:
            continue
        value = params[param]
        action = rule.get("action", "set_field")

        if action == "replace_construction":
            mat_map = rule.get("material_map", {})
            mat_name = mat_map.get(str(value), mat_map.get(value))
            if mat_name:
                obj_name = rule.get("object_name", "")
                text = _replace_construction_layer(text, obj_name, str(mat_name))
        elif action == "modify_vertices" or rule.get("field") == "vertices_z_offset":
            depth = float(value)
            obj_name = rule.get("object_name", "")
            text = _offset_shading_z(text, obj_name, depth)
        elif action == "set_field":
            obj_type = rule.get("object_type", "")
            obj_name = rule.get("object_name", "")
            field = rule.get("field", "")
            text = _set_object_field(text, obj_type, obj_name, field, value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = text.replace("\r\n", "\n").replace("\n", linesep)
    output_path.write_text(normalized, encoding="utf-8")


def _replace_construction_layer(text: str, construction_name: str, layer_name: str) -> str:
    """Replace the outside layer of a Construction object."""
    pattern = (
        rf"(?ms)(Construction,\s*\n\s*{re.escape(construction_name)}\s*,[^\n]*\n)"
        rf"(\s*)([^,;\n]+)"
    )
    return re.sub(pattern, rf"\1\2{layer_name}", text, count=1)


def _offset_shading_z(text: str, shading_name: str, z_offset: float) -> str:
    """Add *z_offset* to vertex Z coordinates in a Shading:Building block."""
    block_re = (
        rf"(?ms)(Shading:Building,\s*\n\s*{re.escape(shading_name)},.*?)"
        rf"(?=\n[A-Za-z][A-Za-z0-9_:]*,|\Z)"
    )
    match = re.search(block_re, text)
    if not match:
        return text
    block = match.group(1)

    def _bump_z(line: str) -> str:
        # Only touch lines with numeric tuples (vertex lines).
        if not re.search(r"-?\d+\.?\d*;", line):
            return line
        parts = [p.strip() for p in line.split(";") if p.strip()]
        if len(parts) < 3:
            return line
        try:
            z_val = float(parts[-1])
        except ValueError:
            return line
        parts[-1] = f"{z_val + z_offset:.6f}"
        return "  " + ", ".join(parts) + ";"

    new_lines = [_bump_z(ln) for ln in block.splitlines()]
    new_block = "\n".join(new_lines)
    return text[: match.start()] + new_block + text[match.end() :]


def _set_object_field(text: str, obj_type: str, obj_name: str, field: str, value: Any) -> str:
    block_re = (
        rf"(?ms)({re.escape(obj_type)},\s*\n\s*{re.escape(obj_name)},.*?)"
        rf"(?=\n[A-Za-z][A-Za-z0-9_:]*,|\Z)"
    )
    match = re.search(block_re, text)
    if not match:
        return text
    lines = match.group(1).splitlines()
    if len(lines) > 2 and field:
        lines[2] = f"  {value},"
        new_block = "\n".join(lines)
        return text[: match.start()] + new_block + text[match.end() :]
    return text
