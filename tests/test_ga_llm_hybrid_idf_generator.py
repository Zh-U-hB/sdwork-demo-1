"""Tests for template IDF mapping."""

from __future__ import annotations

from pathlib import Path

from ga_llm_hybrid.energyplus.idf_generator import apply_mappings


def test_set_field_mapping(tmp_path: Path):
    template = tmp_path / "base.idf"
    template.write_text(
        "Version,9.6;\n\n"
        "Building,\n"
        "  Office,\n"
        "  0.0,\n"
        "  City,\n"
        "  0.04,\n"
        "  0.4,\n"
        "  0.4,\n"
        "  ,;\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.idf"
    apply_mappings(
        template,
        {},
        [],
        out,
    )
    assert out.exists()
    assert "Version" in out.read_text(encoding="utf-8")
