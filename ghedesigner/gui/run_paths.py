from __future__ import annotations

import re
from pathlib import Path

DEFAULT_GUI_RUN_STEM = "ghedesigner_gui_run"


def demo_style_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return stem or DEFAULT_GUI_RUN_STEM


def build_run_paths(output_parent: Path, project_title: str) -> tuple[Path, Path, str]:
    stem = demo_style_stem(project_title)
    output_dir = output_parent if output_parent.name == stem else output_parent / stem
    return output_dir, output_dir / f"{stem}.json", stem
