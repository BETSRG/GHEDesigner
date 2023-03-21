import os
import re
import sys

from pathlib import Path


def update_version(new_version_str: str):
    root_dir = Path(__file__).parent.parent

    # update __init__.py version
    init_path = root_dir / "ghedesigner" / "__init__.py"
    init_contents = init_path.read_text()
    init_contents = re.sub(r"VERSION = .+\n", f"VERSION = \"{new_version_str}\"\n", init_contents)
    with open(init_path, 'w') as f:
        f.write(init_contents)

    # update demo files version
    demos_dir = root_dir / "demos"
    for _, _, files in os.walk(demos_dir):
        for f in files:
            demo_file_path = demos_dir / f
            demo_contents = demo_file_path.read_text()
            demo_contents = re.sub(r"\"version\": \".+\",\n", f"\"version\": \"{new_version_str}\",\n", demo_contents)
            with open(demo_file_path, 'w') as f:
                f.write(demo_contents)


if __name__ == "__main__":
    update_version(sys.argv[1])
