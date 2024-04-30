import json
import sys

from pathlib import Path


def update_results(results_dir: Path, expected_results_path: Path):

    d_expected = {}

    for p in results_dir.iterdir():
        this_dir = results_dir / p
        f_path = this_dir / "SimulationSummary.json"
        d = json.loads(f_path.read_text())
        key = this_dir.stem
        d_expected[key] = {
            "active_borehole_length": d["ghe_system"]["active_borehole_length"]["value"],
            "number_of_boreholes": d["ghe_system"]["number_of_boreholes"]
        }

    with open(expected_results_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(d_expected, sort_keys=True, indent=2, separators=(',', ': ')))


if __name__ == "__main__":
    # python update_demo_results.py path_to_results_dir path_to_expected_results_file
    update_results(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
