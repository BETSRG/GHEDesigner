#!/usr/bin/env python
import csv
import json
import sys
from pathlib import Path


def update_results(results_dir: Path, expected_results_path: Path):
    d_expected = {}

    with open(expected_results_path, encoding="utf-8") as f:
        d_current = json.load(f)

    for p in results_dir.iterdir():
        this_dir = results_dir / p
        f_paths = [this_dir / "SimulationSummary.json", this_dir / "Search_Summary.csv"]
        key = this_dir.stem
        if f_paths[0].is_file():
            f_path = f_paths[0]
            d = json.loads(f_path.read_text())
            if "ghe_system" in d:
                d_expected[key] = {
                    "active_borehole_length": d["ghe_system"]["active_borehole_length"]["value"],
                    "number_of_boreholes": d["ghe_system"]["number_of_boreholes"],
                }
        elif f_paths[1].is_file():
            f_path = f_paths[1]
            with open(f_path) as input_file:
                csv_reader = list(csv.reader(input_file))
                last_row = csv_reader[-1]
                d_expected[key] = {
                    "active_borehole_length": float(last_row[4]),
                    "number_of_boreholes": int(last_row[3]),
                }

    d_current.update(d_expected)

    with open(expected_results_path, "w", encoding="utf-8") as f:
        json.dump(d_current, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    # python update_demo_results.py path_to_results_dir path_to_expected_results_file
    update_results(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
