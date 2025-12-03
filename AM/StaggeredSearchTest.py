import json
import tempfile
import csv

from time import perf_counter
from copy import deepcopy as dc
from pathlib import Path
from ghedesigner.main import run_manager_from_cli as run

def ssearch(input_json: Path, output_base: Path, multipliers: list[float]):

    # 1 - Open base JSON file
    with open(input_json, "r") as f:
        base_cfg = json.load(f)

    # pulling nested loads inside ghe
    ghe_name = next(iter(base_cfg["ground-heat-exchanger"]))
    original_loads = base_cfg["ground-heat-exchanger"][ghe_name]["loads"]

    results = []


    for m in multipliers:
    # 2 - Deepcopy and scale loads array
        cfg = dc(base_cfg)
        cfg["ground-heat-exchanger"][ghe_name]["loads"] = [load *m for load in original_loads] # = scaled

        # Force ghedesigner to do all runs in one go for each multiplier
        cfg["simulation-control"] = {
        "thermal-sizing-run": True,
            "hourly-run": True,
            "hydraulic-run": True,
            "simulation-months": base_cfg.get("simulation-control", {}).get("simulation-months", 240)
        }

    # 3 - Write scaled JSON to temp file
        with tempfile.NamedTemporaryFile(suffix = ".json", mode = "w", delete = False) as tmp:
            json.dump(cfg, tmp)
            tmp_path = Path(tmp.name)

    # 4 - Make dedicated sub_folder for multipliers:
        run_dir = output_base / f"multipliers_{m}"
        run_dir.mkdir(parents = True, exist_ok = True)

    # 5 - Time cli run
        t0 = perf_counter()
    # Bypass SystemExit and ValueErrors
        try:
            run(args=[str(tmp_path), str(run_dir)])
        except SystemExit as e:
            if e.code != 0:
                print(f"GHEDesigner exited unexpectedly with code {e.code} on multiplier {m}")
        except ValueError:
            print(f"ValueError at m={m}")
        t1 = perf_counter()
        elapsed = t1 - t0
        print(f"[m = {m}] : finished in {elapsed:.2f} s")
        # Adding best key print
        summary_path = next(run_dir.glob("SimulationSummary.json"), None)
        nbh = None
        if summary_path:
            with summary_path.open("r") as sf:
                summary = json.load(sf)

            # Get NBH
            nbh = summary["ghe_system"]["number_of_boreholes"]

            # 1) get best design
            best_key = summary["ghe_system"]["field_specifier"]

            # 2( find excess temp in ds search log table
            dlog = summary["design_selection_search_log"]
            excess = None
            for field, excess_temp, *_ in dlog["data"]:
                if field == best_key:
                    excess = excess_temp
                    break
            # 3) print
            if excess is not None:
                print(f"best key for m={m:.1f} is: {best_key} at {excess:.2f}°C")
            else:
                print(f"best key for m={m:.1f} is: {best_key} (excess temp not found)")
        else:
            print(f" no SimulationSummary.json found in {run_dir}")


        results.append((m,elapsed, nbh, t1))

    # 6 - Dump a csv of all time values:
    csv_path = output_base / "staggered_results.csv"
    with open(csv_path, "w", newline = "") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["multiplier", "nbh", "time_s"])
        for m, dt, nbh, ts in results:
            writer.writerow([m, nbh, f"{dt:.2f}"])
    print(f"\nFinished. Results: {csv_path}")


def main():

    input_file = Path(r"C:\\Users\\Student\\PycharmProjects\\GHEDesigner\\demos\\find_design_tilted_line_u_tube.json")
    output_dir = Path(r"C:\\Users\\Student\\Desktop\\StaggeredOnlySearch")

    multipliers = [0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]


    print("Beginning search...")
    start = perf_counter()
    ssearch(input_file, output_dir, multipliers)
    end = perf_counter()
    print(f"Total time: {(end - start):.2f} seconds")

if __name__ == '__main__':
    main()