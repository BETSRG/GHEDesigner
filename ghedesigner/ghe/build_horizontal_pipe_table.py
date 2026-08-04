import json
import multiprocessing
import time
from pathlib import Path

import numpy as np

from ghedesigner.constants import HORZ_LIBRARY_FILENAME
from ghedesigner.ghe.horizontal_pipe_heat_exchange import ParallelPipeSystem, SinglePipeWithSurfaceSystem
from ghedesigner.utilities import float_tuple_to_string

LIBRARY_DECIMALS_OF_PRECISION = 10


class MockPipe:
    def __init__(self, r_out, k):
        self.r_out = r_out
        self.k = k


class MockSoil:
    def __init__(self, k, rhocp):
        self.k = k
        self.rho_cp = rhocp


def worker_single(args):
    d, beta, r, k_s = args
    print(f"    -> Starting SINGLE job: Depth={d}, Beta={beta:.3f}, r={r:.4f}, ks={k_s:.2f}", flush=True)
    pipe = MockPipe(r_out=r, k=0.4)
    soil = MockSoil(k=k_s, rhocp=2.3e6)
    system = SinglePipeWithSurfaceSystem(y_coord=d, pipe=pipe, soil=soil)

    # Calculate characteristic time tp
    alpha_s = soil.k / soil.rho_cp
    t_p = (pipe.r_out**2) / alpha_s

    years = 100
    t_min_hours = 0.1
    tau_seconds = np.logspace(np.log10(t_min_hours * 3600), np.log10(years * 365.25 * 24 * 3600), 80)

    # Calculate heat transfer using dimensional seconds for the Claesson equations
    q_raw = np.array([system.heat_transfer(t, beta) for t in tau_seconds])
    q_start_raw = 1.0 / beta
    final_q = np.insert(q_raw, 0, q_start_raw)

    # Convert the seconds array to true dimensionless time (tau) for the interpolator
    true_tau = tau_seconds / t_p
    final_true_tau = np.insert(true_tau, 0, 0.0)

    q_data = {
        "x": np.round(final_true_tau, decimals=LIBRARY_DECIMALS_OF_PRECISION).tolist(),
        "y": np.round(final_q, decimals=LIBRARY_DECIMALS_OF_PRECISION).tolist(),
    }

    return "single", float_tuple_to_string((d, beta, r, k_s)), q_data


def worker_parallel(args):
    d, b, beta, r, k_s = args
    print(f"    -> Starting PARALLEL job: Depth={d}, Spacing={b}, Beta={beta:.3f}, r={r:.4f}, ks={k_s:.2f}", flush=True)
    pipe = MockPipe(r_out=r, k=0.4)
    soil = MockSoil(k=k_s, rhocp=2.3e6)
    system = ParallelPipeSystem(x_coord=b, y_coord=d, pipe=pipe, soil=soil)

    # Calculate characteristic time tp
    alpha_s = soil.k / soil.rho_cp
    t_p = (pipe.r_out**2) / alpha_s

    years = 100
    t_min_hours = 0.1
    tau_seconds = np.logspace(np.log10(t_min_hours * 3600), np.log10(years * 365.25 * 24 * 3600), 80)

    # Calculate both Even and Odd cases using dimensional seconds
    q_even_raw = np.array([system.heat_transfer(t, 1.0, beta) for t in tau_seconds])
    q_odd_raw = np.array([system.heat_transfer(t, -1.0, beta) for t in tau_seconds])

    q_start_raw = 1.0 / beta
    final_q_even = np.insert(q_even_raw, 0, q_start_raw)
    final_q_odd = np.insert(q_odd_raw, 0, q_start_raw)

    # Convert the seconds array to true dimensionless time (tau) for the interpolators
    true_tau = tau_seconds / t_p
    final_true_tau = np.insert(true_tau, 0, 0.0)

    q_prime_data = {
        "x1": np.round(final_true_tau, decimals=LIBRARY_DECIMALS_OF_PRECISION).tolist(),
        "y1": np.round(final_q_even, decimals=LIBRARY_DECIMALS_OF_PRECISION).tolist(),
        "x2": np.round(final_true_tau, decimals=LIBRARY_DECIMALS_OF_PRECISION).tolist(),
        "y2": np.round(final_q_odd, decimals=LIBRARY_DECIMALS_OF_PRECISION).tolist(),
    }

    return "parallel", float_tuple_to_string((d, b, beta, r, k_s)), q_prime_data


def worker_dispatcher(job):
    job_type = job[0]
    if job_type == "single":
        return worker_single(job[1:])
    elif job_type == "parallel":
        return worker_parallel(job[1:])


def main():
    print("--- Building/Updating Unified Interpolation Library ---")
    t_start = time.perf_counter()
    output_path = Path(__file__).with_name(HORZ_LIBRARY_FILENAME)

    # Define the parameters
    depths = np.array([1.0, 1.5, 5.0, 15.0])
    spacings = np.array([0.25, 0.5, 1.0])
    betas = np.array([0.344, 12.0])  # add 12.0
    soil_ks = np.array([1.0, 1.5, 2.0, 2.5])

    # Input as nominal diameters, then mathematically convert to radii for the solver grid
    diameters = np.array([0.0762, 0.1016, 0.1524, 0.3])  # add 0.3
    radii = diameters / 2.0

    table_single = {}
    table_parallel = {}
    existing_depths, existing_spacings, existing_betas = set(), set(), set()
    existing_radii, existing_soil_ks = set(), set()

    new_single_jobs = []
    new_parallel_jobs = []

    # Parallel format: (depth, spacing, beta, radius, soil_k)
    force_recalc_parallel = [
        # (15.0, 0.053, 4.783, 0.0167 / 2.0, 1.5),
    ]

    # Single format: (depth, beta, radius, soil_k)
    force_recalc_single = [
        # (1.0, 12.0, 0.3 / 2.0, 1.5),
    ]

    # 1. Load file if it exists
    if output_path.exists():
        print(f"Found existing library: '{output_path}'. Loading...")
        with output_path.open("rb") as f:
            existing_data = json.load(f)

        table_single = existing_data.get("table_single", {})
        table_parallel = existing_data.get("table_parallel", {})
        axes = existing_data.get("axes", {})

        existing_depths = set(axes.get("depths", []))
        existing_spacings = set(axes.get("spacings", []))
        existing_betas = set(axes.get("betas", []))
        existing_radii = set(axes.get("radii", []))
        existing_soil_ks = set(axes.get("soil_ks", []))

    # 2. Process forced cases
    for key in force_recalc_parallel:
        json_key = float_tuple_to_string(key)
        if json_key in table_parallel:
            del table_parallel[json_key]
            print(f"Forcing recalculation for parallel case: {key}")
        else:
            print(f"Adding brand new parallel case: {key}")

        new_parallel_jobs.append(("parallel", *key))

        existing_depths.add(key[0])
        existing_spacings.add(key[1])
        existing_betas.add(key[2])
        existing_radii.add(key[3])
        existing_soil_ks.add(key[4])

    for key in force_recalc_single:
        json_key = float_tuple_to_string(key)
        if json_key in table_single:
            del table_single[json_key]
            print(f"Forcing recalculation for single case: {key}")
        else:
            print(f"Adding brand new single case: {key}")

        new_single_jobs.append(("single", *key))

        existing_depths.add(key[0])
        existing_betas.add(key[1])
        existing_radii.add(key[2])
        existing_soil_ks.add(key[3])

    # 3. Process standard grid
    new_single_jobs.extend(
        [
            ("single", d, beta, r, k_s)
            for d in depths
            for beta in betas
            for r in radii
            for k_s in soil_ks
            if (d, beta, r, k_s) not in table_single and (d, beta, r, k_s) not in force_recalc_single
        ]
    )

    new_parallel_jobs.extend(
        [
            ("parallel", d, b, beta, r, k_s)
            for d in depths
            for b in spacings
            for beta in betas
            for r in radii
            for k_s in soil_ks
            if (d, b, beta, r, k_s) not in table_parallel and (d, b, beta, r, k_s) not in force_recalc_parallel
        ]
    )

    all_new_jobs = new_single_jobs + new_parallel_jobs
    total_new_jobs = len(all_new_jobs)

    if total_new_jobs == 0:
        print("\nAll requested parameter combinations already exist in the library. Nothing to compute.")
        return

    print(f"\nIdentified {len(new_single_jobs)} new single-pipe curves to compute.")
    print(f"Identified {len(new_parallel_jobs)} new parallel-pipe curve pairs to compute.")
    print(f"Total new jobs: {total_new_jobs}")

    # 4. Process only the new jobs
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_dispatcher, all_new_jobs), 1):
            job_type, key, payload = result

            if job_type == "single":
                table_single[key] = payload
            elif job_type == "parallel":
                table_parallel[key] = payload

            if i % 10 == 0 or i == total_new_jobs:
                elapsed_time = time.perf_counter() - t_start
                avg_time_per_job = elapsed_time / i
                remaining_jobs = total_new_jobs - i
                eta_seconds = remaining_jobs * avg_time_per_job

                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print(
                    f"  Progress: {i}/{total_new_jobs} | Elapsed: {elapsed_str} |"
                    f" Remaining: ~{eta_str} ({avg_time_per_job:.2f}s/j)    ",
                    end="\r",
                    flush=True,
                )

    print("\n\nPackaging updated data...")

    # Merge old and new axes
    final_depths = np.array(sorted(existing_depths.union(depths)))
    final_spacings = np.array(sorted(existing_spacings.union(spacings)))
    final_betas = np.array(sorted(existing_betas.union(betas)))
    final_radii = np.array(sorted(existing_radii.union(radii)))
    final_soil_ks = np.array(sorted(existing_soil_ks.union(soil_ks)))

    data = {
        "axes": {
            "depths": final_depths.tolist(),
            "spacings": final_spacings.tolist(),
            "betas": final_betas.tolist(),
            "radii": final_radii.tolist(),
            "soil_ks": final_soil_ks.tolist(),
        },
        "table_single": table_single,
        "table_parallel": table_parallel,
    }

    with output_path.open("wb") as f:
        json.dump(data, f)

    print(f"Done. Saved updated library to '{output_path}' in {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
