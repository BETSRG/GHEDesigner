import multiprocessing
import pickle
import time

import numpy as np
from scipy import interpolate

# Ensure this import matches your file structure
from ghedesigner.ghe.horizontal_pipe_heat_exchange import ParallelPipeSystem, SinglePipeWithSurfaceSystem


class MockPipe:
    def __init__(self, r_out, k):
        self.r_out = r_out
        self.k = k


class MockSoil:
    def __init__(self, k, rhocp):
        self.k = k
        self.rho_cp = rhocp


def worker_single(args):
    d, beta, r = args
    print(f"    -> Starting SINGLE job: Depth={d}, Beta={beta:.3f}, r={r:.3f}", flush=True)
    pipe = MockPipe(r_out=r, k=0.4)
    soil = MockSoil(k=1.5, rhocp=2.3e6)
    system = SinglePipeWithSurfaceSystem(y_coord=d, pipe=pipe, soil=soil)

    years = 100
    t_min_hours = 0.1
    tau_seconds = np.logspace(np.log10(t_min_hours * 3600), np.log10(years * 365.25 * 24 * 3600), 80)
    final_tau_seconds = np.insert(tau_seconds, 0, 0.0)

    q_raw = np.array([system.heat_transfer(t, beta) for t in tau_seconds])
    q_start_raw = 1.0 / beta
    final_q = np.insert(q_raw, 0, q_start_raw)

    interp_q = interpolate.interp1d(final_tau_seconds, final_q, kind="cubic", fill_value="extrapolate")

    return ("single", (d, beta, r), interp_q)


def worker_parallel(args):
    d, b, beta, r = args
    print(f"    -> Starting PARALLEL job: Depth={d}, Spacing={b}, Beta={beta:.3f}, r={r:.3f}", flush=True)
    pipe = MockPipe(r_out=r, k=0.4)
    soil = MockSoil(k=1.5, rhocp=2.3e6)
    system = ParallelPipeSystem(x_coord=b, y_coord=d, pipe=pipe, soil=soil)

    years = 100
    t_min_hours = 0.1
    tau_seconds = np.logspace(np.log10(t_min_hours * 3600), np.log10(years * 365.25 * 24 * 3600), 80)
    final_tau_seconds = np.insert(tau_seconds, 0, 0.0)

    # Calculate both Even and Odd cases
    q_even_raw = np.array([system.heat_transfer(t, 1.0, beta) for t in tau_seconds])
    q_odd_raw = np.array([system.heat_transfer(t, -1.0, beta) for t in tau_seconds])

    q_start_raw = 1.0 / beta
    final_q_even = np.insert(q_even_raw, 0, q_start_raw)
    final_q_odd = np.insert(q_odd_raw, 0, q_start_raw)

    interp_even = interpolate.interp1d(final_tau_seconds, final_q_even, kind="cubic", fill_value="extrapolate")
    interp_odd = interpolate.interp1d(final_tau_seconds, final_q_odd, kind="cubic", fill_value="extrapolate")

    return ("parallel", (d, b, beta, r), (interp_even, interp_odd))


def worker_dispatcher(job):
    job_type = job[0]
    if job_type == "single":
        return worker_single(job[1:])
    elif job_type == "parallel":
        return worker_parallel(job[1:])


def main():
    print("--- Building Unified Interpolation Library (Single + Parallel) ---")
    t_start = time.perf_counter()

    # tiny array of values for testing quickly
    depths = np.array([1.5])
    spacings = np.array([0.5, 1.0, 5.0])
    betas = np.array([0.008])
    radii = np.array([0.02108, 0.1016, 0.1524, 0.2032])

    # Build the combined job list
    single_jobs = [("single", d, beta, r) for d in depths for beta in betas for r in radii]
    parallel_jobs = [("parallel", d, b, beta, r) for d in depths for b in spacings for beta in betas for r in radii]

    all_jobs = single_jobs + parallel_jobs
    total_jobs = len(all_jobs)

    print(f"Generating {len(single_jobs)} single-pipe curves...")
    print(f"Generating {len(parallel_jobs)} parallel-pipe curve pairs...")
    print(f"Total jobs: {total_jobs}")

    table_single = {}
    table_parallel = {}

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_dispatcher, all_jobs), 1):
            job_type, key, payload = result

            if job_type == "single":
                table_single[key] = payload
            elif job_type == "parallel":
                table_parallel[key] = payload

            if i % 10 == 0 or i == total_jobs:
                elapsed_time = time.perf_counter() - t_start
                avg_time_per_job = elapsed_time / i
                remaining_jobs = total_jobs - i
                eta_seconds = remaining_jobs * avg_time_per_job

                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print(
                    f"  Progress: {i}/{total_jobs} | Elapsed: {elapsed_str} |"
                    f" Remaining: ~{eta_str} ({avg_time_per_job:.2f}s/j)    ",
                    end="\r",
                    flush=True,
                )

    print("\nPackaging data...")
    data = {
        "axes": {"depths": depths, "spacings": spacings, "betas": betas, "radii": radii},
        "table_single": table_single,
        "table_parallel": table_parallel,
    }

    output_filename = "unified_horizontal_library.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Done. Saved to '{output_filename}' in {time.perf_counter() - t_start:.2f}s")


if __name__ == "__main__":
    main()
