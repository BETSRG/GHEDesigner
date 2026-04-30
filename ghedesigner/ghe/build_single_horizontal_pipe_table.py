import multiprocessing
import pickle
import time
import numpy as np
from scipy import interpolate

# Ensure this import matches your file structure
from ghedesigner.ghe.horizontal_pipe_heat_exchange import SinglePipeWithSurfaceSystem

class MockPipe:
    def __init__(self, r_out, k):
        self.r_out = r_out
        self.k = k

class MockSoil:
    def __init__(self, k, rhocp):
        self.k = k
        self.rhoCp = rhocp

def worker_task(args):
    d, beta, r = args

    pipe = MockPipe(r_out=r, k=0.4)
    soil = MockSoil(k=1.5, rhocp=2.3e6) 
    system = SinglePipeWithSurfaceSystem(y_coord=d, pipe=pipe, soil=soil)

    years = 100
    t_min_hours = 0.1
    tau_seconds = np.logspace(np.log10(t_min_hours * 3600), np.log10(years * 365.25 * 24 * 3600), 80)
    final_tau_seconds = np.insert(tau_seconds, 0, 0.0)

    # Calculate Dimensionless Heat Flux for the single pipe with surface
    q_raw = np.array([system.heat_transfer(t, beta) for t in tau_seconds])

    q_start_raw = 1.0 / beta
    final_q = np.insert(q_raw, 0, q_start_raw)

    interp_q = interpolate.interp1d(final_tau_seconds, final_q, kind="cubic", fill_value="extrapolate")

    return (d, beta, r), interp_q


def main():
    print("--- Building Single Pipe w/ Surface Interpolation Library (3D Grid) ---")
    t_start = time.perf_counter()

    depths = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    betas = np.array([0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]) 
    radii = np.array([0.005, 0.01, 0.015, 0.026, 0.040, 0.06, 0.08, 0.10, 0.15, 0.20])

    all_jobs = [(d, beta, r) for d in depths for beta in betas for r in radii]
    total_jobs = len(all_jobs)
    print(f"Generating {total_jobs} curves...")

    table = {}
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_task, all_jobs), 1):
            key, i_q = result
            table[key] = i_q

            if i % 5 == 0 or i == total_jobs:
                elapsed_time = time.perf_counter() - t_start
                avg_time_per_job = elapsed_time / i
                remaining_jobs = total_jobs - i
                eta_seconds = remaining_jobs * avg_time_per_job

                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
                print(f"  Progress: {i}/{total_jobs} | Elapsed: {elapsed_str} | Remaining: ~{eta_str} ({avg_time_per_job:.2f}s/j)", end="\r", flush=True)

    print("\nPackaging data...")
    data = {
        "axes": {"depths": depths, "betas": betas, "radii": radii},
        "table": table,
    }

    output_filename = "single_pipe_surface_library_3D.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(data, f)

    print(f"Done. Saved to '{output_filename}' in {time.perf_counter() - t_start:.2f}s")

if __name__ == "__main__":
    main()