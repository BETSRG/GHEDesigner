import multiprocessing  # <-- The key to speed
import pickle
import time

import numpy as np
from scipy import interpolate

from ghedesigner.horizontal_pipe_heat_exchange import ParallelPipeSystem


# --- 1. Setup Classes ---
class MockPipe:
    def __init__(self, r_out, k):
        self.r_out = r_out
        self.k = k


class MockSoil:
    def __init__(self, k, rhocp):
        self.k = k
        self.rhocp = rhocp


# --- 2. Worker Function (Runs on each core) ---
def worker_task(args):
    """
    This function runs on a separate CPU core.
    It builds ONE interpolator for a specific (D, B, beta) combination.
    """
    d, b, beta = args

    # Re-create the system objects inside the worker process
    # (This is safer for multiprocessing than passing complex objects)
    pipe = MockPipe(r_out=0.1, k=0.4)
    soil = MockSoil(k=1.5, rhocp=2.3e6)

    system = ParallelPipeSystem(x_coord=b, y_coord=d, pipe=pipe, soil=soil)

    # Parameters for the time simulation
    years = 30
    hours_in_year = 365.25 * 24
    epsilon_x_case = 1.0  # Even case only for this table

    # --- The Calculation ---
    tau_years_precalc = np.logspace(-4, np.log10(years), 100)
    tau_seconds_precalc = tau_years_precalc * hours_in_year * 3600

    q_prime_at_zero = 1.0 / beta

    # Calculate the 100 points (The slow part)
    q_prime_values = [system.heat_transfer(t, epsilon_x_case, beta) for t in tau_seconds_precalc]

    final_tau_seconds = np.insert(tau_seconds_precalc, 0, 0.0)
    final_q_prime_values = np.insert(q_prime_values, 0, q_prime_at_zero)

    interpolator = interpolate.interp1d(final_tau_seconds, final_q_prime_values, kind='cubic', fill_value="extrapolate")

    return (d, b, beta), interpolator


# --- 3. Main Parallel Execution ---
def main():
    print("--- Starting Massive Parallel Table Build (16 Cores) ---")
    total_start_time = time.perf_counter()

    # --- Define the 8x8x9 Grid (576 points) ---
    # Using logspace covers a wide range effectively
    depths_to_run = np.logspace(np.log10(0.5), np.log10(50.0), 8)
    spacings_to_run = np.logspace(np.log10(0.2), np.log10(10.0), 8)
    betas_to_run = np.logspace(np.log10(0.1), np.log10(10.0), 9)

    # Create a list of all 576 jobs
    all_jobs = []
    for d in depths_to_run:
        for b in spacings_to_run:
            for beta in betas_to_run:
                all_jobs.append((d, b, beta))

    print(f"Total interpolators to build: {len(all_jobs)}")
    print("Starting parallel pool...")

    interpolator_table = {}
    completed = 0

    # --- Start the Parallel Pool ---
    # processes=16 uses all your cores. Change this number if you want to leave some free.
    with multiprocessing.Pool(processes=16) as pool:
        # imap_unordered is efficient and lets us track progress as jobs finish
        for result in pool.imap_unordered(worker_task, all_jobs):
            key, interpolator = result
            interpolator_table[key] = interpolator

            completed += 1
            if completed % 10 == 0:
                elapsed = time.perf_counter() - total_start_time
                rate = completed / (elapsed / 60)
                print(
                    f"  Progress: {completed}/{len(all_jobs)} ({completed/len(all_jobs)*100:.1f}%) - Rate: {rate:.1f} jobs/min"
                )

    print("\n--- All interpolators built! ---")

    # --- 4. Save the Table ---
    data_to_save = {
        'axes': (list(depths_to_run), list(spacings_to_run), list(betas_to_run)),
        'table': interpolator_table,
    }

    filename = "interpolator_table_huge.pkl"
    print(f"Saving data to file: {filename}")

    with open(filename, "wb") as f:
        pickle.dump(data_to_save, f)

    print("Save complete!")

    total_end_time = time.perf_counter()
    print(f"Total time taken: {(total_end_time - total_start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
