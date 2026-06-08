import multiprocessing
import traceback
from pathlib import Path
from time import time

from ghedesigner.district_system import GHEHPSystem


def worker_simulation(job_args):
    """
    This function runs entirely inside a separate CPU process.
    It only accepts simple strings/paths to avoid SciPy pickling errors.
    """
    input_file_path, output_file_path = job_args
    start_time = time()

    try:
        system = GHEHPSystem(input_file_path)

        system.solve_system()

        system.create_output(output_file_path)

        end_time = time()
        duration = end_time - start_time

        return (input_file_path.name, True, duration, None)

    except Exception:  # noqa: BLE001
        error_trace = traceback.format_exc()
        return (input_file_path.name, False, 0.0, error_trace)


def main():
    print("--- Starting GHE District Batch Simulation ---")
    start_total = time()

    input_dir = Path("..\\GHEDesigner\\demos")
    output_dir = Path("..\\Documents\\GHEDesigner csv results\\TRT")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_files = [
        "beier_trt_60_ft_1_segment.json",
        "beier_trt_60_ft_10_segments.json",
        "beier_trt_600_ft_1_segment.json",
        "beier_trt_600_ft_2_segments.json",
        "beier_trt_600_ft_3_segments.json",
        "beier_trt_600_ft_4_segments.json",
        "beier_trt_600_ft_5_segments.json",
        "beier_trt_600_ft_7_segments.json",
        "beier_trt_600_ft_9_segments.json",
        "beier_trt_600_ft_10_segments.json",
    ]

    jobs = []
    for file_name in target_files:
        in_path = input_dir / file_name
        out_path = output_dir / file_name.replace(".json", "_no_cap.csv")
        jobs.append((in_path, out_path))

    print(f"Queued {len(jobs)} simulations.")

    total_threads = multiprocessing.cpu_count()
    num_workers = max(1, total_threads - 3)
    print(f"Spinning up {num_workers} parallel workers...")

    success_count = 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(worker_simulation, jobs):
            sim_name, success, duration, err = result

            if success:
                print(f"[SUCCESS] {sim_name} finished in {duration:.2f}s")
                success_count += 1
            else:
                print(f"[FAILED]  {sim_name} encountered an error:\n{err}")

    total_time = time() - start_total
    print("\n--- Batch Run Complete ---")
    print(f"Successfully ran {success_count}/{len(jobs)} simulations.")
    print(f"Total time elapsed: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
