from pathlib import Path
from time import time

from ghedesigner.district_system import GHEHPSystem


def main():
    input_file_path = Path("../demos/simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_with_CT.json")
    output_file_path = Path("results\\simulation_results.csv")

    start_time = time()
    system = GHEHPSystem(input_file_path)
    system.solve_system()
    end_time = time()

    system.create_output(output_file_path)

    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
