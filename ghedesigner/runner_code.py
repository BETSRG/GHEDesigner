from pathlib import Path
from time import time

from ghedesigner.district_system import GHEHPSystem


def main():

    input_file_path = Path("..\\GHEDesigner\\demos\\simulate_1_pipe_3_ghe_6_bldg_district_HOURLY_horizontal.json")
    output_file_path = Path(
        "..\\Documents\\GHEDesigner csv results\\simulation_results_horizontal_3_segments_test13_noUGT_split.csv"
    )

    start_time = time()
    system = GHEHPSystem(input_file_path)
    system.solve_system()
    end_time = time()

    system.create_output(output_file_path)

    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
