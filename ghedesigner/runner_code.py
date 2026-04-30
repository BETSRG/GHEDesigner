from pathlib import Path
from ghedesigner.district_system import GHEHPSystem
from time import time


def main():

    input_file_path = Path("..\\GHEDesigner\\demos\\simulate_1_pipe_3_ghe_6_bldg_district_HOURLY.json")
    output_file_path = Path("..\\GHEDesigner\\ghedesigner\\ghe\\nbast_results\\simulation_results.csv")

    start_time = time()
    system = GHEHPSystem(input_file_path)
    system.solve_system()
    end_time = time()

    system.create_output(output_file_path)

    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
