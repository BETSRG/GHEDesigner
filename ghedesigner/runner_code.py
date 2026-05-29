from pathlib import Path
from time import time

from ghedesigner.district_system import GHEHPSystem


def main():

    input_file_path = Path("..\\GHEDesigner\\demos\\beier_trt_validation.json")
    output_file_path = Path("..\\Documents\\GHEDesigner csv results\\simulation_results_trt_9discstudy.csv")

    start_time = time()
    system = GHEHPSystem(input_file_path)
    system.solve_system()
    end_time = time()

    system.create_output(output_file_path)

    print(f"Total time taken: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
