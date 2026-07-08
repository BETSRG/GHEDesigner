import pickle
import time
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from ghedesigner.district_system import CoupledHorizontalPipe, IsolatedHorizontalPipe
from ghedesigner.enums import CentralLoopType
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Soil


def get_nearest(value, array):
    """Helper function identical to the main system solver for snapping to grid keys."""
    idx = (np.abs(array - value)).argmin()
    return float(array[idx])


def run_horizontal_simulation(config, lib_data):
    print(f"\n--- Starting Simulation: {config['run_name']} ---")

    # --- Setup Time and Boundary Conditions ---
    num_hours = config.get("num_hours", 17520)
    time_array = np.arange(num_hours, dtype=float)
    step_hour = config.get("step_hour", 1)
    mass_flow = config.get("mass_flow", 22)

    # --- Physical Parameters ---
    # Pull base UGT from config to sync Soil with the pipe conditions
    base_ugt = config.get("ugt_avg", 15.0)

    soil = Soil(k=2, rho_cp=2343520, ugt=base_ugt)
    fluid = Fluid(fluid_name="PROPYLENEGLYCOL", percent=20, temperature=15)

    # Dynamically pull pipe dimensions from config
    pipe = Pipe.init_single_u_tube(
        inner_diameter=config.get("inner_diameter", 0.1524),
        outer_diameter=config.get("outer_diameter", 0.1624),
        shank_spacing=0.0,
        roughness=1e-6,
        conductivity=0.4,
        rho_cp=1542000,
    )

    r_pipe = 0.02737  # Example static resistance, update as needed
    beta = r_pipe * (2 * np.pi * soil.k)

    table_single = lib_data["table_single"]
    table_parallel = lib_data["table_parallel"]
    horiz_axes = lib_data["axes"]

    # Snap to nearest axes
    target_d = get_nearest(config["depth"], horiz_axes["depths"])
    target_beta = get_nearest(beta, horiz_axes["betas"])
    target_r = get_nearest(pipe.r_out, horiz_axes["radii"])

    is_coupled = config.get("type", "ISOLATED").upper() == "COUPLED"
    pipes = []

    if not is_coupled:
        # --- ISOLATED SETUP ---
        q_prime_interp = table_single[(target_d, target_beta, target_r)]
        horiz_pipe = IsolatedHorizontalPipe(
            name=f"{config['run_name']}_pipe",
            length=config["length"],
            num_segments=config["segments"],
            pipe=pipe,
            soil=soil,
            fluid=fluid,
            num_timesteps=num_hours,
            time_array=time_array,
            q_prime_interp=q_prime_interp,
            beta=beta,
            ugt_avg=config.get("ugt_avg", 15.0),
            ugt_amp1=config.get("ugt_amp1", 0.0),
            ugt_phase1=config.get("ugt_phase1", 0.0),
            ugt_amp2=config.get("ugt_amp2", 0.0),
            ugt_phase2=config.get("ugt_phase2", 0.0),
            depth=config["depth"],
        )
        horiz_pipe.row_index = 0
        horiz_pipe.matrix_size = 3 * horiz_pipe.num_segments + 1

        # Attach boundary conditions directly to the component
        horiz_pipe.t_in_initial = config.get("t_in_initial", 15.0)
        horiz_pipe.t_in_step = config.get("t_in_step", 35.0)
        horiz_pipe.branch_flow = config.get("mass_flow", 22.0)

        pipes.append(horiz_pipe)
        global_matrix_size = horiz_pipe.matrix_size

    else:
        # --- COUPLED SETUP ---
        target_b = get_nearest(config["spacing"], horiz_axes["spacings"])
        q_prime_even, q_prime_odd = table_parallel[(target_d, target_b, target_beta, target_r)]

        pipe1 = CoupledHorizontalPipe(
            name=f"{config['run_name']}_branch_A",
            length=config["length"],
            num_segments=config["segments"],
            pipe=pipe,
            soil=soil,
            fluid=fluid,
            num_timesteps=num_hours,
            time_array=time_array,
            q_prime_even_interp=q_prime_even,
            q_prime_odd_interp=q_prime_odd,
            beta=beta,
            ugt_avg=config.get("ugt_avg_A", config.get("ugt_avg", 15)),
            ugt_amp1=config.get("ugt_amp1_A", config.get("ugt_amp1", 0.0)),
            ugt_phase1=config.get("ugt_phase1_A", config.get("ugt_phase1", 0.0)),
            ugt_amp2=config.get("ugt_amp2_A", config.get("ugt_amp2", 0.0)),
            ugt_phase2=config.get("ugt_phase2_A", config.get("ugt_phase2", 0.0)),
            depth=config["depth"],
        )
        pipe2 = CoupledHorizontalPipe(
            name=f"{config['run_name']}_branch_B",
            length=config["length"],
            num_segments=config["segments"],
            pipe=pipe,
            soil=soil,
            fluid=fluid,
            num_timesteps=num_hours,
            time_array=time_array,
            q_prime_even_interp=q_prime_even,
            q_prime_odd_interp=q_prime_odd,
            beta=beta,
            ugt_avg=config.get("ugt_avg_B", config.get("ugt_avg", 15)),
            ugt_amp1=config.get("ugt_amp1_B", config.get("ugt_amp1", 0.0)),
            ugt_phase1=config.get("ugt_phase1_B", config.get("ugt_phase1", 0.0)),
            ugt_amp2=config.get("ugt_amp2_B", config.get("ugt_amp2", 0.0)),
            ugt_phase2=config.get("ugt_phase2_B", config.get("ugt_phase2", 0.0)),
            depth=config["depth"],
        )

        # Link them mathematically
        pipe1.coupled_pipe = pipe2
        pipe2.coupled_pipe = pipe1

        # Allocate global matrix positioning
        pipe1.matrix_size = (3 * pipe1.num_segments + 1) * 2
        pipe2.matrix_size = pipe1.matrix_size
        pipe1.row_index = 0
        pipe2.row_index = 3 * pipe1.num_segments + 1

        # Attach independent boundary conditions directly to each branch
        pipe1.t_in_initial = config.get("t_in_initial_A", 15.0)
        pipe1.t_in_step = config.get("t_in_step_A", 35.0)
        pipe1.branch_flow = config.get("mass_flow_A", config.get("mass_flow", 22.0))

        pipe2.t_in_initial = config.get("t_in_initial_B", 15.0)
        pipe2.t_in_step = config.get("t_in_step_B", 10.0)
        pipe2.branch_flow = config.get("mass_flow_B", config.get("mass_flow", 22.0))

        pipes.extend([pipe1, pipe2])
        global_matrix_size = pipe1.matrix_size

    # --- Micro-Solver Loop ---
    start_time = time.perf_counter()
    for t in range(1, num_hours):
        global_rows = []
        global_rhs = []

        # Generate standard matrix rows for all pipes
        for p in pipes:
            rows, rhs = p.generate_matrix(
                _mass_bldg=0.0,
                mass_loop=mass_flow,
                _mass_loop_bldg=0.0,
                mass_flow_pipe=p.branch_flow,
                _mass_loop_ghe=0.0,
                idx_timestep=t,
                configuration=CentralLoopType.ONEPIPE,
                _method=None,
            )
            global_rows.extend(rows)
            global_rhs.extend(rhs)

        # Enforce Boundary Conditions: Pull the specific temperature required for this specific pipe
        for p in pipes:
            p_current_t_in = p.t_in_step if t >= step_hour else p.t_in_initial

            global_rows[p.row_index] = np.zeros(global_matrix_size)
            global_rows[p.row_index][p.row_index] = 1.0
            global_rhs[p.row_index] = p_current_t_in

        # Solve A * X = B
        a_matrix = np.array(global_rows, dtype=float)
        b_vector = np.array(global_rhs, dtype=float)
        x_vector = np.linalg.solve(a_matrix, b_vector)

        # Update History
        for p in pipes:
            p.update_post_solve(x_vector, t)

    end_time = time.perf_counter()
    print(f"Solver finished in {end_time - start_time:.4f} seconds.")

    # --- Export Results ---
    output_columns = {"Time [hr]": time_array[1:]}
    for p in pipes:
        output_columns[f"{p.name}_Inlet [C]"] = p.t_in[1:]
        output_columns[f"{p.name}_Outlet [C]"] = p.t_out_seg[-1, 1:]
        for k in range(p.num_segments):
            output_columns[f"{p.name}_Node{k + 1}_Tmean [C]"] = p.t_mean_seg[k, 1:]
            output_columns[f"{p.name}_Node{k + 1}_Q [W/m]"] = p.q_seg[k, 1:]

    output_data = pd.DataFrame(output_columns).set_index("Time [hr]")

    # Adjust export directory
    base_dir = Path(r"C:\Users\drewm\documents\research")
    output_path = (
        base_dir
        / "GHEDesigner csv results"
        / "horiz pipe component testing"
        / "validation test cases"
        / f"{config['run_name']}.csv"
    )
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True)

    output_data.to_csv(output_path, float_format="%0.4f")
    print(f"Results saved successfully to: {output_path.resolve()}")


def main():
    print("Loading interpolation library...")
    with resources.files("ghedesigner.ghe").joinpath("unified_horizontal_library.pkl").open("rb") as f:
        lib_data = pickle.load(f)  # noqa: S301

    # --- Define Batch Configurations ---
    sim_configs = [
        {
            "run_name": "case1",
            "type": "COUPLED",
            "length": 1.0,
            "segments": 10,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow_A": 22.0,
            "mass_flow_B": 0.0,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 15,
            "ugt_avg": 15,
        },
        {
            "run_name": "case2",
            "type": "COUPLED",
            "length": 1.0,
            "segments": 10,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 22,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
        },
        {
            "run_name": "case3",
            "type": "COUPLED",
            "length": 1.0,
            "segments": 10,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 22,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 13,
            "ugt_avg": 15,
        },
        {
            "run_name": "case4",
            "type": "COUPLED",
            "length": 100.0,
            "segments": 20,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 2.2,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
            "ugt_amp1": 0.0,
            "ugt_phase1": 0.0,
            "ugt_amp2": 0.0,
            "ugt_phase2": 0.0,
        },
        {
            "run_name": "case5",
            "type": "COUPLED",
            "length": 100.0,
            "segments": 20,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 22,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
            "ugt_amp1": 0.0,
            "ugt_phase1": 0.0,
            "ugt_amp2": 0.0,
            "ugt_phase2": 0.0,
        },
        {
            "run_name": "case6",
            "type": "COUPLED",
            "length": 500.0,
            "segments": 20,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 2.2,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
            "ugt_amp1": 0.0,
            "ugt_phase1": 0.0,
            "ugt_amp2": 0.0,
            "ugt_phase2": 0.0,
        },
        {
            "run_name": "case7",
            "type": "COUPLED",
            "length": 500.0,
            "segments": 20,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 22,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
            "ugt_amp1": 0.0,
            "ugt_phase1": 0.0,
            "ugt_amp2": 0.0,
            "ugt_phase2": 0.0,
        },
        {
            "run_name": "case8",
            "type": "COUPLED",
            "length": 100.0,
            "segments": 20,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 2.2,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
            "ugt_amp1": 10.0,
            "ugt_phase1": 30.0,
            "ugt_amp2": 2.0,
            "ugt_phase2": 15.0,
        },
        {
            "run_name": "case9",
            "type": "COUPLED",
            "length": 100.0,
            "segments": 20,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 22,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
            "ugt_amp1": 10.0,
            "ugt_phase1": 30.0,
            "ugt_amp2": 2.0,
            "ugt_phase2": 15.0,
        },
        {
            "run_name": "case10",
            "type": "COUPLED",
            "length": 500.0,
            "segments": 20,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 2.2,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
            "ugt_amp1": 10.0,
            "ugt_phase1": 30.0,
            "ugt_amp2": 2.0,
            "ugt_phase2": 15.0,
        },
        {
            "run_name": "case11",
            "type": "COUPLED",
            "length": 500.0,
            "segments": 20,
            "depth": 1.5,
            "spacing": 0.5,
            "inner_diameter": 0.1524,
            "outer_diameter": 0.1624,
            "mass_flow": 22,
            "t_in_initial_A": 15,
            "t_in_step_A": 25,
            "t_in_initial_B": 15,
            "t_in_step_B": 5,
            "ugt_avg": 15,
            "ugt_amp1": 10.0,
            "ugt_phase1": 30.0,
            "ugt_amp2": 2.0,
            "ugt_phase2": 15.0,
        },
    ]

    for config in sim_configs:
        run_horizontal_simulation(config, lib_data)


if __name__ == "__main__":
    main()
