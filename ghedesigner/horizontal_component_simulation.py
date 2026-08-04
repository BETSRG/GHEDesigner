import argparse
import itertools
import json
import multiprocessing
import time
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

from ghedesigner.constants import HORZ_LIBRARY_FILENAME
from ghedesigner.district_system import CoupledHorizontalPipe, IsolatedHorizontalPipe, timestep_params_generator
from ghedesigner.enums import CentralLoopType
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.media import Fluid, Soil
from ghedesigner.utilities import float_tuple_to_string

# Global variable to hold the library data for each worker process
global_lib_data: dict | None = None


def init_worker():
    """Initializes the worker process by loading the interpolation library once into memory."""
    global global_lib_data  # noqa: PLW0603
    with resources.files("ghedesigner.ghe").joinpath(HORZ_LIBRARY_FILENAME).open("r", encoding="utf-8") as f:
        global_lib_data = json.load(f)


def get_nearest(value, array):
    """Helper function identical to the main system solver for snapping to grid keys."""
    values = np.asarray(array, dtype=float)
    idx = np.abs(values - value).argmin()
    return float(values[idx])


def run_horizontal_simulation(config):
    """Runs a single simulation case using the globally loaded library."""
    try:
        lib_data = global_lib_data
        if lib_data is None:
            raise RuntimeError("Horizontal interpolation library has not been initialized.")

        # --- Setup Time and Boundary Conditions ---
        num_hours = config.get("num_hours", 8760)
        steps_per_hour = config.get("steps_per_hour", 1)
        total_steps = int(num_hours * steps_per_hour)

        # Each simulated interval requires both a starting and ending time boundary.
        time_array = np.linspace(0.0, num_hours, total_steps + 1)
        num_timesteps = len(time_array)
        time_step_params = timestep_params_generator(time_array)

        step_hour = config.get("step_hour", 1)
        mass_flow = config.get("mass_flow", 22)

        load_method = config.get("load_method", "hourly")

        # --- Physical Parameters ---
        base_ugt = config.get("ugt_avg", 15.0)
        soil_k = config.get("soil_k", 2.0)

        soil = Soil(k=soil_k, rho_cp=2343520, ugt=base_ugt)
        fluid = Fluid(fluid_name="WATER", percent=0, temperature=70)

        pipe = Pipe.init_single_u_tube(
            inner_diameter=config.get("inner_diameter", 0.1524),
            outer_diameter=config.get("outer_diameter", 0.1624),
            shank_spacing=0.0,
            roughness=1e-6,
            conductivity=0.4,
            rho_cp=1542000,
        )

        beta = config.get("beta", 0.344)

        table_single = lib_data["table_single"]
        table_parallel = lib_data["table_parallel"]
        horiz_axes = lib_data["axes"]

        target_d = get_nearest(config["depth"], horiz_axes["depths"])
        target_beta = get_nearest(beta, horiz_axes["betas"])
        target_r = get_nearest(pipe.r_out, horiz_axes["radii"])
        target_k = get_nearest(config["soil_k"], horiz_axes["soil_ks"])

        is_coupled = config.get("type", "ISOLATED").upper() == "COUPLED"
        pipes = []

        r_in_steel = config.get("inner_diameter", 0.300) / 2.0
        r_out_steel = r_in_steel + 0.012
        steel_density = 7850
        steel_cp = 500

        if not is_coupled:
            q_prime_data = table_single[float_tuple_to_string((target_d, target_beta, target_r, target_k))]
            q_prime_interp = interpolate.interp1d(
                q_prime_data["x"], q_prime_data["y"], kind="cubic", fill_value="extrapolate"
            )
            horiz_pipe = IsolatedHorizontalPipe(
                name=f"{config['run_name']}_pipe",
                length=config["length"],
                num_segments=config["segments"],
                pipe=pipe,
                soil=soil,
                fluid=fluid,
                num_timesteps=num_timesteps,
                time_array=time_array,
                q_prime_interp=q_prime_interp,
                beta=beta,
                ugt_avg=config.get("ugt_avg", 15.0),
                ugt_amp1=config.get("ugt_amp1", 0.0),
                ugt_phase1=config.get("ugt_phase1", 0.0),
                ugt_amp2=config.get("ugt_amp2", 0.0),
                ugt_phase2=config.get("ugt_phase2", 0.0),
                depth=config["depth"],
                time_step_params=time_step_params,
                load_method=load_method,
            )

            cap_mult = config.get("capacitance_multiplier", 1.0)
            vol_steel_seg = np.pi * (r_out_steel**2 - r_in_steel**2) * horiz_pipe.L_seg
            c_steel_seg = vol_steel_seg * steel_density * steel_cp
            horiz_pipe.C_f_seg += c_steel_seg * cap_mult

            horiz_pipe.row_index = 0
            horiz_pipe.matrix_size = 3 * horiz_pipe.num_segments + 1
            horiz_pipe.t_in_initial = config.get("t_in_initial", 15.0)
            horiz_pipe.t_in_step = config.get("t_in_step", 35.0)
            horiz_pipe.branch_flow = config.get("mass_flow", 22.0)

            pipes.append(horiz_pipe)
            global_matrix_size = horiz_pipe.matrix_size

        else:
            target_b = get_nearest(config["spacing"], horiz_axes["spacings"])
            q_prime_data = table_parallel[float_tuple_to_string((target_d, target_b, target_beta, target_r, target_k))]
            q_prime_even = interpolate.interp1d(
                q_prime_data["x1"], q_prime_data["y1"], kind="cubic", fill_value="extrapolate"
            )
            q_prime_odd = interpolate.interp1d(
                q_prime_data["x2"], q_prime_data["y2"], kind="cubic", fill_value="extrapolate"
            )

            pipe1 = CoupledHorizontalPipe(
                name=f"{config['run_name']}_branch_A",
                length=config["length"],
                num_segments=config["segments"],
                pipe=pipe,
                soil=soil,
                fluid=fluid,
                num_timesteps=num_timesteps,
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
                time_step_params=time_step_params,
                load_method=load_method,
            )
            pipe2 = CoupledHorizontalPipe(
                name=f"{config['run_name']}_branch_B",
                length=config["length"],
                num_segments=config["segments"],
                pipe=pipe,
                soil=soil,
                fluid=fluid,
                num_timesteps=num_timesteps,
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
                time_step_params=time_step_params,
                load_method=load_method,
            )

            vol_steel_seg = np.pi * (r_out_steel**2 - r_in_steel**2) * pipe1.L_seg
            c_steel_seg = vol_steel_seg * steel_density * steel_cp
            pipe1.C_f_seg += c_steel_seg
            pipe2.C_f_seg += c_steel_seg

            pipe1.coupled_pipe = pipe2
            pipe2.coupled_pipe = pipe1
            pipe1.matrix_size = (3 * pipe1.num_segments + 1) * 2
            pipe2.matrix_size = pipe1.matrix_size
            pipe1.row_index = 0
            pipe2.row_index = 3 * pipe1.num_segments + 1
            pipe1.t_in_initial = config.get("t_in_initial_A", 15.0)
            pipe1.t_in_step = config.get("t_in_step_A", 25.0)
            pipe1.branch_flow = config.get("mass_flow_A", config.get("mass_flow", 22.0))
            pipe2.t_in_initial = config.get("t_in_initial_B", 15.0)
            pipe2.t_in_step = config.get("t_in_step_B", 5.0)
            pipe2.branch_flow = config.get("mass_flow_B", config.get("mass_flow", 22.0))

            pipes.extend([pipe1, pipe2])
            global_matrix_size = pipe1.matrix_size

        # --- Load Time-Varying Inlet Temperatures from CSV ---
        csv_path = config.get("t_in_csv_path")
        if csv_path and Path(csv_path).exists():
            df_inlet = pd.read_csv(csv_path)

            if len(df_inlet) < num_hours:
                raise ValueError(f"CSV only has {len(df_inlet)} rows, but simulation requires {num_hours}.")

            # Interpolate the hourly CSV data to our new fractional timestep array
            csv_hours = np.arange(len(df_inlet))
            csv_temps = df_inlet["T_in"].to_numpy()
            t_in_array = np.interp(time_array, csv_hours, csv_temps)

            for p in pipes:
                p.t_in_array = t_in_array
        else:
            for p in pipes:
                p.t_in_array = None

        # --- Micro-Solver Loop ---
        for t in range(1, num_timesteps):
            global_rows = []
            global_rhs = []

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

            for p in pipes:
                if getattr(p, "t_in_array", None) is not None:
                    p_current_t_in = p.t_in_array[t]
                else:
                    current_hour = time_array[t]
                    p_current_t_in = p.t_in_step if current_hour >= step_hour else p.t_in_initial

                global_rows[p.row_index] = np.zeros(global_matrix_size)
                global_rows[p.row_index][p.row_index] = 1.0
                global_rhs[p.row_index] = p_current_t_in

            a_matrix = np.array(global_rows, dtype=float)
            b_vector = np.array(global_rhs, dtype=float)
            x_vector = np.linalg.solve(a_matrix, b_vector)

            for p in pipes:
                p.update_post_solve(x_vector, t)

        # --- Export Results ---
        output_columns = {"Time [hr]": time_array[1:]}
        for p in pipes:
            output_columns[f"{p.name}_Inlet [C]"] = p.t_in[1:]
            output_columns[f"{p.name}_Outlet [C]"] = p.t_out_seg[-1, 1:]
            for k in range(p.num_segments):
                output_columns[f"{p.name}_Node{k + 1}_Tmean [C]"] = p.t_mean_seg[k, 1:]
                output_columns[f"{p.name}_Node{k + 1}_Q [W/m]"] = p.q_seg[k, 1:]

        output_data = pd.DataFrame(output_columns).set_index("Time [hr]")
        output_path = Path(config["output_dir"]) / f"{config['run_name']}.csv"
        output_data.to_csv(output_path, float_format="%0.4f")

        return (config["run_name"], True, None)

    except Exception as e:  # noqa: BLE001
        return (config["run_name"], False, str(e))


def generate_batch_configs(output_dir):
    """Generates all permutations of test cases based on defined input arrays."""
    depths = [1.5, 5.0, 15.0]
    spacings = [0.25, 0.5, 1.0]
    betas = [0.344]
    soil_ks = [1.0, 1.5, 2.0, 2.5]
    diameters = [0.0762, 0.1016, 0.1524]
    lengths = [50, 250, 500]
    segment_counts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    sim_configs = []
    case_num = 5185

    for d, s, beta, k, id_m, length, seg in itertools.product(
        depths, spacings, betas, soil_ks, diameters, lengths, segment_counts
    ):
        od_m = id_m + (0.01 * 0.0254)

        sim_configs.append(
            {
                "run_name": f"ISO_L{length}_d{d}_k{k}_id{id_m:.4f}_seg{seg}",
                "output_dir": output_dir,
                "type": "ISOLATED",
                "length": length,
                "segments": seg,
                "depth": d,
                "spacing": 0.0,
                "inner_diameter": id_m,
                "outer_diameter": od_m,
                "beta": beta,
                "soil_k": k,
                "ugt_avg": 15.0,
                "ugt_amp1": 0.0,
                "ugt_phase1": 0.0,
                "ugt_amp2": 0.0,
                "ugt_phase2": 0.0,
                "mass_flow": 0.5,
                "t_in_initial": 15.0,
                "t_in_step": 25.0,
            }
        )
        case_num += 1

    for d, s, beta, k, id_m, length, seg in itertools.product(
        depths, spacings, betas, soil_ks, diameters, lengths, segment_counts
    ):
        od_m = id_m + (0.01 * 0.0254)

        sim_configs.append(
            {
                "run_name": f"CPL_L{length}_d{d}_s{s}_k{k}_id{id_m:.4f}_seg{seg}",
                "output_dir": output_dir,
                "type": "COUPLED",
                "length": length,
                "segments": seg,
                "depth": d,
                "spacing": s,
                "inner_diameter": id_m,
                "outer_diameter": od_m,
                "beta": beta,
                "soil_k": k,
                "ugt_avg": 15.0,
                "ugt_amp1": 0.0,
                "ugt_phase1": 0.0,
                "ugt_amp2": 0.0,
                "ugt_phase2": 0.0,
                "mass_flow_A": 0.5,
                "mass_flow_B": 0.5,
                "t_in_initial_A": 15.0,
                "t_in_step_A": 25.0,
                "t_in_initial_B": 15.0,
                "t_in_step_B": 5.0,
            }
        )
        case_num += 1

    return sim_configs


def main(output_dir: Path | None = None):
    if output_dir is None:
        output_dir = Path.cwd() / "horizontal_component_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_configs = generate_batch_configs(str(output_dir))
    total_jobs = len(sim_configs)

    num_cores = max(1, multiprocessing.cpu_count() - 1)

    print("\n--- Horizontal Pipe Batch Simulation ---")
    print(f"Total Configurations: {total_jobs}")
    print(f"Allocating CPUs: {num_cores}")
    print(f"Output Directory: {output_dir}\n")

    t_start = time.perf_counter()
    failed_jobs = []

    # Process Pool setup
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(run_horizontal_simulation, sim_configs), 1):
            run_name, success, error_msg = result

            if not success:
                failed_jobs.append((run_name, error_msg))

            if i % 1 == 0 or i == total_jobs:
                elapsed_time = time.perf_counter() - t_start
                avg_time_per_job = elapsed_time / i
                remaining_jobs = total_jobs - i
                eta_seconds = remaining_jobs * avg_time_per_job

                eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds))
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))

                print(
                    f"  Progress: {i}/{total_jobs} | Elapsed: {elapsed_str} | "
                    f"Remaining: ~{eta_str} ({avg_time_per_job:.2f}s/job)    ",
                    end="\r",
                    flush=True,
                )

    print(f"\n\nDone! Total batch time: {time.perf_counter() - t_start:.2f} seconds.")

    if failed_jobs:
        print(f"\nWARNING: {len(failed_jobs)} jobs failed:")
        for name, err in failed_jobs[:10]:
            print(f"  - {name}: {err}")


def run_single_case(output_dir: Path, inlet_temperature_csv: Path | None = None):
    init_worker()
    output_dir.mkdir(parents=True, exist_ok=True)
    single_config = {
        "run_name": "Vilnius_DH_Experiment_12_min_timestep_big_seg",
        "output_dir": str(output_dir),
        "type": "ISOLATED",
        "length": 470.0,
        "segments": 120,
        "depth": 1.0,
        "inner_diameter": 0.300,
        "outer_diameter": 0.450,
        "beta": 12.0,
        "soil_k": 1.5,
        "ugt_avg": 7.0,
        "mass_flow": 2.76,
        "capacitance_multiplier": 1.0,
        "load_method": "hourlyloadagg",
        "num_hours": 8760,
        "steps_per_hour": 5,
    }
    if inlet_temperature_csv is not None:
        single_config["t_in_csv_path"] = str(inlet_temperature_csv)

    print("--- Running Single Horizontal Pipe Simulation ---")
    print(f"Run Name: {single_config['run_name']}")
    t_start = time.perf_counter()
    _, success, error_msg = run_horizontal_simulation(single_config)

    if not success:
        raise RuntimeError(f"Simulation failed: {error_msg}")
    print(f"Success! Finished in {time.perf_counter() - t_start:.2f} seconds.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run standalone horizontal-pipe component simulations.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "horizontal_component_results",
        help="Directory for simulation CSV output.",
    )
    parser.add_argument("--batch", action="store_true", help="Run the generated batch instead of the single case.")
    parser.add_argument("--inlet-temperature-csv", type=Path, help="Optional T_in CSV for the single case.")
    arguments = parser.parse_args()

    if arguments.batch:
        main(arguments.output_dir)
    else:
        run_single_case(arguments.output_dir, arguments.inlet_temperature_csv)
