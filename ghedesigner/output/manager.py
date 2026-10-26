import csv
from json import dumps
from pathlib import Path
from typing import Any

from numpy import ndarray

from ghedesigner.enums import TimestepType
from ghedesigner.ghe.design.base import AnyBisectionType
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.output import columns as csv_columns
from ghedesigner.output.converters import ghe_time_convert, hours_to_month
from ghedesigner.output.json_serializer import JsonSerializer
from ghedesigner.output.text_serializer import TextSerializer
from ghedesigner.utilities import write_json


def _get_borehole_location_data(ghe: GHE, object_name: str) -> list[list[Any]]:
    header = [
        csv_columns.output_column(object_name, "Borehole X Coordinate", "m"),
        csv_columns.output_column(object_name, "Borehole Y Coordinate", "m"),
    ]
    return [header] + [[x, y] for x, y in ghe.gFunction.bore_locations]


def _get_hourly_loading_data(ghe: GHE, object_name: str) -> list[list[Any]]:
    rows: list[list[Any]] = [
        [
            csv_columns.output_column(csv_columns.SIMULATION, "Month", "-"),
            csv_columns.output_column(csv_columns.SIMULATION, "Day", "-"),
            csv_columns.output_column(csv_columns.SIMULATION, "Hour", "-"),
            csv_columns.ELAPSED_TIME.for_object(csv_columns.SIMULATION),
            csv_columns.output_column(object_name, "Ground Heat Extraction Rate", "W"),
        ]
    ]
    for hr, load in enumerate(ghe.hourly_extraction_ground_loads):
        m, d, h = ghe_time_convert(hr)
        rows.append([m, d, h, hr, load])
    return rows


def _get_g_function_data(ghe: GHE, object_name: str) -> list[list[Any]]:
    gf_adj, gf_bhw = ghe.grab_g_function(ghe.b_spacing / ghe.bhe.borehole.H)
    header = [
        csv_columns.output_column(object_name, "Log Time Ratio", "-"),
        csv_columns.output_column(object_name, "G-Function", "-"),
        csv_columns.output_column(object_name, "Borehole-Wall G-Function", "-"),
    ]
    return [header] + [[x, y, z] for x, y, z in zip(gf_adj.x, gf_adj.y, gf_bhw.y)]


def _get_loading_data(ghe: GHE, object_name: str) -> list[list[Any]]:
    times = ghe.times
    d_tb = ghe.dTb
    hp_eft = ghe.hp_eft
    loading = ghe.loading
    denom = ghe.bhe.borehole.H * ghe.nbh
    ugt = ghe.bhe.soil.ugt

    rows: list[list[Any]] = [
        [
            csv_columns.ELAPSED_TIME.for_object(csv_columns.SIMULATION),
            csv_columns.output_column(csv_columns.SIMULATION, "Elapsed Time", "month"),
            csv_columns.output_column(object_name, "Ground Heat Rejection Rate Before Time Step", "W"),
            csv_columns.output_column(object_name, "Ground Heat Rejection Rate Before Time Step", "W/m"),
            csv_columns.output_column(object_name, "Borehole Wall Temperature", "C"),
            csv_columns.EXITING_FLUID_TEMPERATURE.for_object(object_name),
        ]
    ]

    n = len(times)
    for i, t in enumerate(times):
        month = hours_to_month(t)

        # "Before time" row (uses current loading; Tb/EFT from previous index)
        q_before = loading[i] if loading is not None and i > 1 else 0
        rows.append(
            [
                t,
                month,
                q_before,
                (q_before / denom) if i > 1 else 0,
                # TODO The next two lines wrap to the last element when i==0, it's not clear if that's intentional
                ugt + d_tb[i - 1],
                hp_eft[i - 1],
            ]
        )

        # "After time" row (uses next loading; current Tb/EFT)
        q_after = loading[i + 1] if loading is not None and (i + 1) < n else 0
        rows.append(
            [
                t,
                month,
                q_after,
                (q_after / denom) if q_after else 0,
                ugt + d_tb[i],
                hp_eft[i],
            ]
        )

    return rows


class OutputManager:
    """
    Orchestrates writing of all GHE design outputs:
      - text summary
      - multiple CSVs
      - JSON summary
    """

    def __init__(
        self,
        project_name: str,
        notes: str,
        author: str,
        model_name: str,
        allocated_width: int = 100,
        object_name: str = "GHE",
    ) -> None:
        self.project_name = project_name
        self.notes = notes
        self.author = author
        self.model_name = model_name
        self.allocated_width = allocated_width
        self.object_name = object_name

        self.design: AnyBisectionType | None = None
        self.time: float = 0.0
        self.load_method: TimestepType = TimestepType.HYBRID

    @staticmethod
    def just_write_g_function(
        output_directory: Path,
        log_time: ndarray,
        g_values: ndarray,
        g_bhw_values: ndarray,
        object_name: str = "GHE",
    ) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "log_time": log_time.tolist(),
            "g_values": g_values.tolist(),
            "g_bhw_values": g_bhw_values.tolist(),
        }
        write_json(output_directory / "SimulationSummary.json", summary)
        rows = [
            [
                csv_columns.output_column(object_name, "Log Time Ratio", "-"),
                csv_columns.output_column(object_name, "G-Function", "-"),
                csv_columns.output_column(object_name, "Borehole-Wall G-Function", "-"),
            ],
            *zip(log_time, g_values, g_bhw_values),
        ]
        with open(output_directory / "Gfunction.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)

    def set_design_data(
        self,
        design: AnyBisectionType,
        time: float,
        load_method: TimestepType,
    ) -> None:
        """Store design result and runtime for later writing."""
        self.design = design
        self.time = time
        self.load_method = load_method

    def write_all_output_files(
        self,
        output_directory: Path,
        file_suffix: str = "",
    ) -> None:
        """Write .txt, .csv and .json outputs for a full simulation."""
        if self.design is None or self.design.ghe is None:
            raise ValueError("Design data has not been set")

        ghe = self.design.ghe
        tracker = self.design.searchTracker

        output_directory.mkdir(parents=True, exist_ok=True)

        # Text summary
        txt = TextSerializer.summary_text(
            self.allocated_width,
            self.project_name,
            self.model_name,
            self.notes,
            self.author,
            self.time,
            ghe,
            tracker,
        )
        (output_directory / f"SimulationSummary{file_suffix}.txt").write_text(txt)

        # CSVs
        with open(output_directory / f"TimeDependentValues{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(_get_loading_data(ghe, self.object_name))

        with open(output_directory / f"BoreFieldData{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(_get_borehole_location_data(ghe, self.object_name))

        with open(output_directory / f"Loadings{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(_get_hourly_loading_data(ghe, self.object_name))

        with open(output_directory / f"Gfunction{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(_get_g_function_data(ghe, self.object_name))

        # JSON summary
        obj = JsonSerializer.summary_object(
            ghe,
            tracker,
            self.time,
            self.project_name,
            self.notes,
            self.author,
            self.model_name,
            self.load_method,
        )

        with open(output_directory / f"SimulationSummary{file_suffix}.json", "w", newline="") as f:
            f.write(dumps(obj, indent=2))

    def write_presized_output_files(
        self,
        output_directory: Path,
        ghe: GHE,
        file_suffix: str = "",
    ) -> None:
        """Write minimal outputs for a presized design."""
        output_directory.mkdir(parents=True, exist_ok=True)

        txt = TextSerializer.summary_text(
            self.allocated_width,
            self.project_name,
            self.model_name,
            self.notes,
            self.author,
            self.time,
            ghe,
            "none",
        )
        (output_directory / f"SimulationSummary{file_suffix}.txt").write_text(txt)

        with open(output_directory / f"BoreFieldData{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(_get_borehole_location_data(ghe, self.object_name))

        with open(output_directory / f"Gfunction{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(_get_g_function_data(ghe, self.object_name))
