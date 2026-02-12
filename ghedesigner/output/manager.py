import csv
from json import dumps
from pathlib import Path
from typing import Any

from numpy import ndarray

from ghedesigner.enums import TimestepType
from ghedesigner.ghe.design.base import AnyBisectionType
from ghedesigner.ghe.ground_heat_exchangers import GHE
from ghedesigner.output.converters import ghe_time_convert, hours_to_month
from ghedesigner.output.json_serializer import JsonSerializer
from ghedesigner.output.text_serializer import TextSerializer
from ghedesigner.utilities import write_flat_dict_to_csv, write_json


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
    ) -> None:
        self.project_name = project_name
        self.notes = notes
        self.author = author
        self.model_name = model_name
        self.allocated_width = allocated_width

        self.design: AnyBisectionType | None = None
        self.time: float = 0.0
        self.load_method: TimestepType = TimestepType.HYBRID

    @staticmethod
    def just_write_g_function(
        output_directory: Path,
        log_time: ndarray,
        g_values: ndarray,
        g_bhw_values: ndarray,
    ) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        summary = {
            "log_time": log_time.tolist(),
            "g_values": g_values.tolist(),
            "g_bhw_values": g_bhw_values.tolist(),
        }
        write_json(output_directory / "SimulationSummary.json", summary)
        write_flat_dict_to_csv(output_directory / "Gfunction.csv", summary)

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
            csv.writer(f).writerows(self._get_loading_data(ghe))

        with open(output_directory / f"BoreFieldData{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(self._get_borehole_location_data(ghe))

        with open(output_directory / f"Loadings{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(self._get_hourly_loading_data(ghe))

        with open(output_directory / f"Gfunction{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(self._get_g_function_data(ghe))

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
            csv.writer(f).writerows(self._get_borehole_location_data(ghe))

        with open(output_directory / f"Gfunction{file_suffix}.csv", "w", newline="") as f:
            csv.writer(f).writerows(self._get_g_function_data(ghe))

        # This is commented out because it was writing dummy data
        # write_json(output_directory / f"SimulationSummary{file_suffix}.json", {
        #     "ghe_system": {
        #         "number_of_boreholes": 1,
        #         "active_borehole_length": {"value": 1},
        #     }
        # })

    # Internal data extraction for CSVs
    def _get_loading_data(self, ghe: GHE) -> list[list[Any]]:
        times = ghe.times
        d_tb = ghe.dTb
        hp_eft = ghe.hp_eft
        loading = ghe.loading
        denom = ghe.bhe.borehole.H * ghe.nbh
        ugt = ghe.bhe.soil.ugt

        rows: list[list[Any]] = [
            [
                "Time (hr)",
                "Time (month)",
                "Q (Rejection) (W) (before time)",
                "Q (Rejection) (W/m) (before time)",
                "Tb (C)",
                "GHE ExFT (C)",
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

    def _get_borehole_location_data(self, ghe: GHE) -> list[list[Any]]:
        return [["x", "y"]] + [[x, y] for x, y in ghe.gFunction.bore_locations]

    def _get_hourly_loading_data(self, ghe: GHE) -> list[list[Any]]:
        rows: list[list[Any]] = [["Month", "Day", "Hour", "Time (Hours)", "Loading (W) (Extraction)"]]
        for hr, load in enumerate(ghe.hourly_extraction_ground_loads):
            m, d, h = ghe_time_convert(hr)
            rows.append([m, d, h, hr, load])
        return rows

    def _get_g_function_data(self, ghe: GHE) -> list[list[Any]]:
        title = f"H: {ghe.bhe.borehole.H:0.2f} m"
        gf_adj, gf_bhw = ghe.grab_g_function(ghe.b_spacing / ghe.bhe.borehole.H)
        header = ["ln(t/ts)", title, f"{title} bhw"]
        return [header] + [[x, y, z] for x, y, z in zip(gf_adj.x, gf_adj.y, gf_bhw.y)]
