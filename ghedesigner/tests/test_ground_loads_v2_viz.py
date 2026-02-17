"""Visualization and export tests for HybridLoadV2.

These tests generate CSV exports and matplotlib plots for visual inspection
of hourly vs hybrid load profiles. They are separated from the functional
tests in test_ground_loads_v2.py.
"""

import pygfunction as gt
import unittest
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pathlib import Path
from unittest.mock import MagicMock, patch
from ghedesigner.constants import MONTHS_IN_YEAR, TWO_PI
from ghedesigner.ghe.ground_loads_v2 import HybridLoadV2
from ghedesigner.ghe.boreholes.single_u_borehole import SingleUTube
from ghedesigner.media import Fluid, Soil, Grout
from ghedesigner.ghe.pipe import Pipe
from ghedesigner.enums import PipeType
import matplotlib
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helper: build a minimal mock BHE for visualization tests
# ---------------------------------------------------------------------------
def _make_mock_bhe():
    """Create a mock SingleUTube with minimal attributes for HybridLoadV2."""
    bhe = MagicMock()
    bhe.soil.k = 2.0
    bhe.calc_effective_borehole_resistance.return_value = 0.15
    bhe.t_s = 1.0e10  # large ts so log(t/ts) stays in range

    # STS g-function: simple linear g = 0.5 * ln(t/ts) clamped to [0, 10].
    # Note: with ts=1e10, g returns 0 for the first ~126 hours because
    # ln(126*3600/1e10) ≈ -10 → 0.5*(-10)+5 = 0.  This means the ground
    # response is zero at short times and only the borehole resistance term
    # (q/H * Rb) contributes, producing a flat delta_T during that window.
    # A real STS g-function would respond from hour 1 onward.
    def g_sts_func(lntts):
        vals = np.asarray(lntts, dtype=float)
        result = np.clip(0.5 * vals + 5.0, 0.0, 10.0)
        return result

    bhe.g_sts = g_sts_func
    return bhe


def _analyze_and_export(
    loads: list,
    output_dir: Path,
    file_prefix: str,
    undisturbed_ground_t: float = 20.0,
    m_dot: float = 0.1,
    cp: float = 4186.0,
    bhe=None,
) -> None:
    """Build HybridLoadV2 from any hourly load profile and export analysis CSVs.

    Exports two files to output_dir:
      - {file_prefix}_hourly_results.csv  : per-hour load, delta_t, tf_ave, eft, exft
      - {file_prefix}_hybrid_results.csv  : hybrid time steps in step-function format
                                            with predicted delta_T

    :param loads: 8760 hourly loads in Watts
    :param output_dir: directory to write CSV files
    :param file_prefix: prefix for output file names
    :param undisturbed_ground_t: undisturbed ground temperature in °C
    :param m_dot: fluid mass flow rate in kg/s
    :param cp: fluid specific heat in J/(kg·K)
    :param bhe: optional BHE object; defaults to _make_mock_bhe()
    """
    if bhe is None:
        bhe = _make_mock_bhe()
    obj = HybridLoadV2(loads, bhe, bhe, 1, 12)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    print(f"\n===== {file_prefix}: Peak Temperature & Load Summary =====")
    print(f"{'Month':<6} {'Max dT':>8} {'Hour':>6} {'Peak CL (W)':>13} "
          f"{'Min dT':>8} {'Hour':>6} {'Peak HL (W)':>13}")
    print("-" * 68)
    for m in range(MONTHS_IN_YEAR):
        print(f"{month_names[m]:<6} {obj.monthly_max_dt[m]:>8.3f} {obj.monthly_max_dt_hour[m]:>6d} "
              f"{obj.monthly_peak_cl[m]:>13.3f} {obj.monthly_min_dt[m]:>8.3f} "
              f"{obj.monthly_min_dt_hour[m]:>6d} {obj.monthly_peak_hl[m]:>13.3f}")
    print(f"\nPeak cooling months: {[month_names[m] for m in obj.peak_cooling_months]}")
    print(f"Peak heating months: {[month_names[m] for m in obj.peak_heating_months]}")
    print("\nPeak durations (hours):")
    for m in obj.peak_cooling_months:
        print(f"  {month_names[m]} cooling: {obj.monthly_peak_cl_duration[m]:.2f}")
    for m in obj.peak_heating_months:
        print(f"  {month_names[m]} heating: {obj.monthly_peak_hl_duration[m]:.2f}")
    print("=" * 68)

    print(f"\n===== {file_prefix}: Hybrid Load Profile (All Steps) =====")
    print(f"{'Step':>4}  {'Start Hr':>10}  {'End Hr':>10}  {'Duration':>10}  {'Load (W)':>10}")
    print("-" * 52)
    for i in range(1, len(obj.hour)):
        h_start = obj.hour[i - 1]
        h_end = obj.hour[i]
        print(f"{i:>4}  {h_start:>10.1f}  {h_end:>10.1f}  {h_end - h_start:>10.1f}  {obj.load[i]:>10.3f}")
    print("=" * 52)

    # Hourly results: delta_t (Tf_ave-Tg0), tf_ave, eft, exft
    # q is aligned to hourly_delta_t indexing (8761 entries, index 0 = 0)
    q_norm = np.hstack((0.0, obj.normalized_loads))
    tf_ave = undisturbed_ground_t + np.array(obj.hourly_delta_t)
    eft = tf_ave + q_norm / (2 * m_dot * cp)
    exft = tf_ave - q_norm / (2 * m_dot * cp)

    hourly_path = output_dir / f"{file_prefix}_hrly_results.csv"
    with open(hourly_path, "w") as f:
        f.write("hour,norm_load_w,tfave-tg0,exft,tf_ave,eft\n")
        for hour_idx, (nq, dt, exft_val, tf_ave_val, eft_val) in enumerate(
            zip(q_norm, obj.hourly_delta_t, exft, tf_ave, eft)
        ):
            f.write(f"{hour_idx},{nq:.6f},{dt:.6f},{exft_val:.6f},{tf_ave_val:.6f},{eft_val:.6f}\n")
    print(f"\nExported hourly results to {hourly_path}")
    print(f"  entries={len(obj.hourly_delta_t)}, "
          f"delta_T=[{min(obj.hourly_delta_t):.4f}, {max(obj.hourly_delta_t):.4f}], "
          f"EFT=[{min(eft):.4f}, {max(eft):.4f}] °C")

    # Hybrid results: Step 7 output arrays
    hybrid_delta_t = obj.hybrid_dt
    norm_hybrid_q = obj.hybrid_q_norm_w

    month_name_to_idx = {name: idx for idx, name in enumerate(month_names)}

    hybrid_path = output_dir / f"{file_prefix}_hybrid_results.csv"
    with open(hybrid_path, "w") as f:
        f.write("hour,load_w,hybrid_dt,peak,load_factor\n")
        for i in range(len(obj.hour)):
            label = obj.step_label[i]
            if label.startswith("CL_"):
                lf = f"{obj.monthly_peak_cl_load_factor[month_name_to_idx[label[3:]]]:.6f}"
            elif label.startswith("HL_"):
                lf = f"{obj.monthly_peak_hl_load_factor[month_name_to_idx[label[3:]]]:.6f}"
            else:
                lf = ""
            f.write(f"{obj.hour[i]:.4f},{obj.load[i]:.6f},{obj.hybrid_dt[i]:.6f},{label},{lf}\n")
    print(f"Exported hybrid results to {hybrid_path}")

    #------------------------------------------------
    # Matplotlib plots
    #------------------------------------------------

    hourly_hours = np.arange(len(loads))  # 0..8759

    # Plot 1: Normalized hourly loads vs normalized hybrid loads
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(hourly_hours, obj.normalized_loads, linewidth=1, alpha=0.7, label="Hourly normalized")
    ax.step(obj.hour[:-1], norm_hybrid_q[1:], where="post", linewidth=1, color="tab:orange", label="Hybrid normalized")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Normalized Load (W)")
    ax.set_title(f"{file_prefix}: Normalized Hourly vs Hybrid Loads")
    ax.set_xlim(0, 8760)
    ax.set_xticks(np.arange(0, 8761, 730))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}_norm_loads.png", dpi=150)
    plt.close(fig)
    print(f"Exported plot to {output_dir / f'{file_prefix}_norm_loads.png'}")

    # Plot 2: Hourly delta_T vs hybrid delta_T
    fig, ax = plt.subplots(figsize=(14, 5))
    dt_hours = np.arange(len(obj.hourly_delta_t))  # 0..8760
    ax.plot(dt_hours, obj.hourly_delta_t, linewidth=0.3, alpha=0.7, label="Hourly Tf_ave - Tg0")
    ax.plot(obj.hour, hybrid_delta_t, linewidth=1.5, color="tab:orange", marker=".", markersize=4, label="Hybrid Tf_ave - Tg0", )
    ax.set_xlabel("Hour")
    ax.set_ylabel("Tf_ave - Tg0 (K)")
    ax.set_title(f"{file_prefix}: Hourly vs Hybrid Delta-T")
    ax.set_xlim(0, 8760)
    ax.set_xticks(np.arange(0, 8761, 730))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}_delta_t.png", dpi=150)
    plt.close(fig)
    print(f"Exported plot to {output_dir / f'{file_prefix}_delta_t.png'}")

    # Plot 3: Hourly fluid & borehole temperatures (EFT, Tf_ave, ExFT, Tb)
    # Tb = Tf_ave + q/(H * Rb) — borehole wall temp from resistance relation
    # For extraction (q > 0): Tb > Tf_ave (heat flows from wall to fluid)
    resist_bh = obj.bhe.calc_effective_borehole_resistance()
    h_bore = obj.NORM_BOREHOLE_H
    tb = tf_ave + q_norm / h_bore * resist_bh

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dt_hours, eft, linewidth=1, alpha=0.7, color="tab:red", label="hp_EFT")
    ax.plot(dt_hours, tf_ave, linewidth=1, alpha=0.7, color="tab:blue", label="Tf_ave")
    ax.plot(dt_hours, exft, linewidth=1, alpha=0.7, color="tab:green", label="hp_ExFT")
    ax.plot(dt_hours, tb, linewidth=1, alpha=0.7, color="tab:purple", label="Tb")
    ax.axhline(y=undisturbed_ground_t, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Tg0")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"{file_prefix}: Hourly Fluid & Borehole Temperatures")
    ax.set_xlim(0, 8760)
    ax.set_xticks(np.arange(0, 8761, 730))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"{file_prefix}_temperatures.png", dpi=150)
    plt.close(fig)
    print(f"Exported plot to {output_dir / f'{file_prefix}_temperatures.png'}")

    # Plot 4: Dual y-axis — loads (W) + temperatures (°C)
    fig, ax1 = plt.subplots(figsize=(14, 5))

    # Left axis: loads
    ax1.plot(hourly_hours, obj.normalized_loads, linewidth=1, alpha=0.5, color="tab:blue", label="Hourly load")
    ax1.step(obj.hour[:-1], norm_hybrid_q[1:], where="post", linewidth=1, linestyle ='--',color="tab:cyan", marker=".", markersize=4, label="Hybrid load")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Load (W)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xlim(0, 8760)
    ax1.set_xticks(np.arange(0, 8761, 730))

    # Right axis: temperatures
    ax2 = ax1.twinx()
    ax2.plot(dt_hours, np.array(obj.hourly_delta_t), linewidth=1, alpha=0.7, color="tab:orange", label="Hourly Tf_ave - Tg0")
    ax2.plot(obj.hour, hybrid_delta_t, linestyle = 'None', color="tab:green", marker=".", markersize=4,label="Hybrid Tf_ave - Tg0")
    ax2.set_ylabel("Temperature (°C / K)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=8)

    ax1.set_title(f"{file_prefix}: Loads & Temperatures")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_dir / f"{file_prefix}_loads_and_temps.png", dpi=150)
    plt.close(fig)
    print(f"Exported plot to {output_dir / f'{file_prefix}_loads_and_temps.png'}")

#-----------------------------------------------
# CSV generation
#-----------------------------------------------

def test_export_constant_cooling_analysis():
    """Export hourly and hybrid results CSVs for the constant cooling load profile."""
    csv_path = Path(__file__).parent / "test_data" / "constant_1000w_cooling.csv"
    if not csv_path.exists():
        pytest.skip("constant cooling CSV not found")

    raw_lines = csv_path.read_text().split("\n")
    loads = [float(x) for x in raw_lines[1:] if x.strip() != ""]

    output_dir = Path(__file__).parent / "output"
    _analyze_and_export(loads, output_dir, "CC", 15)

def test_export_balanced_ramp_analysis():
    """Export hourly and hybrid results CSVs for the ramp load profile."""
    csv_path = Path(__file__).parent / "test_data" / "balanced_ramp.csv"
    if not csv_path.exists():
        pytest.skip("balanced_ramp CSV not found")

    raw_lines = csv_path.read_text().split("\n")
    loads = [float(x) for x in raw_lines[1:] if x.strip() != ""]

    output_dir = Path(__file__).parent / "output"
    _analyze_and_export(loads, output_dir, "BR", 10)


def test_export_atlanta_analysis():
    """Export hourly and hybrid results CSVs for the Atlanta load profile."""
    csv_path = Path(__file__).parent / "test_data" / "Atlanta_Office_Building_Loads.csv"
    if not csv_path.exists():
        pytest.skip("Atlanta loads CSV not found")

    raw_lines = csv_path.read_text().split("\n")
    loads = [float(x) for x in raw_lines[1:] if x.strip() != ""]

    output_dir = Path(__file__).parent / "output"
    _analyze_and_export(loads, output_dir, "ATL")


