"""
Unit tests for ghedesigner.lcoe.quantities.extract_quantities.

GHE and HP objects are mocked — no GHEDesigner sizing is run.

GHE mock needs:  bhe.borehole.H (float), nbh (int)
HP mock needs:   cop (float), loads (list[float]) — hourly W, + = heating, - = cooling
"""

import pytest
from unittest.mock import MagicMock

from ghedesigner.lcoe.quantities import extract_quantities


# ---------------------------------------------------------------------------
# Mock factories
# ---------------------------------------------------------------------------


def _ghe(borehole_H: float, nbh: int) -> MagicMock:
    ghe = MagicMock()
    ghe.bhe.borehole.H = borehole_H
    ghe.nbh = nbh
    return ghe


def _hp(cop: float, loads_w: list[float]) -> MagicMock:
    hp = MagicMock()
    hp.cop = cop
    hp.loads = loads_w
    return hp


# ---------------------------------------------------------------------------
# Empty inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_no_ghe_no_hp_returns_all_zeros(self):
        q = extract_quantities([], [])
        assert q.total_drilling_m == 0.0
        assert q.n_boreholes == 0
        assert q.n_heat_pumps == 0
        assert q.heat_MWh_yr1 == 0.0
        assert q.cool_MWh_yr1 == 0.0
        assert q.elec_heat_MWh_yr1 == 0.0
        assert q.elec_cool_MWh_yr1 == 0.0

    def test_ghe_only_no_hp_energy_is_zero(self):
        q = extract_quantities([_ghe(100.0, 4)], [])
        assert q.heat_MWh_yr1 == 0.0
        assert q.cool_MWh_yr1 == 0.0


# ---------------------------------------------------------------------------
# GHE geometry
# ---------------------------------------------------------------------------


class TestGHEGeometry:
    def test_single_ghe_drilling_m(self):
        q = extract_quantities([_ghe(100.0, 6)], [])
        assert q.total_drilling_m == pytest.approx(600.0)
        assert q.n_boreholes == 6

    def test_multiple_ghe_aggregates(self):
        # 4 boreholes × 100 m + 2 boreholes × 150 m = 700 m, 6 boreholes
        q = extract_quantities([_ghe(100.0, 4), _ghe(150.0, 2)], [])
        assert q.total_drilling_m == pytest.approx(700.0)
        assert q.n_boreholes == 6


# ---------------------------------------------------------------------------
# HP energy extraction
# ---------------------------------------------------------------------------


class TestHPEnergy:
    def test_heating_only(self):
        # 1 MW heating for 1 hour = 1 MWh; COP=4 → 0.25 MWh elec
        q = extract_quantities([], [_hp(cop=4.0, loads_w=[1_000_000.0])])
        assert q.heat_MWh_yr1 == pytest.approx(1.0)
        assert q.cool_MWh_yr1 == pytest.approx(0.0)
        assert q.elec_heat_MWh_yr1 == pytest.approx(0.25)
        assert q.elec_cool_MWh_yr1 == pytest.approx(0.0)

    def test_cooling_only(self):
        # 1 MW cooling for 1 hour = 1 MWh; COP=4 → 0.25 MWh elec
        q = extract_quantities([], [_hp(cop=4.0, loads_w=[-1_000_000.0])])
        assert q.cool_MWh_yr1 == pytest.approx(1.0)
        assert q.heat_MWh_yr1 == pytest.approx(0.0)
        assert q.elec_cool_MWh_yr1 == pytest.approx(0.25)
        assert q.elec_heat_MWh_yr1 == pytest.approx(0.0)

    def test_zero_loads_ignored(self):
        q = extract_quantities([], [_hp(cop=3.0, loads_w=[0.0, 0.0, 0.0])])
        assert q.heat_MWh_yr1 == 0.0
        assert q.cool_MWh_yr1 == 0.0

    def test_mixed_loads(self):
        # 2 MW heat × 1 hr = 2 MWh; 1 MW cool × 1 hr = 1 MWh; COP=2
        q = extract_quantities([], [_hp(cop=2.0, loads_w=[2_000_000.0, -1_000_000.0])])
        assert q.heat_MWh_yr1 == pytest.approx(2.0)
        assert q.cool_MWh_yr1 == pytest.approx(1.0)
        assert q.elec_heat_MWh_yr1 == pytest.approx(1.0)
        assert q.elec_cool_MWh_yr1 == pytest.approx(0.5)

    def test_elec_scales_with_cop(self):
        """Higher COP → lower electricity consumption for same load."""
        q_low = extract_quantities([], [_hp(cop=2.0, loads_w=[1_000_000.0])])
        q_high = extract_quantities([], [_hp(cop=4.0, loads_w=[1_000_000.0])])
        assert q_low.elec_heat_MWh_yr1 > q_high.elec_heat_MWh_yr1

    def test_multiple_hps_aggregate(self):
        hp1 = _hp(cop=3.0, loads_w=[3_000_000.0])   # 3 MWh heat
        hp2 = _hp(cop=2.0, loads_w=[-2_000_000.0])  # 2 MWh cool
        q = extract_quantities([], [hp1, hp2])
        assert q.n_heat_pumps == 2
        assert q.heat_MWh_yr1 == pytest.approx(3.0)
        assert q.cool_MWh_yr1 == pytest.approx(2.0)
        assert q.elec_heat_MWh_yr1 == pytest.approx(1.0)   # 3 / 3
        assert q.elec_cool_MWh_yr1 == pytest.approx(1.0)   # 2 / 2

    def test_n_heat_pumps_equals_len_hp_objects(self):
        hps = [_hp(3.0, [1_000.0]) for _ in range(5)]
        q = extract_quantities([], hps)
        assert q.n_heat_pumps == 5