from unittest import TestCase

from scp.base_fluid import BaseFluid

from ghedesigner.media import Fluid


class TestMedia(TestCase):
    def test_fluid(self):
        fluid = Fluid("Water")
        self.assertAlmostEqual(fluid.cp, 4181.9, delta=1e-1)
        self.assertAlmostEqual(fluid.k, 0.598, delta=1e-3)
        self.assertAlmostEqual(fluid.mu, 0.001002, delta=1e-6)
        self.assertAlmostEqual(fluid.rho, 998.2, delta=1e-1)
        self.assertAlmostEqual(fluid.rho_cp, 4174435, delta=1e0)

        fluid.update_props_with_new_temp(30)
        self.assertAlmostEqual(fluid.cp, 4177.8, delta=1e-1)
        self.assertAlmostEqual(fluid.k, 0.614, delta=1e-3)
        self.assertAlmostEqual(fluid.mu, 0.0007975, delta=1e-6)
        self.assertAlmostEqual(fluid.rho, 995.6, delta=1e-1)
        self.assertAlmostEqual(fluid.rho_cp, 4159624, delta=1e0)

    def test_fluid_aliases_use_secondary_coolant_factory(self):
        aliases = {
            "MEA": "EthylAlcohol",
            "MEG": "EthyleneGlycol",
            "MMA": "MethylAlcohol",
            "MPG": "PropyleneGlycol",
            "Water": "Water",
        }

        for alias, expected_name in aliases.items():
            with self.subTest(alias=alias):
                fluid = Fluid(alias, percent=20)
                self.assertIsInstance(fluid.scp_fluid, BaseFluid)
                self.assertEqual(fluid.scp_fluid.fluid_name, expected_name)
