import os
import unittest
from math import pi

try:
    # noinspection PyPackageRequirements
    import pandas as pd
    skip_validation = False
except ImportError:
    pd = None
    skip_validation = True
   
import numpy as np

from ghedt.rowwise.rowwise_generation import (
    field_optimization_fr,
    field_optimization_wp_space_fr,
    gen_borehole_config,
    gen_shape,
)

REFERENCE_DATA_FILE = os.path.join(
    os.path.dirname(__file__), "test_data", "rowwise_reference_values.csv"
)

PROPERTY_BOUNDARY_FILE = os.path.join(
    os.path.dirname(__file__), "test_data", "property_polygons", "property_boundary.csv"
)

BUILDING_FILE = os.path.join(
    os.path.dirname(__file__), "test_data", "property_polygons", "building.csv"
)


@unittest.skipIf(skip_validation, "To run these tests, pip install pandas")
class TestRowWise(unittest.TestCase):
    def setUp(self) -> None:

        # Reference Values
        self.reference_values = pd.read_csv(REFERENCE_DATA_FILE)

        # Load Property Boundary
        prop_polygon_df: pd.DataFrame = pd.read_csv(PROPERTY_BOUNDARY_FILE)
        self.prop_polygon_ar: list = prop_polygon_df.values.tolist()

        # Load Building
        build_polygon_df: pd.DataFrame = pd.read_csv(BUILDING_FILE)
        self.building_polygon_ar: list = build_polygon_df.values.tolist()

        # Establish Properties
        self.buildings = None
        self.property = None
        self.perimeter_spac_ratio = 0.7
        self.target_spacing_start = 10.0  # in meters
        self.target_spacing_stop = 20.0  # in meters
        self.target_spacing_step = 1  # in meters
        self.target_spacing_number = (
            int(
                (self.target_spacing_stop - self.target_spacing_start)
                / self.target_spacing_step
            )
            + 1
        )
        self.rotation_step = 1  # in degrees
        self.rotation_start = -90 * (pi / 180)  # in radians
        self.rotation_stop = 90 * (pi / 180)  # in radians
        self.number_of_rotations = (
            int((self.rotation_stop - self.rotation_start) / (self.rotation_step * 0.5))
            + 1
        )
        self.property, self.buildings = gen_shape(
            self.prop_polygon_ar, ng_zones=[self.building_polygon_ar]
        )

    def test_shape_methods(self):

        reference_values = self.reference_values
        area_1 = self.property.get_area()
        area_2 = self.buildings[0].get_area()
        self.assertAlmostEqual(
            area_1, reference_values["test_shape_methods_Area"][0], delta=0.001
        )
        self.assertAlmostEqual(
            area_2, reference_values["test_shape_methods_Area"][1], delta=0.001
        )

        pint_11 = self.property.point_intersect([100.0, 70.0])
        pint_12 = self.property.point_intersect([20.0, 80.0])
        self.assertEqual(
            pint_11, reference_values["test_shape_methods_point_intersections"][0]
        )
        self.assertEqual(
            pint_12, reference_values["test_shape_methods_point_intersections"][1]
        )

        pint_21 = self.buildings[0].point_intersect([100.0, 70.0])
        pint_22 = self.buildings[0].point_intersect([20.0, 80.0])
        self.assertEqual(
            pint_21, reference_values["test_shape_methods_point_intersections"][2]
        )
        self.assertEqual(
            pint_22, reference_values["test_shape_methods_point_intersections"][3]
        )

        shape_1_ex_1 = self.property.line_intersect([60.0, 30.0, 110.0, 130.0])
        shape_1_ex_2 = self.property.line_intersect([60.0, 55.0, 110.0, 50.0])
        shape_1_ex_3 = self.property.line_intersect([20.0, 40.0, 60.0, 40.0])

        def split_points(point_array):
            if len(point_array) > 0:
                p_x = [p[0] for p in point_array]
                p_y = [p[1] for p in point_array]
            else:
                p_x = []
                p_y = []
            return p_x, p_y

        shape_1_ex_1_x, shape_1_ex_1_y = split_points(shape_1_ex_1)
        shape_1_ex_2_x, shape_1_ex_2_y = split_points(shape_1_ex_2)
        shape_1_ex_3_x, shape_1_ex_3_y = split_points(shape_1_ex_3)

        s1e1_ref_x = reference_values["test_shape_methods_Shape_1_r1_x"].to_list()
        s1e1_ref_y = reference_values["test_shape_methods_Shape_1_r1_y"].to_list()
        s1e2_ref_x = reference_values["test_shape_methods_Shape_1_r2_x"].to_list()
        s1e2_ref_y = reference_values["test_shape_methods_Shape_1_r2_y"].to_list()
        s1e3_ref_x = reference_values["test_shape_methods_Shape_1_r3_x"].to_list()
        s1e3_ref_y = reference_values["test_shape_methods_Shape_1_r3_y"].to_list()

        def check_intersections(ref_list, generated_list):
            ref_list_length = np.count_nonzero(~np.isnan(ref_list))
            generated_list_length = len(generated_list)
            self.assertEqual(ref_list_length, generated_list_length)
            if ref_list_length == generated_list_length and generated_list_length > 0:
                for i in range(generated_list_length):
                    self.assertAlmostEqual(ref_list[i], generated_list[i], delta=0.001)

        check_intersections(s1e1_ref_x, shape_1_ex_1_x)
        check_intersections(s1e1_ref_y, shape_1_ex_1_y)
        check_intersections(s1e2_ref_x, shape_1_ex_2_x)
        check_intersections(s1e2_ref_y, shape_1_ex_2_y)
        check_intersections(s1e3_ref_x, shape_1_ex_3_x)
        check_intersections(s1e3_ref_y, shape_1_ex_3_y)

        shape_2_ex_1 = self.buildings[0].line_intersect([60.0, 30.0, 110.0, 130.0])
        shape_2_ex_2 = self.buildings[0].line_intersect([60.0, 55.0, 110.0, 50.0])
        shape_2_ex_3 = self.buildings[0].line_intersect([20.0, 40.0, 60.0, 40.0])

        shape_2_ex_1_x, shape_2_ex_1_y = split_points(shape_2_ex_1)
        shape_2_ex_2_x, shape_2_ex_2_y = split_points(shape_2_ex_2)
        shape_2_ex_3_x, shape_2_ex_3_y = split_points(shape_2_ex_3)

        s2e1_ref_x = reference_values["test_shape_methods_Shape_2_r1_x"].to_list()
        s2e1_ref_y = reference_values["test_shape_methods_Shape_2_r1_y"].to_list()
        s2e2_ref_x = reference_values["test_shape_methods_Shape_2_r2_x"].to_list()
        s2e2_ref_y = reference_values["test_shape_methods_Shape_2_r2_y"].to_list()
        s2e3_ref_x = reference_values["test_shape_methods_Shape_2_r3_x"].to_list()
        s2e3_ref_y = reference_values["test_shape_methods_Shape_2_r3_y"].to_list()

        check_intersections(s2e1_ref_x, shape_2_ex_1_x)
        check_intersections(s2e1_ref_y, shape_2_ex_1_y)
        check_intersections(s2e2_ref_x, shape_2_ex_2_x)
        check_intersections(s2e2_ref_y, shape_2_ex_2_y)
        check_intersections(s2e3_ref_x, shape_2_ex_3_x)
        check_intersections(s2e3_ref_y, shape_2_ex_3_y)

    def test_borehole_config(self):

        target_spacing = (self.target_spacing_start + self.target_spacing_stop) / 2
        rotations = np.linspace(
            self.rotation_start, self.rotation_stop, num=self.number_of_rotations
        )
        nbhs = [
            len(
                gen_borehole_config(
                    self.property,
                    target_spacing,
                    target_spacing,
                    no_go=self.buildings,
                    rotate=rotation,
                )
            )
            for rotation in rotations
        ]
        reference_values = self.reference_values[
            "test_borehole_config_lengths"
        ].to_list()
        for i in range(len(nbhs)):
            self.assertAlmostEqual(reference_values[i], nbhs[i], delta=0.001)

    def test_normal_spacing(self):

        target_spacings = np.linspace(
            self.target_spacing_start,
            self.target_spacing_stop,
            num=self.target_spacing_number,
        )
        nbhs = [
            len(
                field_optimization_fr(
                    ts,
                    self.rotation_step,
                    self.property,
                    ng_zones=self.buildings,
                    rotate_start=self.rotation_start,
                    rotate_stop=self.rotation_stop,
                )[0]
            )
            for ts in target_spacings
        ]
        reference_values = self.reference_values[
            "test_normal_spacing_lengths"
        ].to_list()
        for i in range(len(nbhs)):
            self.assertAlmostEqual(reference_values[i], nbhs[i], delta=0.001)

    def test_perimeter_spacing(self):

        target_spacings = np.linspace(
            self.target_spacing_start,
            self.target_spacing_stop,
            num=self.target_spacing_number,
        )
        nbhs = [
            len(
                field_optimization_wp_space_fr(
                    self.perimeter_spac_ratio,
                    ts,
                    self.rotation_step,
                    self.property,
                    ng_zones=self.buildings,
                    rotate_start=self.rotation_start,
                    rotate_stop=self.rotation_stop,
                )[0]
            )
            for ts in target_spacings
        ]
        reference_values = self.reference_values[
            "test_perimeter_spacing_lengths"
        ].to_list()
        for i in range(len(nbhs)):
            self.assertAlmostEqual(reference_values[i], nbhs[i], delta=0.001)
