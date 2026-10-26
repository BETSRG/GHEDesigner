from ghedesigner.tests.modelica_ghe_comparison import (
    compare_against_modelica,
    l_shaped_11_by_10_coordinates,
    rectangular_4_by_5_coordinates,
)


def test_rectangular_field_constant_rejection_matches_modelica():
    compare_against_modelica(
        "case_1_rectangular_constant.csv",
        rectangular_4_by_5_coordinates(),
    )


def test_rectangular_field_load_profile_matches_modelica():
    compare_against_modelica(
        "case_2_rectangular_profile.csv",
        rectangular_4_by_5_coordinates(),
    )


def test_l_shaped_field_constant_rejection_matches_modelica():
    compare_against_modelica(
        "case_3_l_shaped_constant.csv",
        l_shaped_11_by_10_coordinates(),
    )


def test_l_shaped_field_load_profile_matches_modelica():
    compare_against_modelica(
        "case_4_l_shaped_profile.csv",
        l_shaped_11_by_10_coordinates(),
    )
