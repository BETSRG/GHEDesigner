import pytest

from ghedesigner.tests.test_base_case import GHEBaseTest
from ghedesigner.utilities import read_csv_column

# here is my edit


class TestUtilities(GHEBaseTest):
    def test_read_csv_data(self):
        # test typical csv file
        f_path_loads = self.test_data_directory / "Atlanta_Office_Building_Loads.csv"

        # find by column name
        data = read_csv_column(f_path_loads, column="Hourly heat extraction (W)")
        self.assertEqual(len(data), 8760)
        self.assertEqual(data[0], 0)
        self.assertAlmostEqual(data[5000], -192013, delta=1)
        self.assertEqual(data[-1], 0)

        # find by column index
        data = read_csv_column(f_path_loads, column=0)
        self.assertEqual(len(data), 8760)
        self.assertEqual(data[0], 0)
        self.assertAlmostEqual(data[5000], -192013, delta=1)
        self.assertEqual(data[-1], 0)

        # test irregular csv file
        f_path_data = self.test_data_directory / "rowwise_reference_values.csv"

        # find by column name
        data = read_csv_column(f_path_data, column="test_normal_spacing_target_spacings")
        self.assertEqual(data[0], 10)
        self.assertEqual(data[-1], 20)

        # find by column index
        data = read_csv_column(f_path_data, column=2)
        self.assertEqual(data[0], 10)
        self.assertEqual(data[-1], 20)

        # try passing incorrect column type
        with pytest.raises(TypeError):
            read_csv_column(f_path_data, column=1.5)

        # try non-numeric
        data = read_csv_column(
            f_path_data, column="test_shape_methods_point_intersections", try_convert_to_numeric=False
        )
        self.assertEqual(data[0], "TRUE")
        self.assertEqual(data[-1], "")
