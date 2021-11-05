# Jack C. Cook
# Thursday, November 4, 2021

import os

import json
import numpy as np
import ghedt.PLAT.pygfunction as gt


def compute_rmse(measured, truth):
    rmse = np.linalg.norm(measured - truth) / np.sqrt(len(truth))
    return rmse


def compute_mpe(actual: list, predicted: list) -> float:
    """
    The following mean percentage error formula is used:
    .. math::
        MPE = \dfrac{100\%}{n}\sum_{i=0}^{n-1}\dfrac{a_t-p_t}{a_t}
    Parameters
    ----------
    actual: list
        The actual computed g-function values
    predicted: list
        The predicted g-function values
    Returns
    -------
    **mean_percent_error: float**
        The mean percentage error in percent
    """
    # the lengths of the two lists should be the same
    assert len(actual) == len(predicted)
    # create a summation variable
    summation: float = 0.
    for i in range(len(actual)):
        summation += (predicted[i] - actual[i]) / actual[i]
    mean_percent_error = summation * 100 / len(actual)
    return mean_percent_error


def js_r(filename: str):
    with open(filename) as f_in:
        return json.load(f_in)


def main():
    g_function_folder = 'Calculated_g_Functions/'

    field_type = 'RowWise'

    reference_folder = '96_Equal_Segments_Similarities_MIFT'
    predicted_folders = ['12_Equal_Segments_Similarities_MIFT',
                         '12_Equal_Segments_Equivalent_MIFT',
                         '8_Unequal_Segments_Equivalent_MIFT',
                         '12_Equal_Segments_Similarities_UBWT']

    reference_heights_location = \
        g_function_folder + field_type + '/'

    height_values = os.listdir(reference_heights_location)

    # height_values = ['384m_Depth']

    for k in range(len(height_values)):
        print('Height: {}'.format(height_values[k]))

        height = height_values[k].split('_')[0].replace('m', '')

        height_value_location = \
            reference_heights_location + height_values[k] + '/'

        reference_files_location = \
            height_value_location + reference_folder + '/'

        reference_files = os.listdir(reference_files_location)

        fig = gt.gfunction._initialize_figure()
        ax = fig.add_subplot(111)
        gt.utilities._format_axes(ax)

        markers = ['o', 's', '*', 'x']

        for i in range(len(predicted_folders)):

            predicted_folder_i = predicted_folders[i]
            predicted_files_location_i = \
                height_value_location + predicted_folder_i + '/'
            plotting_data = {predicted_folder_i: {'nbh': [], 'RMSE': []}}

            key = '5.0_96.0_0.075_2.0'
            key = '5.0_{}.0_0.075_2.0'.format(height)

            for j in range(len(reference_files)):
                file_name = reference_files[j]

                reference_data = \
                    js_r(reference_files_location + file_name)
                predicted_data = \
                    js_r(predicted_files_location_i + file_name)
                try:
                    g_ref = np.array(reference_data['g'][key])
                    g_pred = np.array(predicted_data['g'][key])
                except:
                    key = '5.0_{}_0.075_2.0'.format(height)
                    g_ref = np.array(reference_data['g'][key])
                    g_pred = np.array(predicted_data['g'][key])

                nbh = len(reference_data['bore_locations'])

                # rmse = compute_rmse(g_pred, g_ref)
                rmse = compute_mpe(g_ref, g_pred)

                print('{}\t{}'.format(file_name, nbh))

                plotting_data[predicted_folder_i]['nbh'].append(nbh)
                plotting_data[predicted_folder_i]['RMSE'].append(rmse)

            nbh_values = plotting_data[predicted_folder_i]['nbh']
            rmse_values = plotting_data[predicted_folder_i]['RMSE']

            _label = predicted_folder_i.replace('_', ' ').replace('MIFT',
                                                                  'UIFT')
            _label = _label.replace('12 Equal Segments', '12ESL').replace(
                '8 Unequal Segments', '8USL')

            ax.scatter(nbh_values, rmse_values, label=_label, marker=markers[i])

        ax.set_xlabel('Number of boreholes')
        ax.set_ylabel('Mean Percent Error = '
                      r'$\dfrac{\mathbf{p} - \mathbf{r}}{\mathbf{r}} \;\; '
                      r'\dfrac{100\%}{n} $')

        ax.grid()
        ax.set_axisbelow(True)

        fig.legend(bbox_to_anchor=(.83, 1.01), ncol=2)

        fig.tight_layout(rect=(0, 0, 1, 0.93))

        fig.savefig('accuracy_comparison_{}_{}.png'.format(field_type, height))


if __name__ == '__main__':
    main()
