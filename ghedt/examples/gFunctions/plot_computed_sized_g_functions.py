# Jack C. Cook
# Tuesday, October 26, 2021

# Purpose: Plot the sizing results and error in sizing compared to the reference
# case after running the `computed_g_function_sim_and_size.py` file.

import pandas as pd
import ghedt.pygfunction as gt
import numpy as np
import matplotlib.pyplot as plt


def percent_error(ref, pred):
    return (pred - ref) / ref * 100.


def main():
    file = 'Sized_Computed_g_Functions.xlsx'

    d = pd.read_excel(file).to_dict('list')

    calculation_info = d['Unnamed: 0']

    years = [10, 20, 30]

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    # ind = np.arange(len(GLHEPro_d['V_flow_borehole']))
    ind = np.array(years)
    width = 1

    for i in range(len(calculation_info)):
        calculation_specific = calculation_info[i].replace('_', ' ')
        sized_height = []
        for j in range(len(years)):
            year = years[j]
            sized_height_j = d[year][i]
            sized_height.append(sized_height_j)
            # ax.bar(ind-width/2-2*width+i*width, sized_height_j, width)
        rects = ax.bar(ind - width / 2 - 2 * width + i * width, sized_height,
                       width, color='C' + str(0 + i),
                       label=calculation_specific)

    ax.set_xticks(ind)
    ax.tick_params(
        axis='y', which='both', direction='in',
        bottom=True, top=True, left=True, right=True)
    from matplotlib.ticker import AutoMinorLocator
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_xlabel('Design period (years)')
    ax.set_ylabel('Height of each borehole (m)')

    ax.grid()

    ax.set_axisbelow(True)

    fig.legend(bbox_to_anchor=(1., 1.005), ncol=2)

    fig.tight_layout(rect=(0, 0, 1, .89))

    fig.savefig('sizing_comparison_bar.png')

    plt.close(fig)

    reference_case = '96_Equal_Segments_Similarities_UIFT'
    reference_idx = calculation_info.index(reference_case)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)

    # ind = np.arange(len(GLHEPro_d['V_flow_borehole']))
    ind = np.array(years)
    width = 1

    for i in range(len(calculation_info)):
        if i == reference_idx:
            continue
        calculation_specific = calculation_info[i].replace('_', ' ')
        error = []
        for j in range(len(years)):
            year = years[j]
            ref = d[year][reference_idx]
            percent_error_i = percent_error(ref, d[year][i])
            error.append(percent_error_i)
            # ax.bar(ind-width/2-2*width+i*width, sized_height_j, width)
        rects = ax.bar(ind - 2 * width + i * width, error,
                       width, color='C' + str(0 + i),
                       label=calculation_specific)

    ax.set_yticks(np.arange(-2, 8, 1))
    ax.set_xticks(ind)
    ax.tick_params(
        axis='y', which='both', direction='in',
        bottom=True, top=True, left=True, right=True)
    from matplotlib.ticker import AutoMinorLocator
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    ax.set_xlabel('Design period (years)')
    ax.set_ylabel('Error in sized height (%)')

    fig.legend(bbox_to_anchor=(0.95, 1.005), ncol=2)

    ax.grid()

    ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, .89))

    fig.savefig('sizing_comparison_error_bar.png')


if __name__ == '__main__':
    main()
