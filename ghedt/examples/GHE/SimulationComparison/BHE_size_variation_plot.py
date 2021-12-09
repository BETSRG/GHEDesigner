# Jack C. Cook
# Tuesday, October 19, 2021
import matplotlib.pyplot as plt
import pandas as pd
import pygfunction as gt
import numpy as np


def error(ref, pred):
    return (pred-ref) / ref * 100.


def main():
    file = 'BHE_size_variation.xlsx'

    xlsx = pd.ExcelFile(file)

    # Define sheet names
    GLHEPro_sheet = 'GLHEPRO'
    GHEDT_hyts_sheet = 'GHEDT (HYTS)'
    GHEDT_hots_sheet = 'GHEDT (HOTS)'

    # Read excel sheets into dictionary
    GLHEPro_d = \
        pd.read_excel(xlsx, sheet_name=GLHEPro_sheet).to_dict('list')
    GHEDT_hyts = \
        pd.read_excel(xlsx, sheet_name=GHEDT_hyts_sheet).to_dict('list')
    GHEDT_hots = \
        pd.read_excel(xlsx, sheet_name=GHEDT_hots_sheet).to_dict('list')

    # Get keys that are not V_flow_borehole
    keys = list(GLHEPro_d.keys())

    GLHEPRO_error = {}
    GHEDT_error = {}
    for key in keys:
        GLHEPRO_error[key] = []
        GHEDT_error[key] = []

    keys.remove('V_flow_borehole')
    tubes = keys

    for i, V_flow_borehole in enumerate(GHEDT_hots['V_flow_borehole']):
        # Append borehole volume flow rates
        GLHEPRO_error['V_flow_borehole'].append(V_flow_borehole)
        GHEDT_error['V_flow_borehole'].append(V_flow_borehole)
        for j, tube in enumerate(tubes):

            ref = GHEDT_hots[tube][i]

            _GLHEPRO_error = error(ref, GLHEPro_d[tube][i])
            _GHEDT_error = error(ref, GHEDT_hyts[tube][i])

            GLHEPRO_error[tube].append(_GLHEPRO_error)
            GHEDT_error[tube].append(_GHEDT_error)

    fig = gt.gfunction._initialize_figure()
    ax = fig.subplots(3, sharex=True, sharey=True)

    ind = np.arange(len(GLHEPro_d['V_flow_borehole']))
    width = 0.35

    for i, tube in enumerate(tubes):

        ax[i].bar(ind, GHEDT_error[tube], width, label=tube + ' (GHEDT)')
        ax[i].bar(ind + width, GLHEPRO_error[tube], width, label=tube + ' (GLHEPro)')

        plt.xticks(ind + width/2, GLHEPro_d['V_flow_borehole'])

        ax[i].legend()
        ax[i].set_ylim([-14, 12])
        ax[i].set_yscale('symlog')
        ax[i].grid()
        ax[i].set_axisbelow(True)

    ax[2].set_xlabel('Volumetric flow rate per borehole (L/s)')
    ax[1].set_ylabel(r'Height Error (%) = $\dfrac{pred-ref}{ref} * 100 \%$')

    fig.tight_layout()

    fig.savefig('size_comparison_bar.png')


if __name__ == '__main__':
    main()
