# Jack C. Cook
# Wednesday, October 20, 2021

import pandas as pd
import gFunctionDatabase as gfdb
import json


def create_database_file(B_values, H_values, rb_values, D_values, g_values,
                         bore_locations, log_time):
    d_out = {'g': {}}

    for i in range(len(B_values)):
        B = B_values[i]
        H = H_values[i]
        rb = rb_values[i]
        D = D_values[i]
        key = '{}_{}_{}_{}'.format(B, H, rb, D)
        d_out['g'][key] = g_values[i]

    d_out['bore_locations'] = bore_locations
    d_out['logtime'] = log_time

    return d_out


def main():
    file_name = 'GLHEPro_gFunctions_10x12.xlsx'

    xlsx = pd.ExcelFile(file_name)
    sheet_name = xlsx.sheet_names[0]
    d_in = pd.read_excel(xlsx, sheet_name=sheet_name).to_dict('list')

    # Logarithmic times
    log_time = d_in['lntts']
    del d_in['lntts']

    # Height values
    H_values = []
    g_values = []
    for height in d_in:
        H_values.append(height)
        g_values.append(d_in[height])

    rb_values = [0.075] * len(H_values)
    D_values = [5.] * len(H_values)
    B_values = [5.] * len(H_values)

    # Get bore locations
    N = 10
    M = 12
    configuration = 'rectangle'

    # GFunction
    # ---------
    # Access the database for specified configuration
    r = gfdb.Management.retrieval.Retrieve(configuration)
    # There is just one value returned in the unimodal domain for rectangles
    r_unimodal = r.retrieve(N, M)
    key = list(r_unimodal.keys())[0]
    r_data = r_unimodal[key]
    bore_locations = r_data['bore_locations']

    g_data = create_database_file(B_values, H_values, rb_values, D_values,
                                  g_values, bore_locations, log_time)

    with open('GLHEPRO_gFunctions.json', 'w+') as fp:
        json.dump(g_data, fp, indent=4)

    # --------------------------------------------------------------------------
    file_name = 'GLHEPro_gFunctions_12x13.xlsx'

    xlsx = pd.ExcelFile(file_name)
    sheet_name = xlsx.sheet_names[0]
    d_in = pd.read_excel(xlsx, sheet_name=sheet_name).to_dict('list')

    # Logarithmic times
    log_time = d_in['lntts']
    del d_in['lntts']

    # Height values
    H_values = []
    g_values = []
    for height in d_in:
        H_values.append(height)
        g_values.append(d_in[height])

        # Get bore locations
    N = 12
    M = 13
    configuration = 'rectangle'

    # GFunction
    # ---------
    # Access the database for specified configuration
    r = gfdb.Management.retrieval.Retrieve(configuration)
    # There is just one value returned in the unimodal domain for rectangles
    r_unimodal = r.retrieve(N, M)
    key = list(r_unimodal.keys())[0]
    r_data = r_unimodal[key]
    bore_locations = r_data['bore_locations']

    g_data = create_database_file(B_values, H_values, rb_values, D_values,
                                  g_values, bore_locations, log_time)

    with open('GLHEPRO_gFunctions_12x13.json', 'w+') as fp:
        json.dump(g_data, fp, indent=4)


if __name__ == '__main__':
    main()
