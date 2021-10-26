# Jack C. Cook
# Monday, October 25, 2021

import os
import ghedt.PLAT.pygfunction as gt
import pandas as pd


def create_if_not(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return


def main():
    input_path = '100-400_RowWise_Cases/'
    output_folder = 'RowWisePlots/'
    create_if_not(output_folder)

    files = os.listdir(input_path)

    for i in range(len(files)):

        file = files[i]

        fig = gt.utilities._initialize_figure()
        ax = fig.add_subplot(111)

        d = pd.read_csv(input_path + file).to_dict('list')

        x = d['x']
        y = d['y']

        ax.scatter(x, y)

        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        fig.tight_layout()

        fig.savefig(output_folder + file.split('.')[0] + '.png')


if __name__ == '__main__':
    main()
