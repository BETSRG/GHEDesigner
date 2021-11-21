# Jack C. Cook
# Friday, November 12, 2021

import ghedt
from natsort import natsorted
from PIL import Image
import os
import matplotlib.pyplot as plt


def main():
    folder = 'near-square-fields/'
    create_if_not(folder)

    B = 5
    coordinates_domain = ghedt.domains.square_and_near_square(1, 32, B)

    i = int(len(coordinates_domain) / 2)

    for j in range(i):

        coordinates = coordinates_domain[j]

        fig, ax = ghedt.coordinates.visualize_coordinates(coordinates)

        ax.set_xlim([-3, 83])
        ax.set_ylim([-3, 83])

        file_name = str(j).zfill(2)

        fig.savefig(folder + file_name + '.png', bbox_inches='tight',
                    pad_inches=0.1)

        plt.close(fig)

    file_plot_names = os.listdir(folder)
    # we need to reverse the order to start from less zone to more zone, we'll
    # use natsort to do it
    # # sort the keys by order from largest to smallest
    file_plot_names = natsorted(file_plot_names)

    frames = []
    for file in file_plot_names:
        frames.append(Image.open(folder + file))

    frames[0].save(fp='near-square.gif', format='GIF', append_images=frames[0:],
                   save_all=True, duration=800, loop=0, quality=100)


def create_if_not(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    main()
