# Jack C. Cook
# Tuesday, November 16, 2021

import ghedt
from natsort import natsorted
from PIL import Image
import os
import matplotlib.pyplot as plt


def main():
    folder = 'Calculated_Temperature_Fields/'

    file_plot_names = os.listdir(folder)
    # we need to reverse the order to start from less zone to more zone, we'll
    # use natsort to do it
    # # sort the keys by order from largest to smallest
    file_plot_names = natsorted(file_plot_names)

    frames = []
    for file in file_plot_names:
        frames.append(Image.open(folder + file))

    frames[0].save(fp='find_rectangle.gif', format='GIF',
                   append_images=frames[0:] + [frames[-1], frames[-1]],
                   save_all=True, duration=800, loop=0, quality=100)


if __name__ == '__main__':
    main()
