# Jack C. Cook
# Sunday, November 7, 2021

import ghedt.PLAT.pygfunction as gt
import numpy as np


def plot_borehole(segment_ratios, ax, width=0.2, origin=(0., 0.)):
    # Plot the top and outer location
    box = [[origin[0], origin[1]], [origin[0], 1], [origin[0] + width, 1],
           [origin[0] + width, origin[1]], [origin[0], origin[1]]]
    x, y = list(zip(*box))
    ax.plot(x, y, color='k')

    print(len(segment_ratios))

    summation = 0
    for i in range(len(segment_ratios)):
        summation += segment_ratios[i]
        ax.plot([origin[0], origin[0]+width],
                [summation, summation], color='k')

    return


def main():
    width = 0.2
    origin = (0., 0.)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    # Eskilson discretization

    nSegments = 12
    m = int(nSegments / 2)

    num = np.sqrt(2.) - 1.
    den = (np.sqrt(2.) ** m) - 1.
    alpha = 0.5 * num / den

    segment_ratios = [(np.sqrt(2.)**i) * alpha for i in range(m)]
    segment_ratios = segment_ratios + list(reversed(segment_ratios))
    segment_ratios = np.array(segment_ratios)

    plot_borehole(segment_ratios, ax, width=width, origin=origin)

    ax.text(0, 1.05, s='Eskilson'.center(12), fontsize=15)

    # Equal segment lengths

    segment_ratios = np.array([1 / 12] * 12)

    plot_borehole(segment_ratios, ax, width=width, origin=(width*2, 0))

    ax.text(width*2, 1.05, s='Cimmino'.center(11), fontsize=15)

    # Cimmino discretization
    nSegments = 8

    segment_ratios = gt.utilities.segment_ratios(nSegments)

    plot_borehole(segment_ratios, ax, width=width, origin=(width*4, 0))

    ax.text(width * 4, 1.05, s='(Current)'.center(11), fontsize=15)

    fig.tight_layout()

    fig.savefig('borehole_discretization.png')


if __name__ == '__main__':
    main()
