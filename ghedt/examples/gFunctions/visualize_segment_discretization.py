# Jack C. Cook
# Friday, October 29, 2021
import copy

import ghedt.PLAT.pygfunction as gt


def main():
    nSegments = 8
    height_values = [24., 48., 96., 192., 384.]

    segment_ratios = gt.utilities.segment_ratios(nSegments)

    print('{}: {}'.format('Segment Ratios', segment_ratios.tolist()))

    for H in height_values:

        segment_lengths = segment_ratios * H

        print('{}: {}'.format(H, segment_lengths.tolist()))

    H = 100.
    segment_lengths = H * segment_ratios
    segment_lengths = segment_lengths.tolist()

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    for i in range(len(segment_lengths)):
        y = sum(segment_lengths[0:i])
        ax.scatter(0, y)

    plt.show()




if __name__ == '__main__':
    main()
