# Jack C. Cook
# Friday, November 5, 2021

import ghedt.PLAT.pygfunction as gt
import ghedt.utilities
from ghedt.utilities import js_load


def main():
    file_name = 'uniform_rectangular_search.json'

    d = js_load(file_name)

    domain = d['Domain']
    searched = d['Searched']

    NBH = domain['nbh']
    TE = domain['T_excess']

    nbh_s = searched['nbh']
    TE_s = searched['T_excess']

    n = list(range(1, len(nbh_s)+1))

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    import matplotlib.pyplot as plt

    sub_axes = plt.axes([.30, .30, .44, .44])
    # plot the zoomed portion
    sub_axes.scatter(nbh_s, TE_s, s=14, c='red')
    sub_axes.set_ylim([-10, 10])
    sub_axes.set_xlim([40, 300])
    sub_axes.grid()
    sub_axes.set_axisbelow(True)

    ax.scatter(nbh_s, TE_s, s=6, c='red', label='Bisection search points')
    for i, txt in enumerate(n):
        ax.annotate(txt, xy=(nbh_s[i], TE_s[i]), xytext=(nbh_s[i], TE_s[i]+5), size=7)
        sub_axes.annotate(txt, xy=(nbh_s[i], TE_s[i]), xytext=(nbh_s[i], TE_s[i] + 0.5), size=15)
    ax.scatter(NBH, TE, facecolors='none', edgecolors='blue', s=7, label='Unimodal list')
    sub_axes.scatter(NBH, TE, facecolors='none', edgecolors='blue', s=17)

    ax.set_xlabel('Number of boreholes')
    ax.set_ylabel('Heat pump excess temperature ($\degree$C)')
    ax.legend()

    fig.savefig('bisection_search_domain_rectangle.png',
                bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)
    import matplotlib.pyplot as plt

    ax.scatter(NBH, TE, facecolors='none', edgecolors='blue', s=7,
               label='Unimodal list')

    # ax.set_ylim([-20, 50])

    ax.set_yscale('symlog')

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_xlabel('Number of boreholes')
    ax.set_ylabel('Heat pump excess temperature ($\degree$C)')
    ax.legend()

    fig.tight_layout()

    fig.savefig('square_unimodal_domain.png')
    plt.close(fig)

    delta_T_values, unimodal = ghedt.utilities.verify_excess(TE)
    indices = list(range(1, len(delta_T_values)+1))
    print('Unimodal domain: {}'.format(unimodal))

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)

    ax.plot(indices, delta_T_values)

    ax.grid()
    ax.set_axisbelow(True)

    ax.set_ylim([-450, 0.])

    ax.set_yscale('symlog')

    yaxis = plt.gca().yaxis
    yaxis.set_minor_locator(ghedt.utilities.MinorSymLogLocator(1e-1))

    ax.set_ylabel('$\Delta T_j = T_{i} - T_{i-1}$')
    ax.set_xlabel('Index, $j$')

    fig.tight_layout()

    fig.savefig('rectangle_unimodal_proof.png')


if __name__ == '__main__':
    main()
