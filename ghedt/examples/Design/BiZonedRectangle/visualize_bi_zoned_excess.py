# Jack C. Cook
# Friday, October 29, 2021

import ghedt.PLAT.pygfunction as gt
from ghedt.utilities import js_load


def main():
    file_name = 'bi_zoned_search.json'

    d = js_load(file_name)

    domain = d['Outer_Domain']['Domain']
    searched = d['Outer_Domain']['Searched']

    NBH = domain['nbh']
    TE = domain['T_excess']

    nbh_s = searched['nbh']
    TE_s = searched['T_excess']

    n = list(range(1, len(nbh_s)+1))

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    import matplotlib.pyplot as plt

    sub_axes = plt.axes([.30, .30, .5, .5])
    # plot the zoomed portion
    sub_axes.scatter(nbh_s, TE_s, s=14, c='red')
    sub_axes.set_ylim([-2.5, 2.5])
    sub_axes.set_xlim([100, 200])
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

    fig.tight_layout()

    fig.savefig('bi-zoned-outer-search.png')

    domain = d['Selected_Domain']['Domain']
    searched = d['Selected_Domain']['Searched']

    NBH = domain['nbh']
    TE = domain['T_excess']

    nbh_s = searched['nbh']
    TE_s = searched['T_excess']

    sub_axes.scatter(nbh_s, TE_s, s=14, c='red')

    n = list(range(1, len(nbh_s) + 1))

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    import matplotlib.pyplot as plt

    sub_axes = plt.axes([.30, .30, .5, .5])
    # plot the zoomed portion
    sub_axes.scatter(nbh_s, TE_s, s=14, c='red')
    sub_axes.set_ylim([-2.5, 2.5])
    sub_axes.set_xlim([100, 185])
    sub_axes.grid()
    sub_axes.set_axisbelow(True)

    ax.scatter(nbh_s, TE_s, s=6, c='red', label='Bisection search points')
    for i, txt in enumerate(n):
        ax.annotate(txt, xy=(nbh_s[i], TE_s[i]), xytext=(nbh_s[i], TE_s[i] + 5),
                    size=7)
        sub_axes.annotate(txt, xy=(nbh_s[i], TE_s[i]),
                          xytext=(nbh_s[i], TE_s[i] + 0.5), size=15)
    ax.scatter(NBH, TE, facecolors='none', edgecolors='blue', s=7,
               label='Unimodal list')
    sub_axes.scatter(NBH, TE, facecolors='none', edgecolors='blue', s=17)

    ax.set_xlabel('Number of boreholes')
    ax.set_ylabel('Heat pump excess temperature ($\degree$C)')
    ax.legend()

    fig.tight_layout()

    fig.savefig('bi-zoned_selected_domain.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
