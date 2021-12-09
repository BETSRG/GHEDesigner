# Jack C. Cook
# Sunday, November 21, 2021

import ghedt.PLAT.pygfunction as gt
import os

import ghedt.utilities


def main():
    folder = 'SimulationTimings/'

    files = os.listdir(folder)

    hybrid_times = []
    hourly_times = []
    years = []

    for i in range(len(files)):
        file = folder + files[i]
        data = ghedt.utilities.js_load(file)
        hourly_time = data['hourly_time']
        hybrid_time = data['hybrid_time']

        year = int(files[i].split('.')[0])

        hybrid_times.append(hybrid_time)
        hourly_times.append(hourly_time)
        years.append(year)

    fig = gt.utilities._initialize_figure()
    ax = fig.add_subplot(111)
    gt.utilities._format_axes(ax)

    ax.scatter(years, hourly_times, label='Hourly Simulation')
    ax.scatter(years, hybrid_times, label='Hybrid Simulation')

    ax.set_xlabel('Simulation time in years')
    ax.set_ylabel('Computing time for simulation (s)')

    ax.set_yscale('log')

    ax.grid()
    ax.set_axisbelow(True)

    fig.tight_layout()

    fig.legend(bbox_to_anchor=(0.375, 0.95))

    fig.savefig('range_years_sim_clock.png')


if __name__ == '__main__':
    main()
