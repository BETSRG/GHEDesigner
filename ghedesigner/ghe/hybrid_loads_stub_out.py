def step_1():
    #run heatpump with constant COPs to convert building loads to ground loads.
    #these are hourly loads
    #
    #inputs: building_loads, COPc, COPh
    #output: ground_loads
    pass

def step_1point5():
    #parse out ground loads into peak and average for each month of the year
    #inputs: ground_loads
    #outputs: monthly_average_ground_loads (12), monthly_peak_ground_loads (12)
    pass

def step_2():
    # since borehole config and size is supposedly unknown,
    # Assume a 100m deep single borehole and 40W/m peak
    #  load and normalize to this peak (4000W max)
    #
    # inputs: ground_loads
    #outputs: normalized_ground_loads
    pass

def step_3():
    # take the normalized ground loads and use
    # a gfunction and the other ground heat exchanger parameters to get the hourly
    # exiting fluid temperature from the ghe (same as entering to HP)
    #
    # inputs: normalized ground loads, ground heat exchanger parameters
    #output: hourly exiting fluid temperature (ExFT) from the GHE
    pass

def step_4():
    #find the 4 highest peak exiting fluid temperatures for the year for cooling
    # only one peak allowed per month
    # note the time that the peak occurs

    # inputs: hourly_ExFTs
    #outputs: 4 peak ExFTs, associated month, associated hour of month
    pass

def step_5():
    #repeat step 4 but for heating
    #looking for lowest ExFTs instead, 4 month peaks and times
    #
    # inputs: hourly_ExFTs
    #outputs: 4 min ExFTs, associated month, associated hour of the month
    pass

def step_6():
    # figure out the peak load duration that allows for a
    # hybrid time step simulation's ExFT to match the hourly time step's peak ExFT
    # hourly sim's ExFT will be called ExFT_target
    # using the peak load for the month and the peak ExFT
    # assign the peak load of that month to a one hour duration and the average
    # monthly load for the rest of the month.
    # run ExFT simulation
    # if ExFT_hybrid < ExFT_target
        # add 1 hour to the duration of the peak load
        #recalculate the average monthly load to maintain energy balence
    #repeat untin ExFT_hybrid >= ExFT_target
    #record load output for the month
    #
    # inputs: peak_load_of_month_n, peak_ExFT_of_month, peak_ExFT_hour,
    #
    #outputs: load amount and duration for that month(see below for example)
    #           [time,           load]
    #         [hrs 0:300,  average_load_for_month ]
    #         [hrs 301:305, peak_load_for_month]
    #         [hrs 306:720,  average_load_for_month ]
    pass
def step_7():
    # repeat step 6 for remaining heating and cooling peaks
    # run though all 8 months
    #
    #input: 4 min ExFTs, associated month, associated hour of the month,
    #       4 max ExFTs, associated month, associated hour of month
    #
    #outputs: hybridloads_heating_month1, hybrid_timesteps_month1
    #           hybrid_loads_heating_month2, hybrid_timesteps_month2...
    #          ...hybrid_loads_cooling_month8, hybrid_timestep_month8
    pass

def step_8():
    #compile new hybrid loads
    # use monthly average load for all non-peak months
    #inputs: all 12 month's new hybrid load profiles
    #outputs: hybrid_loads_for_the_year, durations
    #example
    # [320 W, 0:60 hrs] (jan)
    # [1600 W, 60:66 hrs] (jan peak)
    # [320 W, 67:720 hrs] (jan)
    # [270 w, 721:789 hrs] (feb)
    # [...] (feb_peak)

    pass

def main():

    #run step 1
    step_1()

if __name__ == "__main__":
    main()
