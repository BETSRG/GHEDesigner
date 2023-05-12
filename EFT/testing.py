def monthindex(mname):
    months = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5,
              'June': 6, 'July': 7, 'August': 8, 'September': 9, 'October': 10,
              'November': 11, 'December': 12}
    mi = months.get(mname)
    return mi
def monthdays(month):
    if month > 12:
        md = month % 12
    else:
        md = month
    ndays = [31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    monthdays = ndays[md]
    return monthdays
def firstmonthhour(month):
    fmh = 1
    if month > 1:
        for i in range(1, month):
            mi = i % 12
            fmh = fmh + 24 * monthdays(mi)
    return fmh
def lastmonthhour(month):
    lmh = 0
    for i in range(1, month + 1):
        lmh = lmh + monthdays(i) * 24
    if month == 1:
        lmh = 31 * 24
    return lmh
def MonthInfo():
    start = firstmonthhour(monthindex('January'))
    last = lastmonthhour(monthindex('January'))
    print(start, last)
    start = firstmonthhour(monthindex('February'))
    last = lastmonthhour(monthindex('February'))
    print(start, last)
    start = firstmonthhour(monthindex('March'))
    last = lastmonthhour(monthindex('March'))
    print(start, last)
    start = firstmonthhour(monthindex('April'))
    last = lastmonthhour(monthindex('April'))
    print(start, last)
    start = firstmonthhour(monthindex('May'))
    last = lastmonthhour(monthindex('May'))
    print(start, last)
    start = firstmonthhour(monthindex('June'))
    last = lastmonthhour(monthindex('June'))
    print(start, last)
    start = firstmonthhour(monthindex('July'))
    last = lastmonthhour(monthindex('July'))
    print(start, last)
    start = firstmonthhour(monthindex('August'))
    last = lastmonthhour(monthindex('August'))
    print(start, last)
    start = firstmonthhour(monthindex('September'))
    last = lastmonthhour(monthindex('September'))
    print(start, last)
    start = firstmonthhour(monthindex('October'))
    last = lastmonthhour(monthindex('October'))
    print(start, last)
    start = firstmonthhour(monthindex('November'))
    last = lastmonthhour(monthindex('November'))
    print(start, last)
    start = firstmonthhour(monthindex('December'))
    last = lastmonthhour(monthindex('December'))
    print(start, last)

MonthInfo()

#pls = HTS.split('\n')

#Matrix=[]
#i = 0
#for line in pls:
    #if i>= 3:
        #row = line.split(' ')
        #NoSpace = []
        #for cell in row:
            #if cell != '':
                #NoSpace.append(cell)
        #Matrix.append(NoSpace)
    #i += 1



import numpy as np
from copy import deepcopy

# Has 2 Outputs (qExt, qRej)
def GLHEPRO_Model(R, MinEFT, MaxEFT, QH, QC):
    qExt = (R[3] + R[4] * MinEFT + R[5] * (MinEFT ** 2)) * QH
    qRej = (R[0] + R[1] * MaxEFT + R[2] * (MaxEFT ** 2)) * QC

    return qExt, qRej

# Has 2 Outputs (qExt, qRej)
def FixedCOP_Model(COPH, COPC, QH, QC):
    qExt = (1 - (1 / COPH)) * QH
    qRej = ((1 / COPC) + 1) * QC

    return qExt, qRej

# Has 1 Output (converted matrix)
def UnitConversion(matrix, units):
    # if-statements to determine the conversion factor needed to go from current units to kW
    if units == 'SI-kW':
        return matrix
    elif units == 'SI-W':
        ConvFac = 0.001
    elif units == 'SI-MW':
        ConvFac = 1000
    elif units == 'IP-Btuh':
        ConvFac = 0.0002930711
    elif units == 'IP-MBtuh':
        ConvFac = 293.0711

    for row in range(len(matrix)):  # loops through all the rows in the matrix
        for col in range(2, len(matrix[0])):  # skips the first two columns with date and time and runs through the rest
            matrix[row][col] *= ConvFac  # Converts data to Watts

    return matrix

# Has 1 Output (matrix)
def PutinOrder(matrix):
    InOrder = []  # Create empty array to store matrix values in order
    i = 0  # Initializing counter

    while i <= len(matrix):  # Loop keeps running until all the rows from matrix have been added to InOrder
        for j in range(len(matrix)):  # Loops through every row in matrix
            if matrix[j][0] == i + 1:  # Checks if the first value in the row is equal to one higher than the counter
                InOrder.append(matrix[j])  # Adds the matrix row to InOrder
        i += 1  # Increases counter once one row has been added from matrix to InOrder
    return InOrder

def readfiles(HLFile, HPFile ):
    # Unpacks the hourly load data, skips over the line with the units, and puts it into a matrix
    HourlyLoads = np.genfromtxt(HLFile, dtype=None, delimiter=",", encoding=None, skip_header=2, unpack=False)
    # Reads the line with units and puts it into an array
    units = np.genfromtxt(HLFile, dtype=None, delimiter=",", encoding=None, skip_footer=len(HourlyLoads),
                          unpack=False)
    IUnits = units[0][0]  # Assigning the correct string to unit variable
    HourlyLoads = UnitConversion(HourlyLoads, IUnits)  # Converts matrix data into Watts

    # Unpacks the heat pump model data and puts it into a matrix
    HPModelData = np.genfromtxt(HPFile, dtype=None, delimiter=",", encoding=None, unpack=False)
    HPModelData = PutinOrder(HPModelData)  # Puts Heat Pumps in order

    return HourlyLoads, HPModelData

def readEFTfile(EFTFile):
    f1 = open(EFTFile, 'r')  # open the file for reading
    lines = f1.readlines()  # read the entire file as a list of strings
    f1.close()  # close the file

    for line in lines:  # loop over all the lines
        cells = line.strip().split(",")  # create a list of "words"
        # set appropriate cells to variables
        EFTUnits = cells[0].lower().strip()
        MinEFT = float(cells[1])
        MaxEFT = float(cells[2])

    if EFTUnits == 'f':  # Convert EFT values into celsius if they were given in fahrenheit
        MinEFT = (MinEFT - 32) / 1.8
        MaxEFT = (MaxEFT - 32) / 1.8
    return MinEFT, MaxEFT

def GetHourlyRates(HourlyLoads, HPModelData, MinEFT, MaxEFT, Start, Stop):
    # Creates an independent copy of HourlyLoads, so we can just write over the q values and not have to reenter dates
    rows = Stop-Start
    cols = len(HPModelData)*2
    HourlyRates = np.empty((rows,cols))
    i = 0  # Creates a counter to make sure we're pulling QH and QC from the correct columns
    for r in range(len(HPModelData)):  # Loops through all the rows in the Heat Pump Data matrix
        I2 = HPModelData[r][2]  # sets I2 equal to the mode number (1 = Fixed, 2 = GLHEPRO)
        if I2 == 1:  # if in fixed mode sets R1=COPH and R2=COPC
            COPH = HPModelData[r][3]
            COPC = HPModelData[r][4]
        elif I2 == 2:
            R = []  # empty array to append coefficients to (a,b,c,u,v,w)
            for c in range(3, len(HPModelData[r])):  # loops through R1-R6 columns
                R.append(HPModelData[r][c])
        else:  # return none if the user gave an input for I2 that wasn't 1 or 2
            return None
        for row in range(Start, Stop):  # Loops through all the rows in HourlyLoads matrix
            QH = HourlyLoads[row][i + 2]  # sets QH value to correct column using counter
            QC = abs(HourlyLoads[row][i + 3])  # sets QC value to correct column using counter

            # checks which mode heat pump has and sends info to correct function model to get extraction rejection rates
            if I2 == 1:
                qExt, qRej = FixedCOP_Model(COPH, COPC, QH, QC)
            elif I2 == 2:
                qExt, qRej = GLHEPRO_Model(R, MinEFT, MaxEFT, QH, QC)

            HourlyRates[row][i] = qExt  # writes extraction value to corresponding cell in HourlyRates
            HourlyRates[row][i + 1] = qRej  # writes rejection value to corresponding cell in HourlyRates

        i += 2  # increases counter to keep columns for QH,QC correct
    return HourlyRates

def TotalRates(HourlyRates):
    # Creates empty matrix with 2 columns and same number of rows as HourlyRates matrix
    TotalHourlyRates = np.empty((len(HourlyRates), 2))
    for row in range(len(HourlyRates)):  # Loops through all the rows in HourlyRates matrix
        Ext = 0  # initialize counters for combined hourly extraction and rejection
        Rej = 0
        for col in range(len(HourlyRates[0])):  # loop through columns of HourlyRates, but skip date/time columns
            if ((col+1) % 2) != 0:  # checks if column number odd to determine if it's a Rejection value
                Ext += HourlyRates[row][col]  # adds rejection value to counter for this row
            else:  # if column number is even it's an Extraction value
                Rej += HourlyRates[row][col]  # adds extraction value to counter for this row

        TotalHourlyRates[row][1] = Ext  # Add combined hourly extraction rate to third column in the empty matrix
        TotalHourlyRates[row][0] = Rej  # Add combined hourly rejection rate to fourth column in the empty matrix

    # Split TotalHourly Rates matrix into two lists for extraction and rejection
    hourly_extraction_loads = TotalHourlyRates[:, 1]
    hourly_rejection_loads = TotalHourlyRates[:, 0]
    ext = np.array(hourly_extraction_loads).tolist()
    rej = np.array(hourly_rejection_loads).tolist()

    return rej, ext

def Rejection_Extraction_Rates(HourlyLoadFile, HPDataFilie, EFT, Start=0, Stop=None):

    HourlyLoads, HPModelData = readfiles(HourlyLoadFile, HPDataFilie)

    if type(EFT) == str:
        MinEFT, MaxEFT = readEFTfile(EFT)
    else:
        MinEFT = EFT
        MaxEFT = EFT

    if Stop == None:
        Stop = len(HourlyLoads)

    HourlyRates = GetHourlyRates(HourlyLoads, HPModelData, MinEFT, MaxEFT, Start, Stop)
    rej, ext = TotalRates(HourlyRates)

    return rej, ext

#HourlyRej, HourlyExt = Rejection_Extraction_Rates('HLTest1.csv', 'HPTest1.csv', 'EFTTest1.csv')

#print(HourlyExt)
old = [0,1,2,3,4,5,6,7,8,9,10,11,12]
new = [66,77,88]
hi = old[:-1]
print(hi)
HourlyExt = [0] + HourlyExt
Start = 3.7
Stop = 5.6
import math
beg = math.floor(Start)
end = math.ceil(Stop)+1


def gi(lnttsi, lntts, g):
    gi = np.interp(lnttsi, lntts, g, left=0.001,
                   right=100)  # we should perhaps prevent going to the right of the
    # max lntts value by calculating an extrapolated large
    # maximum value
    return gi

def HPEFT(AvgLoads, SST, tsh):
    n = len(AvgLoads)
    # calculate lntts for each time and then find g-function value
    DeltaTBHW = 0
    for j in range(1, n):  # This inner loop sums the responses of all previous step functions
        sftime = SST[n - 1] - SST[j - 1]
        if sftime > 0:
            lnttssf = np.log(sftime / tsh)
            gsf = gi(lnttssf)
            stepfunctionload = 1000 * self.sfLoads[j] / (self.H * self.nbh)  # convert loads from total on GHE
            # in kW to W/m
            DeltaTBHW = DeltaTBHW + gsf * stepfunctionload / (2 * math.pi * self.k_s)
    DTBHW = DeltaTBHW

    # TBHW = Borehole wall temperature
    TBHW = DTBHW + self.ugt
    DT_wall_to_fluid = (1000. * self.AvgLoads[n - 1] / (self.H * self.nbh)) * self.Rb
    # MFT = Simple mean fluid temperature
    MFT = TBHW + DT_wall_to_fluid

    mdotrhocp = self.V_flow_borehole * self.rhoCp_f  # Units of flow rate are L/s
    # Units of rhocp are kJ/m3 K
    # Units of mdotrhocp are then W/K

    half_fluidDT = (1000. * self.AvgLoads[n - 1]) / (mdotrhocp * 2.)
    HPEFT = MFT - half_fluidDT

    return HPEFT

#vals = []
#for i in range(Start+1, Stop+1):
    #vals.append(HourlyExt[i])
#print(vals)
#print(vals[-1])
#print(vals[-2])