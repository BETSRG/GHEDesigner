import numpy as np

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
    HPModel = np.genfromtxt(HPFile, dtype=None, delimiter=",", encoding=None, unpack=False)
    HPModelData = HPModel[1:]
    if len(HPModelData) != 1:
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
        j=0
        for row in range(Start, Stop):  # Loops through all the rows in HourlyLoads matrix
            QH = HourlyLoads[row][i + 2]  # sets QH value to correct column using counter
            QC = abs(HourlyLoads[row][i + 3])  # sets QC value to correct column using counter

            # checks which mode heat pump has and sends info to correct function model to get extraction rejection rates
            if I2 == 1:
                qExt, qRej = FixedCOP_Model(COPH, COPC, QH, QC)
            elif I2 == 2:
                qExt, qRej = GLHEPRO_Model(R, MinEFT, MaxEFT, QH, QC)

            HourlyRates[j][i] = qExt  # writes extraction value to corresponding cell in HourlyRates
            HourlyRates[j][i + 1] = qRej  # writes rejection value to corresponding cell in HourlyRates
            j +=1
        i += 2  # increases counter to keep columns for QH,QC correct
    return HourlyRates

def TotalRates(HourlyRates):
    # Creates empty matrix with 2 columns and same number of rows as HourlyRates matrix
    TotalHourlyRates = np.empty((len(HourlyRates), 2))
    for row in range(len(HourlyRates)):  # Loops through all the rows in HourlyRates matrix
        Ext = 0  # initialize counters for combined hourly extraction and rejection
        Rej = 0
        for col in range(len(HourlyRates[0])):  # loop through columns of HourlyRates
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

def Rejection_Extraction_Rates(HourlyLoadFile, HPDataFilie, MinEFT, MaxEFT, Start=0, Stop=None, Mode = 0):

    HourlyLoads, HPModelData = readfiles(HourlyLoadFile, HPDataFilie)

    #if type(EFT) == str:
        #MinEFT, MaxEFT = readEFTfile(EFT)
    #else:
        #MinEFT = EFT
        #MaxEFT = EFT

    if Stop == None:
        Stop = len(HourlyLoads)

    HourlyRates = GetHourlyRates(HourlyLoads, HPModelData, MinEFT, MaxEFT, Start, Stop)
    rej, ext = TotalRates(HourlyRates)

    if Mode == 0:
        return rej, ext
    else:
        return rej, ext, HourlyRates

