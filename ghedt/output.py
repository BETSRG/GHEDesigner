import csv
import math
import os
from datetime import datetime

import numpy as np

from ghedt.peak_load_analysis_tool import borehole_heat_exchangers as BHE


def createTitle(allocatedWidth, title, fillerSymbol=" "):
    return "{:{fS}^{L}s}\n".format(" " + title + " ", L=allocatedWidth, fS=fillerSymbol)


def createRow(allocatedWidth, rowData, dataFormats, centering=">"):
    rS = ""
    nCols = len(rowData)
    colWidth = int(allocatedWidth / nCols)
    leftOver = float(allocatedWidth) % colWidth
    for i in range(len(dataFormats)):
        dF = dataFormats[i]
        data = rowData[i]
        width = colWidth
        if leftOver > 0:
            width = colWidth + 1
            leftOver -= 1
        try:
            rS += "{:{c}{w}{fm}}".format(data, c=centering, w=width, fm=dF)
        except:
            print("Ouput Row creation error: ", dF)
            raise (ValueError)

    rS += "\n"
    return rS


def createTable(
    title, colTitles, rows, allowcatedWidth, colFormats, fillerSymbol=" ", centering=">"
):
    nCols = len(colTitles[0])
    rS = ""
    rS += createTitle(allowcatedWidth, title, fillerSymbol=fillerSymbol)
    blankLine = createLine(allowcatedWidth)
    rS += blankLine
    headerFormat = ["s"] * nCols
    for colT in colTitles:
        rS += createRow(allowcatedWidth, colT, headerFormat, centering="^")
    rS += blankLine
    for row in rows:
        rS += createRow(allowcatedWidth, row, colFormats, centering=centering)
    rS += blankLine
    return rS


def createDRow(rowAllocation, Entry1, Entry2, dType1, dType2, bTabs=0, aTabs=0):
    tabWidth = 8
    tabOffset = 0.5 * tabWidth
    nTabs = bTabs + aTabs
    intitialRatio = 0.5
    # reducedAllocation = rowAllocation-nTabs*tabWidth
    rightOffset = intitialRatio * rowAllocation
    leftOffset = (1 - intitialRatio) * rowAllocation

    rightOffset = int(rightOffset - nTabs * tabOffset)
    leftOffset = int(leftOffset - tabOffset * nTabs)
    if (rightOffset + leftOffset + nTabs * tabWidth) != rowAllocation:
        rightOffset += 1

    lNeeded = len(str(Entry1))
    rNeeded = len(str(Entry2))

    if (lNeeded + rNeeded) > (rightOffset + leftOffset):
        print("Allocation: ", rowAllocation)
        print("Characters Needed: ", (lNeeded + rNeeded + nTabs * tabWidth))
        print("Allocated: ", (rightOffset + leftOffset + tabWidth * nTabs))
        print("Right Offset Allocated: ", rightOffset)
        print("Left Offset Allocated: ", leftOffset)
        print("Tab Space: ", tabWidth * nTabs)
        raise Exception("Not Enough Width Was Provided")
    if lNeeded > leftOffset:
        swing = lNeeded - leftOffset
        leftOffset += swing
        rightOffset -= swing
    if rNeeded > rightOffset:
        swing = rNeeded - rightOffset
        rightOffset += swing
        leftOffset -= swing
    if (rightOffset + leftOffset + tabWidth * nTabs) != rowAllocation:
        print("Allocation: ", rowAllocation)
        print("Allocated: ", (rightOffset + leftOffset + tabWidth * nTabs))
        print("Right Offset Allocated: ", rightOffset)
        print("Left Offset Allocated: ", leftOffset)
        print("Tab Space: ", tabWidth * nTabs)
        raise Exception("Width Allocation Error")
    rS = ""
    for t in range(bTabs):
        rS += "\t"
    rS += "{:<{lO}{f1}}{:>{rO}{f2}}".format(
        Entry1, Entry2, lO=leftOffset, rO=rightOffset, f1=dType1, f2=dType2
    )
    for t in range(aTabs):
        rS += "\t"
    rS += "\n"
    return rS


def createLine(rowAllocation, character="*"):
    return character * rowAllocation + "\n"


def hoursToMonth(hours):
    daysInYear = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hoursInYear = 24 * daysInYear
    nYears = math.floor(hours / np.sum(hoursInYear))
    fracMonth = nYears * len(daysInYear)
    monthInYear = 0
    for i in range(len(daysInYear)):
        hoursLeft = hours - nYears * np.sum(hoursInYear)
        if np.sum(hoursInYear[0 : i + 1]) >= hoursLeft:
            monthInYear = i
            break
    # print("Year Months: ",fracMonth)
    # print("Month Months: ",monthInYear)
    fracMonth += monthInYear
    hL = hours - nYears * np.sum(hoursInYear) - np.sum(hoursInYear[0:monthInYear])
    fracMonth += hL / (hoursInYear[monthInYear])
    # print("Hour Months: ",hL/(hoursInYear[monthInYear]))
    # print(fracMonth)
    return fracMonth


def GHETimeConvert(hours):
    daysInYear = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    hoursInYear = 24 * daysInYear
    monthInYear = 0
    yearHourSum = 0
    for i in range(len(daysInYear)):
        hoursLeft = hours
        cM = i
        if yearHourSum + hoursInYear[cM] - 1 >= hoursLeft:
            monthInYear = i
            break
        else:
            yearHourSum += hoursInYear[cM]
    # print("Year Months: ",fracMonth)
    # print("Month Months: ",monthInYear)
    hL = hours - np.sum(hoursInYear[0:monthInYear])
    dayInMonth = int(math.floor(hL / 24)) + 1
    hourInDay = hL % 24 + 1
    # print("Hour Months: ",hL/(hoursInYear[monthInYear]))
    # print(fracMonth)
    return monthInYear + 1, dayInMonth, hourInDay


def OutputDesignDetails(
    design,
    time,
    projectName,
    notes,
    author,
    modelName,
    allocatedWidth=100,
    roundingAmount=10,
    summaryFile="SimulationSummary.txt",
    csvF1="TimeDependentValues.csv",
    csvF2="BoreFieldData.csv",
    csvF3="Loadings.csv",
    csvF4="Gfunction.csv",
    loadMethod="hybrid",
    outputDirectory="",
):
    try:
        ghe = design.ghe
    except:
        ghe = design
    bhe = ghe.bhe
    gfunction = ghe.GFunction
    bH = bhe.b
    b = gfunction.bore_locations

    floatFormat = ".3f"
    stringFormat = "s"
    intFormat = ".0f"
    # roundingAmount = 10
    sciFormat = ".3e"

    blankLine = createLine(allocatedWidth)
    emptyLine = createLine(allocatedWidth, character=" ")
    oS = ""
    # oS += middleSpacingString.format("Project Name:",projectName,rO=rightOffset,lO=leftColLength) + "\n"
    oS += createDRow(
        allocatedWidth, "Project Name:", projectName, stringFormat, stringFormat
    )
    oS += blankLine
    oS += "Notes:\n\n" + notes + "\n"
    oS += blankLine
    oS += createDRow(
        allocatedWidth, "File/Model Name:", modelName, stringFormat, stringFormat
    )
    now = datetime.now()
    timeString = now.strftime("%m/%d/%Y %H:%M:%S %p")
    oS += createDRow(
        allocatedWidth, "Simulated On:", timeString, stringFormat, stringFormat
    )
    oS += createDRow(
        allocatedWidth, "Simulated By:", author, stringFormat, stringFormat
    )
    oS += createDRow(
        allocatedWidth,
        "Calculation Time, s:",
        round(time, roundingAmount),
        stringFormat,
        floatFormat,
    )
    oS += emptyLine
    oS += createTitle(allocatedWidth, "Design Selection", fillerSymbol="-")

    designHeader = [
        ["Field", "Excess Temperature", "Max Temperature", "Min Temperature"],
        [" ", "(C)", "(C)", "(C)"],
    ]
    try:
        designValues = design.searchTracker
    except:
        designValues = ""
    designFormats = ["s", ".3f", ".3f", ".3f"]

    oS += createTable(
        "Field Search Log",
        designHeader,
        designValues,
        allocatedWidth,
        designFormats,
        fillerSymbol="-",
        centering="^",
    )

    oS += emptyLine
    oS += createTitle(allocatedWidth, "GHE System", fillerSymbol="-")

    # GFunction LTS Table
    gfunctionTableFormats = [".3f"]
    gfTableFF = [".3f"] * (len(gfunction.g_lts) + 1)
    gfunctionTableFormats.extend(gfTableFF)
    gfunctionColTitles = ["ln(t/ts)"]

    for gfunctionName in list(gfunction.g_lts):
        gfunctionColTitles.append("H:" + str(round(gfunctionName, 0)) + "m")
    gfunctionColTitles.append("H:" + str(round(bH.H, 2)) + "m")

    gfunctionData = []
    gheGF = gfunction.g_function_interpolation(float(ghe.B_spacing) / bH.H)[0]
    for i in range(len(gfunction.log_time)):
        gfRow = []
        gfRow.append(gfunction.log_time[i])
        for gfunctionName in list(gfunction.g_lts):
            # print(gfunction.g_lts[gfunctionName][i])
            gfRow.append(gfunction.g_lts[gfunctionName][i])
        gfRow.append(gheGF[i])
        gfunctionData.append(gfRow)

    oS += createTable(
        "GFunction LTS Values",
        [gfunctionColTitles],
        gfunctionData,
        allocatedWidth,
        gfunctionTableFormats,
        fillerSymbol="-",
        centering="^",
    )
    oS += emptyLine

    """

    """

    oS += "------ System parameters ------" + "\n"
    oS += createDRow(
        allocatedWidth, "Active Borehole Length, m:", bH.H, stringFormat, intFormat
    )
    oS += createDRow(
        allocatedWidth,
        "Borehole Radius, m:",
        round(bH.r_b, roundingAmount),
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Borehole Spacing, m:",
        round(ghe.B_spacing, roundingAmount),
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Total Drilling, m:",
        round(bH.H * len(b), roundingAmount),
        stringFormat,
        floatFormat,
    )

    indentedAmount = 2

    oS += "Field Geomtetry: " + "\n"
    # rightAd = rightOffset-indentedAmount*tabOffset+math.ceil(indentedAmount/2)
    # leftAd = leftColLength-tabOffset*indentedAmount+math.floor(indentedAmount/2)
    oS += createDRow(
        allocatedWidth,
        "Field Type:",
        ghe.fieldType,
        stringFormat,
        stringFormat,
        bTabs=indentedAmount,
    )
    # oS += middleSpacingIndentedString.format("\t\tField Type:",ghe.fieldType,rO=rightAd,lO=leftAd)
    oS += createDRow(
        allocatedWidth,
        "Field Specifier:",
        ghe.fieldSpecifier,
        stringFormat,
        stringFormat,
        bTabs=indentedAmount,
    )
    # oS += middleSpacingIndentedString.format("\t\tField Specifier:",ghe.fieldSpecifier,rO=rightAd,lO=leftAd)
    oS += createDRow(
        allocatedWidth, "NBH:", len(b), stringFormat, intFormat, bTabs=indentedAmount
    )
    # oS += middleSpacingIndentedString.format("\t\tNBH:",len(b),rO=rightAd,lO=leftAd)
    # Field NBH Borehole locations, field identification
    # System Details

    oS += "Borehole Information: " + "\n"
    oS += createDRow(
        allocatedWidth,
        "Shank Spacing, m:",
        round(bhe.pipe.s, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )

    if isinstance(bhe.pipe.r_out, float):
        oS += createDRow(
            allocatedWidth,
            "Pipe Outer Radius, m:",
            round(bhe.pipe.r_out, roundingAmount),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )
        oS += createDRow(
            allocatedWidth,
            "Pipe Inner Radius, m:",
            round(bhe.pipe.r_in, roundingAmount),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )
    else:
        oS += createDRow(
            allocatedWidth,
            "Outer Pipe Outer Radius, m:",
            round(bhe.pipe.r_out[0], roundingAmount),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )
        oS += createDRow(
            allocatedWidth,
            "Inner Pipe Outer Pipe Outer Radius, m:",
            round(bhe.pipe.r_out[1], roundingAmount),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )
        oS += createDRow(
            allocatedWidth,
            "Outer Pipe Inner Radius, m:",
            round(bhe.pipe.r_in[0], roundingAmount),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )
        oS += createDRow(
            allocatedWidth,
            "Inner Pipe Inner Radius, m:",
            round(bhe.pipe.r_in[1], roundingAmount),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )

    oS += createDRow(
        allocatedWidth,
        "Pipe Roughness, m:",
        round(bhe.pipe.eps, roundingAmount),
        stringFormat,
        sciFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Shank Spacing, m:",
        round(bhe.pipe.s, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Grout Thermal Conductivity, W/(m*K):",
        round(bhe.grout.k, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Grout Volumetric Heat Capacity, kJ/(K*m^3):",
        round(bhe.pipe.s / 1000, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    if isinstance(bhe.pipe.r_out, float):
        oS += createDRow(
            allocatedWidth,
            "Reynold's Number:",
            round(
                BHE.compute_Reynolds(
                    bhe.m_flow_borehole, bhe.pipe.r_in, bhe.pipe.eps, bhe.fluid
                ),
                roundingAmount,
            ),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )
    else:

        oS += createDRow(
            allocatedWidth,
            "Reynold's Number:",
            round(
                BHE.compute_Reynolds_concentric(
                    bhe.m_flow_pipe, bhe.r_in_out, bhe.r_out_in, bhe.fluid
                ),
                roundingAmount,
            ),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )

    oS += createDRow(
        allocatedWidth,
        "Effective Borehole Resistance, W/(m*K):",
        round(bhe.compute_effective_borehole_resistance(), roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    # Shank Spacing, Pipe Type, etc.

    oS += "Soil Properties: " + "\n"
    oS += createDRow(
        allocatedWidth,
        "Thermal Conductivity, W/(m*K):",
        round(bhe.soil.k, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Volumetric Heat Capacity, kJ/(K*m^3):",
        round(bhe.soil.rhoCp / 1000, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Undisturbed Ground Temperature, C:",
        round(bhe.soil.ugt, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )

    oS += "Fluid Properties" + "\n"
    oS += createDRow(
        allocatedWidth,
        "Volumetric Heat Capacity, kJ/(K*m^3):",
        round(bhe.fluid.rhoCp / 1000, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Thermal Conductivity, W/(m*K):",
        round(bhe.fluid.k, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Fluid Mix:",
        bhe.fluid.fluid.fluid_name,
        stringFormat,
        stringFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Density, kg/m^3:",
        round(bhe.fluid.rho, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    oS += createDRow(
        allocatedWidth,
        "Mass Flow Rate Per Borehole, kg/s:",
        round(bhe.m_flow_borehole, roundingAmount),
        stringFormat,
        floatFormat,
        bTabs=indentedAmount,
    )
    if hasattr(bhe, "h_f"):
        oS += createDRow(
            allocatedWidth,
            "Fluid Convection Coefficient, W/(m*K):",
            round(bhe.h_f, roundingAmount),
            stringFormat,
            floatFormat,
            bTabs=indentedAmount,
        )
    oS += emptyLine

    monthlyLoadValues = []
    mCL = ghe.hybrid_load.monthly_cl
    mHL = ghe.hybrid_load.monthly_hl
    pCL = ghe.hybrid_load.monthly_peak_cl
    pHL = ghe.hybrid_load.monthly_peak_hl
    dCL = ghe.hybrid_load.monthly_peak_cl_duration
    dHL = ghe.hybrid_load.monthly_peak_hl_duration
    n_months = len(ghe.hybrid_load.monthly_cl) - 1
    n_years = int(n_months / 12)
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ] * n_years

    startInd = 1
    stopInd = n_months
    for i in range(startInd, stopInd + 1):
        monthlyLoadValues.append(
            [months[i - 1], mHL[i], mCL[i], pHL[i], dHL[i], pCL[i], dCL[i]]
        )
    monthHeader = [
        [
            "Month",
            "Total Heating",
            "Total Cooling",
            "Peak Heating",
            "PH Duration",
            "Peak Cooling",
            "PC Duration",
        ],
        ["", "KW-Hr", "KW-Hr", "KW", "hr", "KW", "hr"],
    ]

    monthTableFormats = ["s", ".3f", ".3f", ".3f", ".3f", ".3f", ".3f"]

    oS += createTable(
        "GLHE Monthly Loads",
        monthHeader,
        monthlyLoadValues,
        allocatedWidth,
        monthTableFormats,
        fillerSymbol="-",
        centering="^",
    )

    oS += emptyLine

    oS += createTitle(allocatedWidth, "Simulation Parameters")
    oS += createDRow(
        allocatedWidth,
        "Start Month: ",
        ghe.sim_params.start_month,
        stringFormat,
        intFormat,
    )
    oS += createDRow(
        allocatedWidth, "End Month: ", ghe.sim_params.end_month, stringFormat, intFormat
    )
    oS += createDRow(
        allocatedWidth,
        "Maximum Allowable HPEFT, C: ",
        ghe.sim_params.max_EFT_allowable,
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Minimum Allowable HPEFT, C: ",
        ghe.sim_params.min_EFT_allowable,
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Maximum Allowable Height, m: ",
        ghe.sim_params.max_Height,
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Minimum Allowable Height, m: ",
        ghe.sim_params.min_Height,
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Simulation Time, years: ",
        int(ghe.sim_params.end_month / 12),
        stringFormat,
        intFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Simulation Loading Type: ",
        loadMethod,
        stringFormat,
        stringFormat,
    )

    oS += emptyLine

    # Loading Stuff
    oS += createTitle(allocatedWidth, "Simulation Results")
    oS += emptyLine

    # Simulation Results
    EFTTableTitle = "Monthly Temperature Summary"
    # daysInYear = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    # hoursInYear = 24 * daysInYear
    timeVals = ghe.times
    EFTVals = []
    EFTVals.extend(ghe.HPEFT)
    dTbVals = []
    dTbVals.extend(ghe.dTb)
    nYears = 0
    # hTotalYear = np.sum(hoursInYear)
    outArray = []
    lastMonth = -1
    monthTbVals = []
    monthEFTVals = []
    for i in range(len(timeVals)):
        # currentHourMonth = timeVals[i] - hTotalYear * nYears
        currentMonth = int(math.floor(hoursToMonth(timeVals[i])))
        # print(monthEFTVals)
        if currentMonth == lastMonth:
            monthTbVals.append(dTbVals[i])
            monthEFTVals.append(EFTVals[i])
        elif currentMonth != lastMonth:
            if len(monthTbVals) > 0:
                previousTemp = None
                if len(outArray) == 0:
                    previousTemp = bhe.soil.ugt
                else:
                    # print(i)
                    # print(len(outArray))
                    # previousTemp = outArray[-1][1]
                    previousTemp = bhe.soil.ugt
                outArray.append(
                    [
                        currentMonth,
                        previousTemp + monthTbVals[-1],
                        np.max(monthEFTVals),
                        np.min(monthEFTVals),
                    ]
                )
            lastMonth = currentMonth
            monthTbVals = [dTbVals[i]]
            monthEFTVals = [EFTVals[i]]
        if currentMonth % 11 == 0:
            nYears += 1

    headerArray = [
        ["Time", "Tbw", "Max HPEFT", "Min HPEFT"],
        ["(months)", "(C)", "(C)", "(C)"],
    ]
    EFTTableFormats = [".0f", ".3f", ".3f", ".3f"]

    oS += createTitle(allocatedWidth, "Peak Temperature", fillerSymbol="-")
    maxEFT = np.max(EFTVals)
    minEFT = np.min(EFTVals)
    maxEFTTime = timeVals[EFTVals.index(max(EFTVals))]
    minEFTTime = timeVals[EFTVals.index(min(EFTVals))]
    maxEFTTime = hoursToMonth(maxEFTTime)
    minEFTTime = hoursToMonth(minEFTTime)
    oS += createDRow(
        allocatedWidth,
        "Max HPEFT, C:",
        round(maxEFT, roundingAmount),
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Max HPEFT Time, Months:",
        round(maxEFTTime, roundingAmount),
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Min HPEFT, C:",
        round(minEFT, roundingAmount),
        stringFormat,
        floatFormat,
    )
    oS += createDRow(
        allocatedWidth,
        "Min HPEFT Time, Months:",
        round(minEFTTime, roundingAmount),
        stringFormat,
        floatFormat,
    )

    oS += createTable(
        EFTTableTitle,
        headerArray,
        outArray,
        allocatedWidth,
        EFTTableFormats,
        fillerSymbol="-",
        centering="^",
    )

    with open(os.path.join(outputDirectory, summaryFile), "w", newline="") as txtF:
        txtF.write(oS)

    csv1Array = []

    loadingValues = ghe.loading
    # loadingValues_dt = np.hstack((loadingValues[1:] - loadingValues[:-1]))
    currentTime = None
    currentMonth = None
    normalizedLoading = None
    wallTemperature = None
    HPEFTVal = None
    for i in range(len(timeVals)):
        if i + 1 < len(timeVals):
            currentTime = timeVals[i]
            loading = loadingValues[i + 1]
            currentMonth = hoursToMonth(timeVals[i])
            normalizedLoading = loading / (ghe.averageHeight() * ghe.nbh)
            wallTemperature = bhe.soil.ugt + dTbVals[i]
            HPEFTVal = EFTVals[i]
            if True:
                csv1Row = []
                csv1Row.append(timeVals[i])
                csv1Row.append(hoursToMonth(timeVals[i]))
                if i > 1:
                    csv1Row.append(loadingValues[i])
                    csv1Row.append(loadingValues[i] / (ghe.averageHeight() * ghe.nbh))
                else:
                    csv1Row.append(0)
                    csv1Row.append(0)
                csv1Row.append(bhe.soil.ugt + dTbVals[i - 1])
                csv1Row.append(EFTVals[i - 1])
                csv1Array.append(csv1Row)

        else:

            if True:
                csv1Row = []
                csv1Row.append(timeVals[i])
                csv1Row.append(hoursToMonth(timeVals[i]))
                if i > 1:
                    csv1Row.append(loadingValues[i])
                    csv1Row.append(loadingValues[i] / (ghe.averageHeight() * ghe.nbh))
                else:
                    csv1Row.append(0)
                    csv1Row.append(0)
                csv1Row.append(bhe.soil.ugt + dTbVals[i - 1])
                csv1Row.append(EFTVals[i - 1])
                csv1Array.append(csv1Row)

            currentTime = timeVals[i]
            loading = 0
            currentMonth = hoursToMonth(timeVals[i])
            normalizedLoading = loading / (ghe.averageHeight() * ghe.nbh)
            wallTemperature = bhe.soil.ugt + dTbVals[i]
            HPEFTVal = EFTVals[i]
        csv1Row = []
        csv1Row.append(currentTime)
        csv1Row.append(currentMonth)
        csv1Row.append(loading)
        csv1Row.append(normalizedLoading)
        csv1Row.append(wallTemperature)
        csv1Row.append(HPEFTVal)
        csv1Array.append(csv1Row)
    with open(os.path.join(outputDirectory, csvF1), "w", newline="") as csv1OF:
        cW = csv.writer(csv1OF)
        cW.writerow(
            [
                "Time (hr)",
                "Time (month)",
                "Q (Rejection) (w) (before time)",
                "Q (Rejection) (W/m) (before time)",
                "Tb (C)",
                "GHE ExFT (C)",
            ]
        )
        cW.writerows(csv1Array)

    csv2Array = []

    csv2Array.append(["x", "y"])
    for bL in gfunction.bore_locations:
        csv2Array.append([bL[0], bL[1]])

    with open(os.path.join(outputDirectory, csvF2), "w", newline="") as csv2OF:
        cW = csv.writer(csv2OF)
        cW.writerows(csv2Array)

    hourlyLoadings = ghe.hourly_extraction_ground_loads
    csv3Array = []
    csv3Array.append(
        ["Month", "Day", "Hour", "Time (Hours)", "Loading (W) (Extraction)"]
    )
    for i in range(len(hourlyLoadings)):
        hour = i
        month, dayInMonth, hourInDay = GHETimeConvert(hour)
        csv3Array.append([month, dayInMonth, hourInDay, hour, hourlyLoadings[i]])

    with open(os.path.join(outputDirectory, csvF3), "w", newline="") as csv3OF:
        cW = csv.writer(csv3OF)
        cW.writerows(csv3Array)

        # GFunction STS+LTS Table
    # gfunctionTableFormats = [".3f"]
    # gfTableFF = [".3f"] * (1)
    # gfunctionTableFormats.extend(gfTableFF)
    # gfunctionColTitles = ["ln(t/ts)"]

    # gfunctionColTitles.append("H:" + str(round(bH.H, 2)) + "m")

    gfunctionData = []
    csv4Array = [["ln(t/ts)", "H:{:.2f}".format(bhe.b.H)]]
    gheGFAdjusted = ghe.grab_g_function(ghe.B_spacing / float(ghe.averageHeight()))
    gfunctionLogVals = gheGFAdjusted.x
    gfunctionGVals = gheGFAdjusted.y
    for i in range(len(gfunctionLogVals)):
        gfRow = []
        gfRow.append(gfunctionLogVals[i])
        # for gfunctionName in list(gfunction.g_lts):
        # print(gfunction.g_lts[gfunctionName][i])
        # print(gfunctionName)
        # gfRow.append(gfunction.g_lts[gfunctionName][i])
        gfRow.append(gfunctionGVals[i])
        csv4Array.append(gfRow)

    # oS += createTable("GFunction Combined Values", [gfunctionColTitles], gfunctionData, allocatedWidth,
    #                  gfunctionTableFormats,
    #                  fillerSymbol="-", centering="^")
    # oS += emptyLine

    with open(os.path.join(outputDirectory, csvF4), "w", newline="") as csv4OF:
        cW = csv.writer(csv4OF)
        cW.writerows(csv4Array)

    return oS
