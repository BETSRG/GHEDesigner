import csv
import os
from math import pi

from ghedt.RowWise.RowWiseGeneration import genShape, fieldOptimizationWPSpac


def main():
    NogoZoneDir = "NogoZones"
    propFile = "PropBound.csv"
    prop = []
    ng = []
    with open(propFile, "r", newline="") as iF:
        cW = csv.reader(iF)
        for line in cW:
            L = []
            for row in line:
                L.append(float(row))
            prop.append(L)
    for file in os.listdir(NogoZoneDir):
        with open(os.path.join(NogoZoneDir, file), "r", newline="") as iF:
            cW = csv.reader(iF)
            ng.append([])
            for line in cW:
                L = []
                for row in line:
                    L.append(float(row))
                ng[-1].append(L)

    buildVert, nogoVert = genShape(prop, ngZones=ng)
    Directory = (
        r"D:\Work\GSHPResearch\RowWiseCoordGen\Row-wiseCoordinateGenerator\RowWise\\"
    )
    pSpacs = [0.7]
    spacStart = 13.5
    spacStop = 13.7
    spacStep = 0.1
    rotateStep = 1
    fieldOptimizationWPSpac(
        pSpacs,
        spacStart,
        spacStop,
        spacStep,
        rotateStep,
        Directory,
        buildVert,
        ngZones=nogoVert,
        rotateStart=-pi / 2 + 1 / 10000.0,
        rotateStop=pi / 2 - 1 / 10000.0,
        pdfOutputName="Graphs.pdf",
    )


if __name__ == "__main__":
    main()
