import matplotlib.pyplot as plt
from RowWiseGeneration import fieldOptimizationWPSpac
from RowWiseGeneration import plotField
from RowWiseGeneration import fieldGenerator
from RowWiseGeneration import genShape
import numpy
import csv

def main():
    '''
    buildVert = [
        [0, 0],
        [70.104, 0],
        [70.104, 80.772],
        [0, 80.722]

    ]
    nogoVert = [[
        [7.62, 38.1],
        [26, 38.1],
        [26, 58],
        [4, 41]

    ],
    [
        [32,5],
        [57,30],
        [30,20]
    ]]


    xSpac, ySpac, propBound,ngZones = 5, 10,buildVert,nogoVert

    buildVert,nogoVert = genShape(buildVert,ngZones=nogoVert)

    field1 = numpy.array(fieldGenerator(xSpac, ySpac, propBound))
    plotField(field1,shape=buildVert)
    '''
    pB = [
      [125.8,197.1],
        [129.1,206.5],
        [232.5,176.0],
        [232.1,171.5],
        [230.0,150],
        [210,150],
        [215, 160],
        [182,180 ]
    ]
    pC = [
        [143,7.5 ],
        [192, -5],
        [210,65 ],
        [221, 68],
        [216.9476649,37.55276038 ],
        [209.4772373,-11.17091042 ],
        [134.5,-11.9829716 ],
        [143,7.5 ]
    ]
    nogoVert = []
    xSpac, ySpac, propBound, ngZones = 7, 7, pC, nogoVert

    buildVert, nogoVert = genShape(propBound, ngZones=nogoVert)
    field1 = numpy.array(fieldGenerator(xSpac, ySpac, propBound))
    plotField(field1, shape=buildVert)

    outputFileName = "C03Coords.csv"
    with open(outputFileName,"w",newline="") as outputFile:
        cW = csv.writer(outputFile)
        cW.writerow(["x","y"])
        cW.writerows(field1)
if __name__ == "__main__":
    main()