import json

import numpy as np

from ghedt.RowWise.RowWiseGeneration import genBoreHoleConfig, genShape, plotField


def genFields(
    NBHUpperLim,
    nogos,
    MaxNumberOfFields=None,
    offsetIterator=5,
    XSpac=5,
    YSpac=5,
    SampleRate=0,
    outPutDir="",
    rotate=0,
):
    fields = []
    NBHs = []
    lastNBH = 0
    numberOfFields = 0
    currentOffset = offsetIterator
    nogoVert = nogos[0]
    sel = SampleRate
    while lastNBH < NBHUpperLim:
        fieldVert = [
            [0, 0],
            [nogoVert[1][0] + 2 * currentOffset, 0],
            [nogoVert[1][0] + 2 * currentOffset, nogoVert[2][1] + 2 * currentOffset],
            [0, nogoVert[2][1] + 2 * currentOffset],
        ]
        nogoVertC = [
            [
                [nogoVert[0][0] + currentOffset, nogoVert[0][1] + currentOffset],
                [nogoVert[1][0] + currentOffset, nogoVert[1][1] + currentOffset],
                [nogoVert[2][0] + currentOffset, nogoVert[2][1] + currentOffset],
                [nogoVert[3][0] + currentOffset, nogoVert[3][1] + currentOffset],
            ]
        ]
        buildVert, nogoV = genShape(fieldVert, ngZones=nogoVertC)
        fieldCandidate = genBoreHoleConfig(
            buildVert, YSpac, XSpac, nogo=nogoV, rotate=rotate
        )
        fieldCandidate = np.array(
            [fieldCandidate[element] for element in fieldCandidate]
        )
        # fieldCandidate = fieldGenerator(XSpac, YSpac, fieldVert, ngZones=nogos)

        NBH = len(fieldCandidate)

        if NBH > lastNBH:
            NBHs.append(NBH)
            lastNBH = NBH
            numberOfFields += 1
            if SampleRate != 0 and numberOfFields == sel:
                plotField(fieldCandidate, shape=buildVert, shapes=nogoV)
            l1 = list(fieldCandidate[:, 0])
            l2 = list(fieldCandidate[:, 1])
            fieldCandidate = {"x": l1, "y": l2}
            fields.append(fieldCandidate)
            print("NBH: " + str(NBH))
        else:
            pass

        if MaxNumberOfFields is not None and numberOfFields > MaxNumberOfFields:
            break
        currentOffset += offsetIterator
    i = 0
    for field in fields:
        with open(outPutDir + str(NBHs[i]) + ".json", "w", newline="") as outPutFile:
            json.dump(field, outPutFile)

        i += 1


def main():
    nogoVert = [[[0, 0], [60, 0], [60, 45], [0, 45]]]
    genFields(
        1000, nogoVert, outPutDir="FieldSizeOutput\\", SampleRate=10, offsetIterator=2
    )


if __name__ == "__main__":
    main()
