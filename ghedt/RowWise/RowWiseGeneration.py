import numpy as np
from math import pi
import ghedt.RowWise.Shape as Shape
import matplotlib.pyplot as plt
from math import atan
from math import sqrt
from math import cos
from math import sin
from ghedt.RowWise.Shape import Shapes
import ghedt.RowWise.Shape
from math import copysign
from matplotlib.backends.backend_pdf import PdfPages
import csv
import os
'''
Used to create an instance of a borefield

Functions:

    genBoreHoleConfig(float,float,float,float,float,float,object,int,float) -> [[float]]
    plotField([[float]],object,string,boolean,object) -> none
    processRows(float,float,object,float,float,float,float,int)
    def Between(float,float,float) -> double
    distribute(float,float,float,float,[[float]]) -> appends to given array

'''

def genShape (propBound,ngZones=None):
    '''Returns an array of shapes objects representing the coordinates given '''
    rA = []
    rA.append(Shape.Shapes(propBound))
    if ngZones!=None:
        rN = []
        for ngZone in ngZones:
            rN.append(Shape.Shapes(ngZone))
        rA.append(rN)
    else:
        rA.append(None)
    return rA
def fieldOptimizationWPSpac_FR(pSpacs,spacStart,rotateStep,propBound,ngZones=None,rotateStart=None,rotateStop=None,pdfOutputName = "Graphs.pdf"):
    ''' Optimizes a Field by iterating over input values w/o perimeter spacing

        Parameters:
            pSpacs(float): Ratio of perimeter spacing to other target spacing
            spacStart(float): the initial target spacing that the optimization program will start with
            spacStop(float): the final target spacing that the optimization program will end with (inclusive)
            spacStep(float): the value that each step will take in optimization program
            rotateStep(float): the amount of rotation that will be changed per step (in degrees)
            Directory(String): Directory where output files should be sent
            propBound([[float,float]]): 2d array of floats that represent the property boundary (counter clockwise)
            ngZones([[[float,float]]]): 3d array representing the different zones on the property where no boreholes can be placed
            rotateStart(float): the rotation that the field will start at (-pi/2 < rotateStart < pi/2)
            rotateStop(float): the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop < pi/2)
            pdfOutputName(string): the name of the output pdf containing the graphs for the max field at each target spacing

        Outputs:
            CSV's containing the coordinates for the max field for each target spacing, their respective graphs, and their respective data

        '''
    if rotateStart == None:
        rotateStart = (-90.0+rotateStep)*(pi/180.0)
    if rotateStop == None:
        rotateStop = pi/2
    if rotateStart >= pi / 2 or rotateStart <= -pi / 2 or rotateStop > pi / 2 or rotateStop < -pi / 2:
        print("Invalid Rotation")
        return
    field = None
    fieldName = None
    for pSpac in pSpacs:


        #Target Spacing iterates

        spac = spacStart
        rt = rotateStart


        yS = spac
        xS = yS

        maxL = 0
        maxHole = None
        maxrt = None

        while rt < rotateStop:
            #print("Current Rotation: ",rt)
            hole = twoSpacGenBHC(propBound, yS, xS, rotate=rt, nogo=ngZones,PSpace=pSpac*xS)

            #Assuming that the rotation with the maximum number of boreholes is most efficiently using space
            if len(hole) > maxL:
                maxL = len(hole)
                maxrt = rt*(180/pi)
                maxHole = hole

            rt+=rotateStep*(pi/180)

            #Ensures that there are no repeated boreholes
            maxHole = np.array(remove_duplicates(maxHole, pSpac * xS))
            #print("MaxHOle: ",maxHole)

            #Reports removal of repeated boreholes
            if maxL > len(maxHole):
                print(maxL-len(maxHole)," holes removed")
                maxL = len(maxHole)
            field = (maxHole)
            fieldName = ("P"+str(pSpac)+"_S"+str(spac)+"_rt"+str(maxrt))
        return [field,fieldName]
def fieldOptimization_FR(spacStart,rotateStep,propBound,ngZones=None,rotateStart=None,rotateStop=None,pdfOutputName = "Graphs.pdf"):
    ''' Optimizes a Field by iterating over input values w/o perimeter spacing

        Parameters:
            spacStart(float): the initial target spacing that the optimization program will start with
            spacStop(float): the final target spacing that the optimization program will end with (inclusive)
            spacStep(float): the value that each step will take in optimization program
            rotateStep(float): the amount of rotation that will be changed per step (in degrees)
            Directory(String): Directory where output files should be sent
            propBound([[float,float]]): 2d array of floats that represent the property boundary (counter clockwise)
            ngZones([[[float,float]]]): 3d array representing the different zones on the property where no boreholes can be placed
            rotateStart(float): the rotation that the field will start at (-pi/2 < rotateStart < pi/2)
            rotateStop(float): the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop < pi/2)
            pdfOutputName(string): the name of the output pdf containing the graphs for the max field at each target spacing

        Outputs:
            CSV's containing the coordinates for the max field for each target spacing, their respective graphs, and their respective data

        '''
    if rotateStart == None:
        rotateStart = (-90.0+rotateStep)*(pi/180.0)
    if rotateStop == None:
        rotateStop = pi/2
    if rotateStart >= pi / 2 or rotateStart <= -pi / 2 or rotateStop > pi / 2 or rotateStop < -pi / 2:
        print("Invalid Rotation")
        return
    field = None
    fieldName = None


    #Target Spacing iterates

    spac = spacStart
    rt = rotateStart


    yS = spac
    xS = yS

    maxL = 0
    maxHole = None
    maxrt = None

    while rt < rotateStop:
        #print("Current Rotation: ",rt)
        hole = genBoreHoleConfig(propBound, yS, xS, rotate=rt, nogo=ngZones)

        #Assuming that the rotation with the maximum number of boreholes is most efficiently using space
        if len(hole) > maxL:
            maxL = len(hole)
            maxrt = rt*(180/pi)
            maxHole = hole

        rt+=rotateStep*(pi/180)

        #Ensures that there are no repeated boreholes
        maxHole = np.array(remove_duplicates(maxHole, xS*1.2))
        #print("MaxHOle: ",maxHole)

        #Reports removal of repeated boreholes
        if maxL > len(maxHole):
            print(maxL-len(maxHole)," holes removed")
            maxL = len(maxHole)
        field = (maxHole)
        fieldName = ("S"+str(spac)+"_rt"+str(maxrt))
        return [field,fieldName]
def fieldOptimizationWPSpac(pSpacs,spacStart,spacStop,spacStep,rotateStep,Directory,propBound,ngZones=None,rotateStart=None,rotateStop=None,pdfOutputName = "Graphs.pdf"):
    ''' Optimizes a Field by iterating over input values w/o perimeter spacing

        Parameters:
            pSpacs(float): Ratio of perimeter spacing to other target spacing
            spacStart(float): the initial target spacing that the optimization program will start with
            spacStop(float): the final target spacing that the optimization program will end with (inclusive)
            spacStep(float): the value that each step will take in optimization program
            rotateStep(float): the amount of rotation that will be changed per step (in degrees)
            Directory(String): Directory where output files should be sent
            propBound([[float,float]]): 2d array of floats that represent the property boundary (counter clockwise)
            ngZones([[[float,float]]]): 3d array representing the different zones on the property where no boreholes can be placed
            rotateStart(float): the rotation that the field will start at (-pi/2 < rotateStart < pi/2)
            rotateStop(float): the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop < pi/2)
            pdfOutputName(string): the name of the output pdf containing the graphs for the max field at each target spacing

        Outputs:
            CSV's containing the coordinates for the max field for each target spacing, their respective graphs, and their respective data

        '''
    if rotateStart == None:
        rotateStart = (-90.0+rotateStep)*(pi/180.0)
    if rotateStop == None:
        rotateStop = pi/2
    if rotateStart >= pi / 2 or rotateStart <= -pi / 2 or rotateStop > pi / 2 or rotateStop < -pi / 2:
        print("Invalid Rotation")
        return

    for pSpac in pSpacs:

        #Will place all output information in a sub-directory \P0.X
        pSpacDir = Directory + "\\P" + str(pSpac)
        if not os.path.isdir(pSpacDir):
            os.mkdir(pSpacDir)
        with PdfPages(pSpacDir + "\\"+pdfOutputName) as PdP:

            #Target Spacing iterates
            for spac in [spacStart+i*spacStep for i in range(int((spacStop-spacStart)/spacStep))]:

                rt = rotateStart


                yS = spac
                xS = yS

                maxL = 0
                maxHole = None
                maxrt = None

                while rt < rotateStop:
                    #print("Rotation: ",rt)
                    #print("Current Rotation: ",rt)
                    hole = twoSpacGenBHC(propBound, yS, xS, rotate=rt, nogo=ngZones,PSpace=pSpac*xS)

                    #Assuming that the rotation with the maximum number of boreholes is most efficiently using space
                    if len(hole) > maxL:
                        maxL = len(hole)
                        maxrt = rt*(180/pi)
                        maxHole = hole

                    rt+=rotateStep*(pi/180)

                #Ensures that there are no repeated boreholes
                maxHole = np.array(remove_duplicates(maxHole, pSpac * xS))

                #Reports removal of repeated boreholes
                if maxL > len(maxHole):
                    #print(maxL-len(maxHole)," holes removed")
                    maxL = len(maxHole)

                #Reports Current Progress (Helpful for longer sequences to give an idea of progress being made)
                print("Rotate Max: %f, MaxLength: %f"%(maxrt,maxL))

                #Exports a graph of the selected field
                plotField(maxHole,shape=propBound,shapes=ngZones,SaveOrShow=False,Pdf=PdP,title="".join(["Spacing: ","{:.1f}".format(spac),"m NBH=",str(maxL)]))

                #Creates Further Subdirectories for the output csv's
                MDir = pSpacDir+"\\MaxData"
                PDir = pSpacDir+"\\FieldCoords"
                if not os.path.isdir(MDir):
                    os.mkdir(MDir)
                if not os.path.isdir(PDir):
                    os.mkdir(PDir)

                #Export CSV's
                with open("".join([MDir, "\MaxData_Spac_",str(spac), ".csv"]), "w",newline="") as outputFile:
                    csvWriter = csv.writer(outputFile)
                    csvWriter.writerow(["Rotation: ",maxrt,"Length: ",maxL,"Spacing: ",spac])
                with open("".join([PDir, "\MaxL_Spac_",str(spac), ".csv"]), "w",newline="") as outputFile:
                    csvWriter = csv.writer(outputFile)
                    csvWriter.writerow(["x" , "y" ])
                    csvWriter.writerows(maxHole)
def fieldGenerator(xSpac,ySpac,propBound,ngZones=None,rotateStart=None,rotateStop=None,rotateStep=.1,pSpac = None):
    ''' Optimizes a Field by iterating over input values w/o perimeter spacing

        Parameters:
            pSpac(float): Perimeter Spacing
            xSpac(float): the intra-row target spacing
            ySpac(float): the inter-row target spacing
            rotateStep(float): the amount of rotation that will be changed per step (in degrees)
            propBound([[float,float]]): 2d array of floats that represent the property boundary (counter clockwise)
            ngZones([[[float,float]]]): 3d array representing the different zones on the property where no boreholes can be placed
            rotateStart(float): the rotation that the field will start at (-pi/2 < rotateStart < pi/2)
            rotateStop(float): the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop < pi/2)

        Outputs:
            CSV's containing the coordinates for the max field for each target spacing, their respective graphs, and their respective data

        '''
    propBound,ngZones = genShape(propBound,ngZones=ngZones)
    if rotateStart == None:
        rotateStart = (-90.0+rotateStep)*(pi/180.0)
    if rotateStop == None:
        rotateStop = pi/2
    if rotateStart >= pi / 2 or rotateStart <= -pi / 2 or rotateStop > pi / 2 or rotateStop < -pi / 2:
        print("Invalid Rotation")
        return

    #Target Spacing iterates

    rt = rotateStart

    maxL = 0
    maxHole = None

    while rt < rotateStop:
        #print("Current Rotation: ",rt)
        hole = None
        if pSpac == None:
            hole = genBoreHoleConfig(propBound, ySpac, xSpac, rotate=rt, nogo=ngZones)
        else:
            hole = twoSpacGenBHC(propBound, ySpac, xSpac, rotate=rt, nogo=ngZones,PSpace=pSpac)

        #Assuming that the rotation with the maximum number of boreholes is most efficiently using space
        if len(hole) > maxL:
            maxL = len(hole)
            maxHole = hole

        rt+=rotateStep*(pi/180)

    #Ensures that there are no repeated boreholes
    if pSpac == None:
        maxHole = np.array(remove_duplicates(maxHole, min(xSpac,ySpac)))
    else:
        maxHole = np.array(remove_duplicates(maxHole,min( pSpac,xSpac,ySpac)))

    return maxHole


def fieldOptimization(spacStart,spacStop,spacStep,rotateStep,Directory,propBound,ngZones=None,rotateStart=None,rotateStop=None,pdfOutputName = "Graphs.pdf"):
    ''' Optimizes a Field by iterating over input values w/o perimeter spacing

    Parameters:
        spacStart(float): the initial target spacing that the optimization program will start with
        spacStop(float): the final target spacing that the optimization program will end with (inclusive)
        spacStep(float): the value that each step will take in optimization program
        rotateStep(float): the amount of rotation that will be changed per step (in degrees)
        Directory(String): Directory where output files should be sent
        propBound([[float,float]]): 2d array of floats that represent the property boundary (counter clockwise)
        ngZones([[[float,float]]]): 3d array representing the different zones on the property where no boreholes can be placed
        rotateStart(float): the rotation that the field will start at (-pi/2 < rotateStart < pi/2)
        rotateStop(float): the rotation that the field will stop at (exclusive) (-pi/2 < rotateStop < pi/2)
        pdfOutputName(string): the name of the output pdf containing the graphs for the max field at each target spacing

    Outputs:
        CSV's containing the coordinates for the max field for each target spacing, their respective graphs, and their respective data

    '''
    if rotateStart == None:
        rotateStart = (-90.0+rotateStep)*(pi/180.0)
    if rotateStop == None:
        rotateStop = pi/2
    if rotateStart >= pi/2 or rotateStart <= -pi/2 or rotateStop > pi/2 or rotateStop < -pi/2:
        print("Invalid Rotation")
        return

    with PdfPages(Directory + "\\" + pdfOutputName) as PdP:

        # Target Spacing iterates by .1
        for spac in [spacStart+i*spacStep for i in range(int((spacStop-spacStart)/spacStep))]:

            rt = rotateStart

            # Converts from tenths of a unit to a full units
            yS = spac
            xS = yS

            maxL = 0
            maxHole = None
            maxrt = None

            while rt < rotateStop:

                # print("Current Rotation: ",rt)
                hole = genBoreHoleConfig(propBound, yS, xS, rotate=rt, nogo=ngZones)

                # Assuming that the rotation with the maximum number of boreholes is most efficiently using space
                if len(hole) > maxL:
                    maxL = len(hole)
                    maxrt = rt * (180 / pi)
                    maxHole = hole

                rt += rotateStep * (pi / 180)

            # Ensures that there are no repeated boreholes
            maxHole = np.array(remove_duplicates(maxHole, xS))

            # Reports removal of repeated boreholes
            if maxL > len(maxHole):
                print(maxL - len(maxHole), " holes removed")
                maxL = len(maxHole)

            # Reports Current Progress (Helpful for longer sequences to give an idea of progress being made)
            print("Rotate Max: %f, MaxLength: %f, NBH: %f" % (maxrt, maxL,len(maxHole)))

            # Exports a graph of the selected field
            plotField(maxHole, shape=propBound, shapes=ngZones, SaveOrShow=False, Pdf=PdP,
                      title="".join(["Spacing: ", "{:.1f}".format(spac), "m NBH=", str(maxL)]))

            # Creates Subdirectories for the output csv's
            MDir = Directory + "\\MaxData"
            PDir = Directory + "\\FieldCoords"
            if not os.path.isdir(MDir):
                os.mkdir(MDir)
            if not os.path.isdir(PDir):
                os.mkdir(PDir)

            # Export CSV's
            with open("".join([MDir, "\MaxData_Spac_", str(spac), ".csv"]), "w", newline="") as outputFile:
                csvWriter = csv.writer(outputFile)
                csvWriter.writerow(["Rotation: ", maxrt, "Length: ", maxL, "Spacing: ", spac])
            with open("".join([PDir, "\MaxL_Spac_", str(spac), ".csv"]), "w", newline="") as outputFile:
                csvWriter = csv.writer(outputFile)
                csvWriter.writerow(["x", "y"])
                csvWriter.writerows(maxHole)

#This is adapted code from Jack Cook's Borefield processing code
def find_duplicates(boreField,spac, disp=False):
    """
    The distance method :func:`Borehole.distance` is utilized to find all
    duplicate boreholes in a boreField.
    This function considers a duplicate to be any pair of points that fall
    within each others radius. The lower index (i) is always stored in the
    0 position of the tuple, while the higher index (j) is stored in the 1
    position.
    Parameters
    ----------
    boreField : list
        A list of :class:`Borehole` objects
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    Returns
    -------
    duplicate_pairs : list
        A list of tuples where the tuples are pairs of duplicates
    """

    duplicate_pairs = []   # define an empty list to be appended to
    for i in range(len(boreField)):
        borehole_1 = boreField[i]
        for j in range(i, len(boreField)):  # only loop unique interactions
            borehole_2 = boreField[j]
            if i == j:  # skip the borehole itself
                continue
            else:
                dist = sqDist(borehole_1,borehole_2)
            if abs(dist) < (spac*10**-1):
                print(borehole_1)
                duplicate_pairs.append((i, j))
    if disp:
        # pad with '-' align in center
        output = f"{ '*gt.boreholes.find_duplicates()*' :-^50}"
        # keep a space between the function name
        print(output.replace('*', ' '))
        print('The duplicate pairs of boreholes found: {}'.format(duplicate_pairs))
    return duplicate_pairs

def sqDist(p1,p2):
    '''Returns the cartesian distance between two points'''
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
def remove_duplicates(boreField,spac, disp=False):
    """
    Removes all of the duplicates found from the duplicate pairs returned in
    :func:`check_duplicates`.
    For each pair of duplicates, the first borehole (with the lower index) is
    kept and the other (with the higher index) is removed.
    Parameters
    ----------
    boreField : list
        A list of :class:`Borehole` objects
    disp : bool, optional
        Set to true to print progression messages.
        Default is False.
    Returns
    -------
    new_boreField : list
        A boreField without duplicates
    """
    # get a list of tuple
    duplicate_pairs = find_duplicates(boreField,spac, disp=disp)

    new_boreField = []

    # values not to be included
    duplicate_bores = []
    for i in range(len(duplicate_pairs)):
        duplicate_bores.append(duplicate_pairs[i][1])

    for i in range(len(boreField)):
        if i in duplicate_bores:
            continue
        else:
            new_boreField.append(boreField[i])
    if disp:
        # pad with '-' align in center
        print(f"{'*gt.boreholes.remove_duplicates()*' :-^50}".\
              replace('*', ' '))  # keep a space between the function name
        n_duplicates = len(boreField) - len(new_boreField)
        print('The number of duplicates removed: {}'.format(n_duplicates))

    return new_boreField



def twoSpacGenBHC(field,YSpac,XSpac,nogo = None,maxLength = 250,minVertP=0,rotate = 0,PSpace = None,ISpace = None):
    ''' Generates a borefield that has perimeter spacing

    Parameters:
        field: The outer boundary of the property representated as a array of points
        YSpac: Target Spacing in y-dir
        XSpac: Target Spacing in x-dir
        nogo: a 3d array representing all the areas where bore holes cannot be placed
        maxLength: maximum amount of boreholes allowed
        minVertP: minimum vertices required for a field to be made
        rotate: the amount of rotation (rad)
        PSpace: Perimeter spacing
        ISpace: Spacing required between perimeter and the rest
        ISpace: Min spacing required from all edges

    '''
    if PSpace == None:
        PSpace = .9*XSpac
    if ISpace == None:
        ISpace = XSpac

    #calls the standard row-wise coord generator w/ the adjusted vertices
    holes = genBoreHoleConfig(field,YSpac,XSpac,nogo=nogo,maxLength=maxLength,minVertP=minVertP,rotate=rotate)
    #plotField(np.array([holes[element] for element in holes]),shape=field,shapes=nogo)
    holes = [holes[element] for element in holes]
    removePointsToClose(field,holes,ISpace,nogos=nogo)
    #plotField(np.array(holes),shape=field,shapes=nogo)

    #places the boreholes along the perimeter of the property boundary and nogo zone(s)
    perimeterDistribute(field,PSpace,holes)
    if nogo != None:
        for ng in nogo:
            perimeterDistribute(ng,PSpace,holes)
    #plotField(np.array(holes),shape=field,shapes=nogo)
    #returns the Holes as a numpy array for easier manipulation
    returnArray = np.array(holes)
    return returnArray


def removePointsToClose(field,holes,ISpace,nogos=None):
    '''Will Remove All points too close to the field and nogozones

    Parameters:
        field: The outer boundary of the property representated as a array of points
        nogos: a 3d array representing all the areas where bore holes cannot be placed
        holes: 2d array containing all of the current bore holes
        ISpace: Min spacing required from all edges
    '''
    field = field.c
    lF = len(field)
    for i in range(lF):
        p1 = field[i]
        p2 = None
        if i == lF-1:
            p2 = field[0]
        else:
            p2 = field[i+1]
        removePointsCloseToLine(p1,p2,holes,ISpace)
    if nogos != None:
        lNGS = len(nogos)
        for i in range(lNGS):
            ng = nogos[i].c
            lN = len(ng)
            for j in range(lN):
                p1 = ng[j]
                p2 = None
                if j == lN-1:
                    p2 = ng[0]
                else:
                    p2 = ng[j+1]
                removePointsCloseToLine(p1,p2,holes,ISpace)
    return

def removePointsCloseToLine(p1,p2,holes,ISpace):
    '''Removes points that are close to the given line

    Paremeters:
        p1([float,float]): first point in line
        p2([float,float]): second point in line
        holes: 2d array containing a bunch of points
        ISpace(float): distance cutoff for how close points can be

    '''
    lH = len(holes)
    i=0
    kRS = []
    while i < lH:
        hole = holes[i]
        dp = distanceFromLine(p1,p2,hole,ISpace)
        if dp < ISpace:
            del holes[i]
            lH-=1
        else:
            i+=1
    #for kR in kRS:
        #del holes[kR]
    return
def distanceFromLine(p1,p2,otherpoint,ISpace):
    '''Calculates the distance from a point to a line (closest distance): https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line

    Parameter:
        p1: first point on line
        p2: second point on line
        otherpoint: point which is being measured to

    '''
    dxl = p2[0]-p1[0]
    dyl = p2[1]-p1[1]
    dx = p1[0]-otherpoint[0]
    dy = p1[1]-otherpoint[1]
    num = abs(dxl*dy-dx*dyl)
    denom = sqrt(dxl*dxl+dyl*dyl)
    dp = num/denom
    dL = sqDist(p1, p2)
    d01 = sqDist(p1, otherpoint)
    d02 = sqDist(p2, otherpoint)
    #if dp > d01:
        #print("P1({},{}),P2({},{}), otherpoint({},{}))".format(p1[0],p1[1],p2[0],p2[1],otherpoint[0],otherpoint[1]))
    if d01*d01-dp*dp < 0:
        return d01
    if sqrt(d01 * d01 - dp * dp) / dL > 1:
        return min(d01, d02)
    if otherpoint[0] > min(p1[0],p2[0]) and otherpoint[0] < max(p1[0],p2[0]) or otherpoint[1] > min(p1[1],p2[1]) and otherpoint[1] < max(p1[1],p2[1]):
        return min(d01,d02,dp)
    else:
        return min(d01,d02)
    '''
    if p1[0] > p2[0]:
        if otherpoint[0] > p1[0]:
            return d01
        elif otherpoint[0] > p2[0]:
            return dp
        else:
            return d02
    elif p1[0] < p2[0]:
        if otherpoint[0] < p1[0]:
            return d01
        elif otherpoint[0] < p2[0]:
            return dp
        else:
            return d02
    else:
        if p1[1] > p2[1]:
            if otherpoint[1] > p1[1]:
                return d01
            elif otherpoint[1] > p2[1]:
                return dp
            else:
                return d02
        else:
            if otherpoint[1] < p1[1]:
                return d01
            elif otherpoint[1] < p2[1]:
                return dp
            else:
                return d02
    #'''

''' Legacy Code
def cornerExtension (P1,P2,P3,sp,inOrOut):

    #Representing the corner with two intersecting lines (P2 being the intersection point)
    l1 = [P1[0],P1[1],P2[0],P2[1]]
    l2 = [P2[0],P2[1],P3[0],P3[1]]

    #getting interior angle of the intersection
    theta = angleBetweenLines(l1,l2)
    #print("Theta value: ",theta/2.0)

    #get the average slope of the two lines (represented as dx and dy individually in case dx = 0)
    dx,dy = avgSlope(l1,l2)
    #print("DX: ",dx)
    #print("DY: ",dy)

    #Getting angle of the slope relative to the x-axis
    theta2 = getAngle(dx,dy)

    #Calculating the Distance between the original corner and the adjusted one
    s = sp / (sin(theta / 2))
    #print("S: ",s)

    #Generating the four possible locations that the adjusted vertex could be located (only two of them really being important)
    p41 = [P2[0]+s*cos(theta2),P2[1]+s*sin(theta2)]
    p42 = [P2[0]-s*cos(theta2),P2[1]-s*sin(theta2)]
    p43 = [P2[0]+s*cos(theta2),P2[1]-s*sin(theta2)]
    p44 = [P2[0]-s*cos(theta2),P2[1]+s*sin(theta2)]

    #Calculating a reference point to help determine which possible vertex location makes sense
    pRef = avgPoint(P1,P3)
    #print(theta2)

    #Calculating all of the distances between the possible vertex locations and the reference point
    d41=sqDist(pRef,p41)
    d42=sqDist(pRef,p42)
    d43 = sqDist(pRef, p43)
    d44 = sqDist(pRef, p44)

    #Combining the distances and possible vertex locations in an array in order to easily choose one
    rA = []
    rA.append([d41,p41])
    rA.append([d42, p42])
    rA.append([d43, p43])
    rA.append([d44, p44])

    #Figures out which point is the furthest away and which one is the closest to the reference point
    maxI = -1
    maxV = float('-inf')
    minI = -1
    minV = float('inf')
    for i in range(len(rA)):
        if rA[i][0] > maxV:
            maxV = rA[i][0]
            maxI = i
        if rA[i][0] < minV:
            minV = rA[i][0]
            minI = i
    # Chooses the legitimate vertex location based on which one is furthest or closest to the reference point (based on whether the vertex should be shoved outward or inward)
    if inOrOut:
        return rA[minI][1]
    else:
        return rA[maxI][1]

def avgPoint(p1,p2):
    return [(p1[0]+p2[0])/2,(p1[1]+p2[1])/2]

def avgSlope(l1,l2):
    x11, y11, x12, y12 = l1
    x21, y21, x22, y22 = l2
    len1 = sqDist([x11,y11],[x12,y12])
    len2 = sqDist([x12,y12],[x22,y22])
    dx1 = (x12-x11)/len1
    dx2 = (x22-x21)/len2
    dy1 = (y12-y11)/len1
    dy2 = (y22 - y21)/len2
    return [(dx1+dx2)/2.0,(dy1+dy2)/2.0]

def angleBetweenLines(l1,l2):
    x11,y11,x12,y12 = l1
    x21, y21, x22, y22 = l2
    theta1 = getAngle(x12-x11,y12-y11)
    theta2 = getAngle(x22-x21,y22-y21)
    return abs(theta1-theta2)

def getAngle (dx,dy):
    rotate = None
    if dx == 0:
        if dy < 0:
            rotate = -pi/2.0
        else:
            rotate = pi/2.0
    else:
        rotate = atan(dy / dx)
    return rotate

def sqDist(p1,p2):
    return sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))
'''
def perimeterDistribute(field,spac,r):
    '''Distributes boreholes along the perimeter of a given shape

    Parameters:
        field: array of points representing closed polygon
        spac (float): spacing that the boreholes should have from one another
        r (dict{}) existing dictionary of bore holes which will be appended to

    '''
    #print(r)
    for i in range(len(field.c)):
        #print(i)
        vert1 = None
        vert2 = None
        if i == len(field.c)-1:
            vert1 = field.c[i]
            vert2 = field.c[0]
        else:
            vert1 = field.c[i]
            vert2 = field.c[i+1]
        dx = vert2[0]-vert1[0]
        dy = vert2[1]-vert1[1]

        #Checking how many boreholes can be distributed along the line
        dist = sqDist(vert1,vert2)
        numHoles = int(dist//spac)


        #Distributing the spacing to the x and y directions
        xSpac = None
        ySpac = None
        if numHoles > 0:
            xSpac = dx/numHoles
            ySpac = dy/numHoles
        #print("".join(["Dy: ",str(dy),", Dx: ",str(dx),", xNum: ",str(dx // (spac * cos(theta))),", yNum: ",str(( dy // (spac * sin(theta))))]))
        currentP = [vert1[0],vert1[1]]

        #for loop is tuned to leave one spot empty for the next line
        for i in range(numHoles):
            #if i==2:
                #print("XD: ",vert2[0],", YD: ",vert2[1],", Current X: ",currentP[0],", Current Y: ",currentP[1],", theta: ",theta,", actSpac*cos(theta): ",cos(theta) )
            r.append([currentP[0],currentP[1]])
            currentP[0]+=xSpac
            currentP[1]+=ySpac




def genBoreHoleConfig(field,YSpac,XSpac,nogo = None,maxLength = 250,minVertP=0,rotate = 0):
    '''
    Function generates a series of x,y points repersenting a field of bore holes
    in a trapezoidal shape. Returs empty if boreHole field does not meet given requirements

    Parameters
    -------------
        :param BaseSLoc: [float,float]
            x,y coordinates of the leftmost point of the base of the trapezoid
        :param BaseWidth: float
            the width of the base of the trapezoid
        :param TopSLoc: [float,float]
            x,y coordinates of the leftmost point of the top of the trapezoid
        :param TopWidth: float
            the width of the top of the trapezoid
        :param YSpac: float
            the minimum spacing between points in the y-dir
        :param XSpac: float
            the minimum spacing between points in the x-dir
        :param shape: shape object
            the obstruction or "nogo zone" to the borefield
        :param maxLength: int
            the maximum number of boreholes allowed in the field
        :param minVert: float
            the fraction of vertices of the no-go zone required to be in the bore hole field
        :return: [[float]] -> 2 col + n rows
    '''

    if nogo == None:
        nogo = []
    #Decides which vertex to start generating boreholes at by finding the "lowest" vertex relative to a rotated x-axis
    lowestVertVal = float('inf')
    highestVertVal = float('-inf')
    lowestVert = None
    highestVert = None
    for vert in field.c:
        #print("Vertex X value: ",vert[0])
        #print("Vertex Y value: ", vert[1])
        phi = 0
        if vert[0] != 0:
            phi = atan(vert[1]/vert[0])
        else:
            phi = pi/2
        R = sqrt(vert[1]*vert[1]+vert[0]*vert[0])
        refang = phi
        #print(phi)
        #print("The Phi value is: %f"%phi)
        #sign = 1
        if phi > pi/2:
            if phi > pi:
                if phi > 3*pi/2.0:
                    refang = 2*rotate + 3*pi/2 - phi
                else:
                    refang = 2*rotate + pi -phi
            else:
                refang = pi - phi +2*rotate
       # if phi > pi + rotate or phi < rotate:
        #    sign = -1
        yp = R*sin(refang-rotate)
        #print("yp is: ", yp)
        if yp < lowestVertVal:
            lowestVertVal = yp
            lowestVert = vert
        if yp > highestVertVal:
            highestVertVal = yp
            highestVert = vert

    #Determines the number of rows as well as the distance between the rows
    nrows = int((highestVertVal-lowestVertVal)//YSpac)
    d = highestVertVal-lowestVertVal
    s = d/nrows
    rowspace = [-1*s*cos(pi/2  - rotate),s*sin(pi/2 - rotate)]

    #Establishes the dictionary where the boreholes will be added two as well as establishing a point on the first row
    boreHoles = {}
    rowPoint = [lowestVert[0],lowestVert[1]]

    #This is just a value that is combined with the slope of the row's to establish two points defining a row (could be any value)
    pointShift = 100.0

    for ri in range(nrows+1):

        #Row Defined by two points
        row = [rowPoint[0],rowPoint[1],rowPoint[0]+pointShift,rowPoint[1]+(-rowspace[0]/rowspace[1])*(pointShift)]


        #Gets Intersection between current row and property boundary
        finters = field.lineintersect(row,rotate)

        #Stores the number of intersections with the row
        flen = len(finters)

        #Checks for edge case where a single intersection is reported as two and treats it as one
        if (flen>1 and abs(finters[0][0]-finters[1][0])<1e-5 and abs(finters[0][1]-finters[1][1])<1e-5):
            fi = 0
            fij = 0
            while fi < flen:
                while fij < flen:
                    if fij == fi:
                        fij+=1
                        continue
                    if abs(finters[fi][0]-finters[fij][0])<1e-5 and abs(finters[fi][1]-finters[fij][1])<1e-5:
                        finters.pop(fij)
                        if fi >= fij:
                            fi-=1
                        fij -= 1
                        flen-=1
                    fij +=1
                fi += 1


        #Checks for edge case where there are no intersections detected due to a rounding error (can sometimes happen with the last row)
        if flen == 0 and ri == nrows:
            ins = False

            #Checks if the predicted point (ghost point that was expected but not found) is inside one of the nogo zones
            for shape in nogo:
                if shape.pointintersect(highestVert):
                    ins = True
            if not ins:

                #Double checks that this borehole has not already been included
                if len(boreHoles)==0 or not (boreHoles[len(boreHoles) - 1][0] == highestVert[0] and boreHoles[len(boreHoles) - 1][1] ==highestVert[1]):
                    boreHoles[len(boreHoles)] = highestVert

        #Handles cases with odd number of intersections
        elif flen%2 ==0:

            #Specific case with two intersections
            if flen == 2:

                #Checks for the edge case where two intersections are very close together and replaces them with one point
                if sqrt((finters[0][0]-finters[1][0])*(finters[0][0]-finters[1][0])+(finters[0][1]-finters[1][1])*(finters[0][1]-finters[1][1]) )< XSpac:
                    ins = False
                    for ngShape in nogo:
                        if ngShape.pointintersect(highestVert):
                            ins = True
                    if not ins:
                        boreHoles[len(boreHoles)] = finters[0]
                        flen = 0 #skips the while loop

            i = 0
            while i < flen-1:

                leftOffset = [0,0]
                rightOffset = [0,0]

                #Checks if there is enough distance between this point and another and then will offset the point if there is not enough room
                if i>0 and (dls := sqrt((finters[i][0]-finters[i-1][0])*(finters[i][0]-finters[i-1][0]) + (finters[i][1]-finters[i-1][1])*(finters[i][1]-finters[i-1][1]))) < XSpac:
                    leftOffset = [dls*cos(rotate),dls*sin(rotate)]

                elif i<flen-1 and (drs := sqrt((finters[i][0]-finters[i+1][0])*(finters[i][0]-finters[i+1][0]) + (finters[i][1]-finters[i+1][1])*(finters[i][1]-finters[i+1][1]))) < XSpac:
                    rightOffset = [-drs*cos(rotate),-drs*sin(rotate)]

                ProcessRows(row, [finters[i][0]+leftOffset[0],finters[i][1]+leftOffset[1]], [finters[i+1][0]+rightOffset[0],finters[i+1][1]+rightOffset[1]], nogo, XSpac, boreHoles, rotate=rotate)

                i+=2
        elif flen == 1:
            ins = False
            for ngShape in nogo:
                if ngShape.pointintersect(highestVert):
                    ins = True
            if not ins:
                if len(boreHoles) == 0:
                    boreHoles[len(boreHoles)] = finters[0]
                if not (boreHoles[len(boreHoles)-1][0]==finters[0][0] and boreHoles[len(boreHoles)-1][1]==finters[0][1]):
                    boreHoles[len(boreHoles)]=finters[0]
        else:
            i = 0
            while i < flen-1:
                if field.pointintersect([(finters[i][0]+finters[i+1][0])/2,(finters[i][1]+finters[i+1][1])/2]):
                    ProcessRows(row, finters[i], finters[i + 1], nogo, XSpac,boreHoles,rotate=rotate)
                i += 1
        rowPoint[0] += rowspace[0]
        rowPoint[1] += rowspace[1]
    #if len(boreHoles) > maxLength:
        #return {}
    #print(boreHoles)
    rA = np.array([boreHoles[element] for element in boreHoles])
    #print(boreHoles)
    return boreHoles


def plotField (points,shape=None,shapes = None,title = "BoreField",SaveOrShow = True,Pdf = None):
    '''
     Function graphs given bore hole field
    Parameters
    -------------
        :param points: [[float]]
            array containing series of x,y points
        :param shape: object shape
            shape object representing obstruction used to generate series of x,y points
        :param title: string
            tile of figure
        :param SaveOrShow: boolean
            True if diplay is desired, False if figure should be saved
        :param Pdf: Pdf Object
            Object where figure should be saved
        :return: Nothing

    '''
    #plt.clf()
    fig = None
    if not SaveOrShow:
        fig = plt.figure(figsize=(10,7))
    plt.rcParams.update({"font.size": 18})
    plt.title(title)
    if shape != None:
        for i in range(len(shape.c)):
            if i == len(shape.c)-1:
                plt.plot([shape.c[i][0],shape.c[0][0]],[shape.c[i][1],shape.c[0][1]], "r")
            else:
                plt.plot([shape.c[i][0], shape.c[i+1][0]], [shape.c[i][1],shape.c[i+1][1]], "r")
    if shapes != None:
        for shape1 in shapes:
            for i in range(len(shape1.c)):
                if i == len(shape1.c)-1:
                    plt.plot([shape1.c[i][0],shape1.c[0][0]],[shape1.c[i][1],shape1.c[0][1]], "g")
                else:
                    plt.plot([shape1.c[i][0], shape1.c[i+1][0]], [shape1.c[i][1],shape1.c[i+1][1]], "g")
    X = points[:,0]
    Y = points[:,1]
    plt.plot(X,Y,"bo")
    plt.axis("equal")
    if SaveOrShow:
        plt.show()
    else:
        Pdf.savefig(fig)
    #plt.close()
    return


def ProcessRows (row, rowsx, rowex, nogo, rowspace,rA,rotate):
    #print("Processing Row")
    '''
    Function generates a row of the borefield
    *Note: the formatting from the rows can be a little unexpected. Some adjustment
    may be required to correct the formatting. The genBoreHoleConfig function already accounts for this.
    Parameters
    -------------
    :param BaseSLoc: [float,float]
        x,y location of the leftmost point of the trapezoid
    :param rowspace: float
        the spacing between rows
    :param shape: shape object
        object representing "no-go" zone
    :param XSpac: float
        minimum spacing between columns
    :param s1: float
        slope of the left side of the trapezoid
    :param BaseWidth: float
        with of the base of the trapezoid
    :param s2: float
        slope of the rights side of the trapezoid
    :param i: int
        the index of the currunt row
    :return: [[float]]
        two dimensional array containing the x,y values of the bore holes for this row
    '''
    if nogo == None:
        #print("There is no nogo zone")
        distribute(rowsx,rowex,rowspace,rA,rotate)
        return rA
    currentXP = rowsx
    ncol = int(sqrt((rowsx[0]-rowex[0])*(rowsx[0]-rowex[0])+(rowsx[1]-rowex[1])*(rowsx[1]-rowex[1])) // rowspace)

    inters = [point  for shape in nogo for point in shape.lineintersect(row,rotate=rotate)]
    inters = Shape.sortIntersections(inters,rotate)
    #print("Inters: ",inters)
    noin = len(inters)
    #if rowspace == 9.9:
     #   print("Correct Rowspace")
      #  print(rotate*(180/pi))
    #if rowspace == 9.9 and rotate-(-36.3*(pi/180)) < 1e-5 :
     #   print("For the row: ",row)
      #  print("The intersections are: ",inters)

    if noin > 1:
        if lessThan(inters[0],rowsx) and lessThan(rowex,inters[len(inters)-1]):
            inside = False
            for inter in inters:
                if lessThan(rowsx,inters[0]) and lessThan(inters[0],rowex):
                    inside = True
            if not inside:
                pointIn = False
                for ngShape in nogo:
                    if ngShape.pointintersect([(rowex[0]+rowsx[0])/2,(rowex[1]+rowsx[1])/2]):
                        pointIn = True
                if pointIn:
                    return
    inters = np.array(inters)
    indices = [j for j in range(noin) if not (lessThan(rowex,inters[j]) or lessThan(inters[j],rowsx))]
    inters = inters[indices]
    #print(inters)
    noin = len(inters)
    #print("noin value is: ",noin)
    for i in range(noin-1):
        #print("Inside for loop")
        spac = float(sqrt((inters[i+1][0]-inters[i][0])*(inters[i+1][0]-inters[i][0]) + (inters[i+1][1]-inters[i][1])*(inters[i+1][1]-inters[i][1])))
        if spac < rowspace:
            inone = False
            for shape in nogo:
                if shape.pointintersect([(inters[i+1][0]+inters[i][0])/2,(inters[i+1][1]+inters[i][1])/2]):
                    inone = True
            if inone:
                d = (rowspace-spac)/2
                inters[i+1][0] += d*cos(rotate)
                inters[i + 1][1] += d*sin(rotate)
                inters[i][0] -= d*cos(rotate)
                inters[i][1] -= d*sin(rotate)
    if ncol < 1:
        #print("ncol <1")
        ins = False
        for shape in nogo:
            if shape.pointintersect([(rowex[0]+rowsx[0])/2,(rowex[1]+rowsx[1])/2]):
                ins = True
        if not ins:
            if len(rA)==0 or not (rA[len(rA) - 1][0] == (rowex[0]+rowsx[0])/2 and rA[len(rA) - 1][1] == (rowex[1]+rowsx[1])/2):
                rA[len(rA)] = [(rowex[0]+rowsx[0])/2,(rowex[1]+rowsx[1])/2]
            return rA
    else:
        if noin == 0:
            #print("noin == 0")
            if notInside(rowsx,nogo) and notInside(rowex,nogo):
                distribute(rowsx, rowex, rowspace,rA,rotate)
        elif noin == 2:
            #print("noin == 2")
            distribute(rowsx, inters[0], rowspace,rA,rotate)
            distribute(inters[1], rowex, rowspace,rA,rotate)
        elif noin == 1:
            #print("noin == 1")
            #print("rowsx: [%f,%f], inters[0]: [%f,%f]"%(rowsx[0],rowsx[1],inters[0][0],inters[0][1]))
            ins = False
            for shape in nogo:
                if shape.pointintersect([(inters[0][0] + rowsx[0]) / 2, (inters[0][1] + rowsx[1]) / 2]):
                    ins = True
            if not ins:
                distribute(rowsx, inters[0],rowspace,rA,rotate)
            else:
                distribute(inters[0], rowex, rowspace, rA, rotate)
        elif noin % 2 == 0:
            #print("n%2 == 0")
            #print(inters)
            i=0
            while i < noin:
                if i == 0:
                    distribute(rowsx, inters[0],rowspace,rA,rotate)
                    #print(rA)
                    i = 1
                    continue
                elif i == noin - 1:
                    distribute(inters[noin - 1], rowex, rowspace,rA,rotate)
                else:
                    distribute(inters[i], inters[i + 1],rowspace,rA,rotate)
                i += 2
        else:
            #print("n%2 == 1")
            ins = False
            for shape in nogo:
                if shape.pointintersect([(inters[0][0] + rowsx[0]) / 2, (inters[0][1] + rowsx[1]) / 2]):
                    ins = True
            if not ins:
                i = 0
                while i < noin:
                    if i == 0:
                        distribute(rowsx, inters[0], rowspace, rA, rotate)
                        i = 1
                        continue
                    elif i == noin - 1:
                        i += 2
                        continue
                    else:
                        distribute(inters[i], inters[i + 1], rowspace, rA, rotate)
                    i += 2
            else:
                i = 0
                while i < noin:
                    if i == 0:
                        distribute(inters[0],inters[1], rowspace, rA, rotate)
                        i = 2
                        continue
                    elif i == noin - 1:
                        distribute(inters[i], rowex, rowspace, rA, rotate)
                        i += 2
                        continue
                    else:
                        distribute(inters[i], inters[i + 1], rowspace, rA, rotate)
                    i += 2

    return rA
def notInside(p,ngs):
    inside = False
    for ng in ngs:
        if ng.pointintersect(p):
            inside = True
    return not inside
def lessThan(p1,p2,rotate=0):
    x1,y1 = p1
    x2,y2 = p2
    phi1 = atan(y1/x1)
    R1 = sqrt(y1 * y1 + x1 * x1)
    refang1 = pi / 2 - phi1
    signx1 = 1
    if phi1 > pi / 2:
        if phi1 > pi:
            if phi1 > 3 * pi / 2.0:
                refang1 = 2 * pi - phi1
            else:
                refang1 = 3.0 * pi / 2.0 - phi1
        else:
            refang1 = pi - phi1
    if phi1 > pi/2 + rotate and phi1 < 3*pi/2 + rotate:
        signx1 = -1
    val1 = R1*sin(refang1+rotate)*signx1
    phi2 = 0
    if x2 == 0:
        phi2 = pi/2
    else:
        phi2 = atan(y2/x2)
    R2 = sqrt(y2 * y2 + x2 * x2)
    refang2 = pi / 2 - phi2
    signx2 = 1
    if phi2 > pi / 2:
        if phi2 > pi:
            if phi2 > 3 * pi / 2.0:
                refang2 = 2 * pi - phi1
            else:
                refang2 = 3.0 * pi / 2.0 - phi1
        else:
            refang2 = pi - phi2
    if phi2 > pi / 2 + rotate and phi2 < 3 * pi / 2 + rotate:
        signx2 = -1
    val2 = R2 * sin(refang2 + rotate) * signx2
    if val1 < val2:
        return True
    return False
def Between(x1,x2,x3):
    '''
    Function determines whenther x1 lies between x2 and x3
    Parameters
    -------------
    :param x1: [float,float]
        point 1
    :param x2: [float,float]
        point 2
    :param x3: [float,float]
        point 3
    :return:
    '''
    if x1 >= x2 and x1 <= x3:
        return True
    return False
def distribute(x1,x2,spacing,r,rotate):
    '''
      Function generates a series of boreholes between x1 and x2
    Parameters
    -------------
    :param x1: float
        left x value
    :param x2: float
        right x value
    :param y: float
        y value of row
    :param spacing: float
        spacing between columns
    :param r: [[float]]
        existing array of points
    :return:
    '''
    #print("made it here")
    #print("Left Point: (%f,%f), Right Point: (%f,%f)" % (x1[0],x1[1],x2[0],x2[1]))
    #print(spacing)
    #if spacing <= 0:
        #print("Received Invalid spacing value")
    dx = sqrt((x1[0]-x2[0])*(x1[0]-x2[0])+(x1[1]-x2[1])*(x1[1]-x2[1]))
    #print("The Value of dx is: ",dx)
    if dx < spacing:
        if len(r)==0 or not (r[len(r)-1][0]==(x1[0]+x2[0])/2 and r[len(r)-1][1]==(x1[1]+x2[1])/2):
            r[len(r)]=[(x1[0]+x2[0])/2,(x1[1]+x2[1])/2]
        #print([(x1[0]+x2[0])/2,(x1[1]+x2[1])/2])
        return
    currentX = x1
    initialX = [x1[0],x1[1]]
    actncol = int(dx // spacing)
    actSpac = dx / actncol
    #i=0
    #oldSpace = dx
    while (sqrt((currentX[0]-x2[0])*(currentX[0]-x2[0])+(currentX[1]-x2[1])*(currentX[1]-x2[1]))) >= (1e-8):
        #if currentX[0] - x2[0] <= (1e-15):
         #   r[len(r)] = [x2[0],x2[1]]
          #  break
        #oldSpace = sqrt((currentX[0]-x2[0])*(currentX[0]-x2[0])+(currentX[1]-x2[1])*(currentX[1]-x2[1]))
        #print("Final Values: (%f,%f). Distance: %f, Iteration: %f" % (x2[0],x2[1],sqrt((currentX[0]-x2[0])*(currentX[0]-x2[0])+(currentX[1]-x2[1])*(currentX[1]-x2[1])), i))
        #i+=1
        if len(r)==0 or not (r[len(r) - 1][0] == currentX[0] and r[len(r) - 1][1] == currentX[1]):
            r[len(r)]=[ currentX[0],currentX[1] ]
        #print([ currentX[0], x1[1] + ((x2[1]-x1[1])/(x2[0]-x1[0])) * currentX[0] ])
        currentX[0] += actSpac*cos(rotate)
        currentX[1] += actSpac*sin(rotate)
    if not (r[len(r) - 1][0] == x2[0] and r[len(r) - 1][1] == x2[1]):
        r[len(r)] = [x2[0],x2[1]]
    return



def main():
    vertices = [
        [3.55,5.04],
        [2.55,1.77],
        [5.21,4.10],
        [6.89,3.27],
        [8.31,5.45]
    ]
    nogoVertices1 = [
        [7,4],
        [7.2,5],
        [5,4.5]

    ]
    nogoVertices2 = [
        [4.5, 4],
        [4.8, 5],
        [3.5, 4.2],
        [3.8,3.8]

    ]
    ng1 = Shapes(nogoVertices1)
    ng2 = Shapes(nogoVertices2)
    ng = [ng1,ng2]
    field = Shapes(vertices)
    #print(field.pointintersect([5,2]))
    yS=.25
    xS=.3
    rt = -1
    holes = genBoreHoleConfig(field,yS,xS,rotate = rt, nogo = ng)
    if len(holes) < 1:
        #print("Returned Holes: ")
        #print(holes)
        return
    plotField(holes,shape = field, shapes = ng)


if __name__ == "__main__":
    main()