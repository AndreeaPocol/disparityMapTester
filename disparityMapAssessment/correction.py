import pyransac3d as pyrsc
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb
from fileUtils import roundInt
from outliers import *
from config import *


def correctPixels(fix, segmentCoordsDict, leftDispMap, newLeftDispMap, outliers):
    for segmentId in segmentCoordsDict:
        if fix == "local" and not outliers[segmentId]:
            continue # no segment outliers to fix
        # construct the plane
        points = []
        for pixel in segmentCoordsDict[segmentId]:
            x = pixel[0]
            y = pixel[1]
            segmentPixelDisp = roundInt(leftDispMap[x][y])
            if segmentPixelDisp == 0:
                continue
            newPoint = [x, y, segmentPixelDisp]
            points.append(newPoint)
        plane = pyrsc.Plane()
        try:
            best_eq, _ = plane.fit(np.array(points), thresh=0.01)
        except:
            # print(f"[{fix} fix] Can't fit plane of size {len(points)}.")
            continue
        # if fix == "local":
        #     print(f"[{fix} fix] Fitting plane with {len(points)} points")
        # print("Plane equation: ", best_eq)
        if best_eq == []:
            continue
        # the plane equation is of the form: Ax+By+Cz+D, e.g., [0.720, -0.253, 0.646, 1.100]
        A = best_eq[0]
        B = best_eq[1]
        C = best_eq[2]
        D = best_eq[3]
        if C == 0:
            continue
        
        # use the plane to correct outliers
        for pixel in segmentCoordsDict[segmentId]:
            x = pixel[0]
            y = pixel[1]
            # (A * x) + (B * y) + (C * z) + D = 0
            z = (-D - (A * x) - (B * y))/C # if z < 0, it's because of the disparity jumps caused by bad segmentation
            segmentPixelDisp = roundInt(leftDispMap[x][y])
            # if z < 0:
            #     print(f"'correct' disparity is negative for segment {segmentId}: ", points, f", on disparity: [{x}, {y}, {segmentPixelDisp}]")
            # if z > 256:
            #     print(f"'correct' disparity is > 256 for segment {segmentId}: ", points, f", on disparity: [{x}, {y}, {segmentPixelDisp}]")
            
            if (fix == "global") and ((segmentPixelDisp in outliers) or (segmentPixelDisp == 0)):
                newLeftDispMap[x][y] = roundInt(z)
            elif (fix == "local") and (segmentPixelDisp in outliers[segmentId]):
                newLeftDispMap[x][y] = roundInt(z)


def fixDispMap(segmentCoordsDict, segmentOutliersDict, globalOutliers, leftDispMap, newLeftDispMap, leftOriginalImage):
    rows = leftDispMap.shape[0]
    cols = leftDispMap.shape[1]
    globalSegmentCoordsDict = {}
    # leftOriginalImage = increaseContrast(leftOriginalImage)
    segments = slic(leftOriginalImage, n_segments=NUM_SEGS_SLIC_GLOBAL, compactness=10)
    cv2.imshow("Broadly segmented image", label2rgb(segments, leftOriginalImage, kind="avg"))       
    # cv2.waitKey(0)

    for r in range(0, rows):
        for c in range(0, cols):
            disp = roundInt(leftDispMap[r][c])
            segmentCoords = [[r, c, disp]]
            segmentId = segments[r][c]
            if segmentId in globalSegmentCoordsDict:
                segmentCoords = segmentCoords + globalSegmentCoordsDict[segmentId]
            globalSegmentCoordsDict[segmentId] = segmentCoords
    
    segmentsToSplit = []
    segmentDispDict = {}
    globalSegmentOutliersDict = {}

    # remove segment outliers prior to correction
    for segmentId, coords in globalSegmentCoordsDict.items():
        segmentDisps = list(map(lambda x: x[2], coords))
        segmentDispDict[segmentId] = segmentDisps
        segmentOutliers, needToSplit = detectOutliersByContinuityHeuristic(segmentDisps, "segment", verbose=False)
        globalSegmentOutliersDict[segmentId] = segmentOutliers
        globalSegmentCoordsDict[segmentId] = list(filter(lambda x: x[2] == 0 or x[2] not in segmentOutliers, coords))
        if needToSplit == True:
            segmentsToSplit.append(segmentId)
    globalSegmentCoordsDict = splitSegments(segmentsToSplit, globalSegmentCoordsDict, segmentDispDict, globalSegmentOutliersDict, leftDispMap)
    
    correctPixels("global", globalSegmentCoordsDict, leftDispMap, newLeftDispMap, globalOutliers) # pass 1: correct unknown pixels
    correctPixels("local", segmentCoordsDict, leftDispMap, newLeftDispMap, segmentOutliersDict) # pass 2: correct segment outliers

