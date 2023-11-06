import numpy as np
from webcolors import name_to_rgb
from config import *
import cv2


def roundInt(x):
    if x in [float("-inf"),float("inf")]: return 0
    return int(round(x))


def group(L):
    if len(L) == 0:
        return []
    first = last = L[0]
    firstIdx = lastIdx = c = 0
    for n in L[1:]:
        c += 1
        if (n - 1 == last) or (n == last):  # Part of the group, bump the end
            last = n
            lastIdx = c
        else:  # Not part of the group, yield current group and start a new
            if firstIdx == lastIdx:
                yield [L[firstIdx]]
            else:
                yield L[firstIdx:lastIdx]
            first = last = n
            firstIdx = lastIdx = c
    if firstIdx == lastIdx:
        yield [L[firstIdx]]
    else:
        yield L[firstIdx:lastIdx]  # Yield the last group
        

def detectOutliersStatistically(data, leftDispMap):
    rows = leftDispMap.shape[0]
    cols = leftDispMap.shape[1]

    outliers = []
    lower = np.percentile(data, 2.5)
    upper = np.percentile(data, 97.5)
    print("Lower bound: ", lower, "Upper bound: ", upper)

    for r in range(0, rows):
        for c in range(0, cols):
            curPixelDisp = leftDispMap[r][c]
            # curPixelDisp = roundInt(leftDispMap[r][c])
            if curPixelDisp < lower or curPixelDisp > upper:
                outliers.append(curPixelDisp)
    return outliers, lower, upper


def detectOutliersByContinuityHeuristic(data, verbose=True):
    outliers = []
    data.sort()
    ranges = list(group(data))
    numRanges = len(ranges)
    if numRanges == 0:
        return []
    if verbose:
        print("\n", "num ranges: ", numRanges, "\n")
        for range in ranges:
            print("[{} ... {}] size {} ".format(range[0], range[-1], len(range)))
    if numRanges == 2:
        diff = ranges[1][0] - ranges[0][-1]
        # the outlier is the smallest range, provided there's a substantial gap
        if diff > OUTLIER_THRESH:
            if len(ranges[1]) > len(ranges[0]):
                outliers = ranges[0]
            elif len(ranges[1]) < len(ranges[0]):
                outliers = ranges[1]
    if numRanges > 2:
        # the outlier is the first or last range in the sorted list of ranges,
        # provided there's a substantial gap
        if (ranges[1][0] - ranges[0][-1]) > OUTLIER_THRESH:
            if len(ranges[1]) > len(ranges[0]):
                outliers += ranges[0]
        if (ranges[-1][0] - ranges[-2][-1]) > OUTLIER_THRESH:
            if len(ranges[-2]) > len(ranges[-1]):
                outliers += ranges[-1]
    if verbose:
        print("OUTLIER(S): ", outliers, "\n")
    return outliers


def displaySegments(segmentCoordsDict, segmentDispDict, segmentedImage):
    rows = segmentedImage.shape[0]
    cols = segmentedImage.shape[1]
    
    numSegments = len(segmentCoordsDict)
    print("Number of segments: {}".format(numSegments))

    # for every segment...
    for segmentId, segmentCoords in segmentCoordsDict.items():
        curSegment = np.copy(segmentedImage)
        # find only pixels pertaining to a single segment
        # (the rest should be black)
        for r in range(0, rows):
            for c in range(0, cols):
                if [r, c] not in segmentCoords:
                    curSegment[r][c] = (0, 0, 0)
        if len(segmentDispDict[segmentId]) > 4:
            cv2.imshow("Segment {id}".format(id=segmentId), curSegment)
            # plotHistogram(segmentDispDict[segmentId])
            cv2.waitKey(0)


def getPixelNeighbors(pixel, dispMap):
    ri = 0
    ci = 0
    r = pixel[0]
    c = pixel[1]
    rows = dispMap.shape[0]
    cols = dispMap.shape[1]
    neighbors = []
    for i in range(1, 9):
        if (i == 1):
            ri = -1
            ci = -1
        if (i == 2): 
            ri = -1
            ci = 0
        if (i == 3): 
            ri = -1
            ci = 1
        if (i == 4): 
            ri = 0
            ci = 1
        if (i == 5): 
            ri = 1
            ci = 1
        if (i == 6): 
            ri = 1
            ci = 0
        if (i == 7): 
            ri = 1
            ci = -1
        if (i == 8): 
            ri = 0
            ci = -1
        newR = r+ri
        newC = c+ci
        if not ((newR >= 0) and (newC >= 0) and (newR < rows) and (newC < cols)):
            continue
        n = [newR, newC, roundInt(dispMap[newR][newC])]
        neighbors += [n]
    return neighbors


def markOutliers(segments, outputScore, leftDispMap, rows, cols, segmentedImage):
    segmentDispDict = {}
    segmentCoordsDict = {}
    segmentOutliersDict = {}
    globalDisps = list(np.concatenate(np.asarray(leftDispMap)).flat)
    globalDisps = [roundInt(disp) for disp in globalDisps]
    data = list(filter(lambda x: x > 0, globalDisps))
    # globalOutliers, lower, upper = detectOutliersStatistically(data, leftDispMap)
    # plotHistogram(globalDisps, lower, upper)
    globalOutliers = detectOutliersByContinuityHeuristic(data, verbose=False)

    for r in range(0, rows):
        for c in range(0, cols):
            curPixelDisp = roundInt(leftDispMap[r][c])
            if curPixelDisp in globalOutliers:
                outputScore[r][c] = name_to_rgb(code_2_color["maybeWrongGlobalOutlier"])
            # Update segment disparities in segmentDispDict.
            # segmentId is the label. We want to know all the
            # disparities in the segment with id segmentId
            segmentId = segments[r][c]
            segmentDisps = [curPixelDisp]
            if segmentId in segmentDispDict:
                segmentDisps = segmentDisps + segmentDispDict[segmentId]
            segmentDispDict[segmentId] = segmentDisps
            # Update segment coordinates. We want to know all the pixels
            # in the segment with id segmentId and their coordinates.
            segmentCoords = [[r, c]]
            if segmentId in segmentCoordsDict:
                segmentCoords = segmentCoords + segmentCoordsDict[segmentId]
            segmentCoordsDict[segmentId] = segmentCoords
    segmentCoordsDictFinal = segmentCoordsDict.copy()
    for segmentId in segmentDispDict:
        disps = list(filter(lambda x: x > 0, segmentDispDict[segmentId]))
        segmentOutliers = detectOutliersByContinuityHeuristic(disps, verbose=False)
        for pixel in segmentCoordsDict[segmentId]:
            x = pixel[0]
            y = pixel[1]
            segmentPixelDisp = roundInt(leftDispMap[x][y])
            if segmentPixelDisp in segmentOutliers:
                # if one of pixel's neighbors outside the segment has a matching disparity,
                # move pixel to that segment
                neighbors = getPixelNeighbors(pixel, leftDispMap)
                neighborsFromOtherSegmentWithMatchingDisparity = list(
                    filter
                        (lambda x: 
                            abs(x[2] - segmentPixelDisp) <= OUTLIER_THRESH and segments[x[0]][x[1]] != segmentId, 
                        neighbors)
                )
                if len(neighborsFromOtherSegmentWithMatchingDisparity) == 0:
                    outputScore[x][y] = name_to_rgb(code_2_color["maybeWrongSegmentOutlier"])
                else:
                    n = neighborsFromOtherSegmentWithMatchingDisparity[0]
                    nSegmentId = segments[n[0]][n[1]]
                    # remove pixel's disparity from segment outliers
                    segmentOutliers = list(filter((segmentPixelDisp).__ne__, segmentOutliers))
                    # remove pixel from incorrect segment
                    segments[x][y] = nSegmentId
                    outputScore[x][y] = name_to_rgb(code_2_color["maybeRight"])
            else:
                outputScore[x][y] = name_to_rgb(code_2_color["maybeRight"])
        segmentOutliersDict[segmentId] = segmentOutliers
    # displaySegments(segmentCoordsDict, segmentDispDict, segmentedImage)
    
    return segmentCoordsDictFinal, segmentOutliersDict, globalOutliers