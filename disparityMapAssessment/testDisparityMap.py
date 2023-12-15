import sys
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from webcolors import name_to_rgb
from scipy.interpolate import interp1d
from correction import fixDispMap
from fileUtils import *
from segmentation import *
from outliers import *
from config import *


def displayLegend():
    handles = []
    for code, color in code_2_color.items():
        handle = mpatches.Patch(
            color=color,
            label=code,
        )
        handles.append(handle)
    plt.legend(handles=handles)
    plt.title = "Legend"
    plt.show()


def pixelIsUnknown(pixelDisp):
    return pixelDisp == 0


def pixelDoesNotFuse(c, d):
    return (c - d) < 0


def pixelDoesNotFuseProperly(r, c, d, leftOriginalImage, rightOriginalImage):
    m = interp1d([0, 255], [0, 1])
    c1 = leftOriginalImage[r][c]
    c2 = rightOriginalImage[r][c - d]
    distR = m(c1[0]) - m(c2[0])
    distG = m(c1[1]) - m(c2[1])
    distB = m(c1[2]) - m(c2[2])

    eDist = math.sqrt(pow(distR, 2) + pow(distG, 2) + pow(distB, 2))
    return eDist > COLOR_DIFF_THRESH


def pixelIsOccluded(r, c, leftDispMap, rightDispMap):
    P = [r, c]
    dispAtP = roundInt(leftDispMap[P[0]][P[1]])
    Q = [r, P[1] - dispAtP]
    if Q[1] < 0:
        return "OOB"
    dispAtQ = roundInt(rightDispMap[Q[0]][Q[1]])
    R = [r, Q[1] + dispAtQ]
    if R[1] >= leftDispMap.shape[1]:
        return "OOB"
    dispAtR = roundInt(leftDispMap[R[0]][R[1]])
    T = [r, R[1] - dispAtR]
    if T[1] < 0:
        return "OOB"

    occlusion = abs(R[1] - P[1]) > 1
    if occlusion and (P[1] > R[1]):
        return "OCC_ERR"  # P is to the right of R, P is occluded from behind (OCCLUSION ERROR)
    elif occlusion:
        return "OCC"  # P is to the left of R, P is occluding (NORMAL)
    else:
        return "NO_OCC"


def processPixels(
    leftDispMap,
    rightDispMap,
    outputScore,
    segments,
    segmentedImage,
    leftOriginalImage,
    rightOriginalImage,
    newLeftDispMap
):
    rows = leftDispMap.shape[0]
    cols = leftDispMap.shape[1]

    segmentCoordsDict, segmentOutliersDict, globalOutliers = markOutliers(segments, outputScore, leftDispMap, rows, cols, segmentedImage)
    fixDispMap(segmentCoordsDict, segmentOutliersDict, globalOutliers, leftDispMap, newLeftDispMap, leftOriginalImage)

    for r in range(0, rows):
        for c in range(0, cols):
            curPixelDisp = roundInt(leftDispMap[r][c])
            if pixelIsUnknown(curPixelDisp):
                outputScore[r][c] = name_to_rgb(code_2_color["definitelyWrongUnknown"])
                continue
            occlusion = pixelIsOccluded(r, c, leftDispMap, rightDispMap)
            if occlusion == "OOB":
                outputScore[r][c] = name_to_rgb(code_2_color["outOfBoundsOcclusion"])
            elif occlusion == "OCC":
                outputScore[r][c] = name_to_rgb(code_2_color["uncertainOcclusion"])
                # TODO: use segmentation to identify whether some of these occluded pixels make sense
            elif occlusion == "OCC_ERR":
                outputScore[r][c] = name_to_rgb(
                    code_2_color["definitelyWrongOcclusionError"]
                )
            elif pixelDoesNotFuse(c, curPixelDisp):
                outputScore[r][c] = name_to_rgb(code_2_color["definitelyWrongNoFuse"])
            elif pixelDoesNotFuseProperly(
                r,
                c,
                curPixelDisp,
                leftOriginalImage,
                rightOriginalImage,
            ):
                outputScore[r][c] = name_to_rgb(
                    code_2_color["maybeWrongFuseColorMismatch"]
                )


def main():
    dispType = ""
    leftDispMapFile = ""
    rightDispMapFile = ""
    leftOriginalImageFile = ""
    rightOriginalImageFile = ""
    if len(sys.argv) == 7:
        dispType = sys.argv[1]
        leftDispMapFile = sys.argv[2]
        rightDispMapFile = sys.argv[3]
        leftOriginalImageFile = sys.argv[4]
        rightOriginalImageFile = sys.argv[5]
        fileOutputPrefix = sys.argv[6]
    else:
        print(
            "Usage: {name} [ dispType \
            leftDispMapFile \
            rightDispMapFile \
            leftOriginalImageFile \
            rightOriginalImageFile \
            fileOutputPrefix".format(
                name=sys.argv[0]
            )
        )
        exit()

    if dispType == "PGM" or dispType == "PFM":
        leftDispMap = cv2.imread(leftDispMapFile, -1)
        rightDispMap = cv2.imread(rightDispMapFile, -1)
        scale = getScaleFromPMFile(leftDispMapFile)
        leftDispMap = leftDispMap * scale
        rightDispMap = rightDispMap * scale
    elif dispType == "RGB":
        leftDispMap = cv2.imread(leftDispMapFile)
        rightDispMap = cv2.imread(rightDispMapFile)
        leftDispMap = convertRgbToGrayscale(leftDispMap, "left")
        rightDispMap = convertRgbToGrayscale(rightDispMap, "right")
    else:
        leftDispMap = cv2.imread(leftDispMapFile, 0)  # grayscale mode
        rightDispMap = cv2.imread(rightDispMapFile, 0)
    
    leftDispMap = leftDispMap/DISP_MAP_DIVISOR
    rightDispMap = rightDispMap/DISP_MAP_DIVISOR

    leftOutputScore = cv2.imread(leftOriginalImageFile, 1)  # colour mode
    leftOriginalImage = cv2.imread(leftOriginalImageFile, 1)
    rightOriginalImage = cv2.imread(rightOriginalImageFile, 1)

    newLeftDispMap = leftDispMap.copy()
    
    leftSegments, leftSegmentedImage = segment(leftOriginalImage)
    
    processPixels(
        leftDispMap,
        rightDispMap,
        leftOutputScore,
        leftSegments,
        leftSegmentedImage,
        leftOriginalImage,
        rightOriginalImage,
        newLeftDispMap
    )
    leftOutputScore = cv2.cvtColor(leftOutputScore, cv2.COLOR_BGR2RGB)

    if DISPLAY:
        cv2.imshow("Original (left) disparity map", leftDispMap)
        cv2.imshow("Marked (left) disparity map", leftOutputScore)
        cv2.imshow("Original (left) image", leftOriginalImage)
        cv2.imshow("Segmented (left) image", leftSegmentedImage)
        cv2.imshow("Corrected (left) disparity map", newLeftDispMap)
        # displayLegend()
        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()

    cv2.imwrite(fileOutputPrefix + "_score_left.png", leftOutputScore)
    cv2.imwrite(fileOutputPrefix + "_corrected_left.png", newLeftDispMap)

    if RIGHT_DISP_MAP:
        rightOutputScore = cv2.imread(rightOriginalImageFile, 1)  # colour mode
        newRightDispMap = rightDispMap.copy()
        rightSegments, rightSegmentedImage = segment(rightOriginalImage)
        processPixels(
            rightDispMap[:, ::-1],
            leftDispMap[:, ::-1],
            rightOutputScore[:, ::-1],
            rightSegments[:, ::-1],
            rightSegmentedImage[:, ::-1],
            rightOriginalImage[:, ::-1],
            leftOriginalImage[:, ::-1],
            newRightDispMap[:, ::-1]
        )

        rightOutputScore = cv2.cvtColor(rightOutputScore, cv2.COLOR_BGR2RGB)
        
        if DISPLAY:
            cv2.imshow("Original (right) disparity map", rightDispMap)
            cv2.imshow("Marked (right) disparity map", rightOutputScore)
            cv2.imshow("Original (right) image", rightOriginalImage)
            cv2.imshow("Segmented (right) image", rightSegmentedImage)
            cv2.imshow("Corrected (right) disparity map", newRightDispMap)

            # displayLegend()
            cv2.waitKey(0)  # waits until a key is pressed
            cv2.destroyAllWindows()
        cv2.imwrite(fileOutputPrefix + "_score_right.png", rightOutputScore)
        cv2.imwrite(fileOutputPrefix + "_corrected_right.png", newRightDispMap)

if __name__ == "__main__":
    main()
