import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import slic
from skimage.color import label2rgb
from matplotlib import colors
from webcolors import name_to_rgb
from scipy.interpolate import interp1d
import seaborn as sns

COLOR_DIFF_TRESH = math.sqrt(3) / 2  # TODO make a slider

code_2_color = {
    "definitelyWrongOcclusionError": "brown",
    "definitelyWrongUnknown": "red",
    "definitelyWrongNoFuse": "purple",
    "maybeWrongFuseColorMismatch": "pink",
    "uncertainOcclusion": "orange",
    "outOfBoundsOcclusion": "yellow",
    "maybeWrongSegmentOutlier": "magenta",
    "maybeRight": "blue",
    "definitelyRight": "green",
}


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
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()


def pixelIsUnknown(pixelDisp):
    return pixelDisp == 0


def detectOutliers(data, leftDispMap):
    rows = leftDispMap.shape[0]
    cols = leftDispMap.shape[1]

    outliers = []
    lower = np.percentile(data, 2.5)
    upper = np.percentile(data, 97.5)
    print("Lower bound: ", lower, "Upper bound: ", upper)

    for r in range(0, rows):
        for c in range(0, cols):
            curPixelDisp = leftDispMap[r][c]
            if curPixelDisp < lower or curPixelDisp > upper:
                outliers.append(curPixelDisp)
    return outliers, lower, upper


def plotHistogram(disps, lower, upper):
    # plt parameters
    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    plt.style.use('seaborn-dark-palette')
    plt.rcParams['axes.grid'] = True
    plt.rcParams["patch.force_edgecolor"] = True

    p = sns.histplot(disps, stat='density')

    plt.xlabel("Disparity Value")
    plt.ylabel("Frequency")
    plt.title("Disparity value vs frequency")

    for rectangle in p.patches:
        if rectangle.get_x() > upper or rectangle.get_x() < lower:
            rectangle.set_facecolor('red')
        else:
            rectangle.set_facecolor('blue')

    plt.axvline(lower, color='black')
    plt.axvline(upper, color='black')
    plt.show()


def showColourDist(img):
    h, s, v = cv2.split(img)
    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    pixel_colors = img.reshape((np.shape(img)[0] * np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()
    axis.scatter(
        h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker="."
    )
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.show()


def segment(img):
    # Applying Simple Linear Iterative
    # Clustering on the image
    segments = slic(img, n_segments=800, compactness=20)
    # Converts a label image into
    # an RGB color image for visualizing
    # the labeled regions.
    return segments, label2rgb(segments, img, kind="avg")


def displaySegments(segmentCoordsDict, segmentDispDict, segmentedImage):
    rows = segmentedImage.shape[0]
    cols = segmentedImage.shape[1]
    # for every segment...
    numSegments = len(segmentCoordsDict)
    print("Number of segments: {}".format(numSegments))
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
            plotHistogram(segmentDispDict[segmentId])
            cv2.waitKey(0)


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
    return eDist > COLOR_DIFF_TRESH


def pixelIsOccluded(r, c, leftDispMap, rightDispMap):
    P = [r, c]
    dispAtP = leftDispMap[P[0]][P[1]]
    Q = [r, P[1] - dispAtP]
    if Q[1] < 0:
        return "OOB"
    dispAtQ = rightDispMap[Q[0]][Q[1]]
    R = [r, Q[1] + dispAtQ]
    if R[1] >= leftDispMap.shape[1]:
        return "OOB"
    dispAtR = leftDispMap[R[0]][R[1]]
    T = [r, R[1] - dispAtR]
    if T[1] < 0:
        return "OOB"

    occlusion = (R != P) and (T != Q)
    if occlusion and (P[1] < R[1]) and (T[1] < Q[1]):
        return "OCC"  # P is to the left of R, T is to the left of Q; pixel is occluded from the front (NORMAL)
    elif occlusion and (P[1] > R[1]) and (T[1] > Q[1]):
        return "OCC_ERR"  # P is to the right of R, T is to the right of Q; pixel is occluded from behind (OCCLUSION ERROR)
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
):

    rows = leftDispMap.shape[0]
    cols = leftDispMap.shape[1]

    assert (segments.shape[0] == rows) and (segments.shape[1] == cols)

    segmentDispDict = {}
    segmentCoordsDict = {}
    disps = []

    for r in range(0, rows):
        for c in range(0, cols):
            curPixelDisp = leftDispMap[r][c]
            if pixelIsUnknown(curPixelDisp):
                outputScore[r][c] = name_to_rgb(code_2_color["definitelyWrongUnknown"])
                continue
            occlusion = pixelIsOccluded(r, c, leftDispMap, rightDispMap)
            if occlusion == "OOB": # TODO: should probably be counted with the other disps
                outputScore[r][c] = name_to_rgb(code_2_color["outOfBoundsOcclusion"])
            elif occlusion == "OCC": # TODO: should probably be counted with the other disps
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
            else:
                disps.append(curPixelDisp)
                # Update segment disparities in segmentDispDict.
                # segmentId is the label. We want to know all the
                # disparities in the segment with id segmentId
                segmentId = segments[r][c]
                segmentDisps = [curPixelDisp]
                if segmentId in segmentDispDict:
                    segmentDisps = segmentDisps + segmentDispDict[segmentId]
                segmentDispDict[segmentId] = segmentDisps
                # Update segment coordinates.
                # We want to know all the pixels
                # in the segment with id segmentId
                # and their coordinates.
                segmentCoords = [[r, c]]
                if segmentId in segmentCoordsDict:
                    segmentCoords = segmentCoords + segmentCoordsDict[segmentId]
                segmentCoordsDict[segmentId] = segmentCoords

    # displaySegments(segmentCoordsDict, segmentDispDict, segmentedImage)

    # globalOutliers, lower, upper = detectOutliers(disps, leftDispMap)
    # plotHistogram(disps, lower, upper)

    for segmentId in segmentDispDict:
        disps = segmentDispDict[segmentId]
        segmentOutliers, lower, upper = detectOutliers(
            np.array(disps), leftDispMap
        )
        plotHistogram(disps, lower, upper)
        for pixel in segmentCoordsDict[segmentId]:
            x = pixel[0]
            y = pixel[1]
            if leftDispMap[x][y] in segmentOutliers:
                outputScore[x][y] = name_to_rgb(
                        code_2_color["maybeWrongSegmentOutlier"]
                    )
            else:
                outputScore[x][y] = name_to_rgb(
                        code_2_color["maybeRight"]
                    )


def main():

    leftDispMapFile = ""
    rightDispMapFile = ""
    leftOriginalImageFile = ""
    rightOriginalImageFile = ""
    if len(sys.argv) == 6:
        leftDispMapFile = sys.argv[1]
        rightDispMapFile = sys.argv[2]
        leftOriginalImageFile = sys.argv[3]
        rightOriginalImageFile = sys.argv[4]
        dispMapScoreOutputFile = sys.argv[5]
    else:
        print(
            "Usage: {name} [ leftDispMapFile rightDispMapFile originalImage dispMapScoreOutputFile ]".format(
                name=sys.argv[0]
            )
        )
        exit

    leftDispMap = cv2.imread(leftDispMapFile, 0)  # grayscale mode
    rightDispMap = cv2.imread(rightDispMapFile, 0)  # grayscale mode
    outputScore = cv2.imread(leftDispMapFile, 1)  # colour mode
    leftOriginalImage = cv2.imread(leftOriginalImageFile, 1)
    rightOriginalImage = cv2.imread(rightOriginalImageFile, 1)

    segments, segmentedImage = segment(leftOriginalImage)

    # cv2.imshow("Colour-segmented image", segmentedImage)

    # showColourDist(originalImage)
    processPixels(
        leftDispMap,
        rightDispMap,
        outputScore,
        segments,
        segmentedImage,
        leftOriginalImage,
        rightOriginalImage,
    )

    outputScore = cv2.cvtColor(outputScore, cv2.COLOR_BGR2RGB)

    cv2.imshow("Original (left) disparity map", leftDispMap)
    cv2.imshow("Marked (left) disparity map", outputScore)
    cv2.imshow("Original (left) image", leftOriginalImage)
    displayLegend()
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()
    cv2.imwrite(dispMapScoreOutputFile, outputScore)


if __name__ == "__main__":
    main()
