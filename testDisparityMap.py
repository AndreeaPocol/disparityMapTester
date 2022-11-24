import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import slic
from skimage.color import label2rgb
from matplotlib import colors
from webcolors import name_to_rgb
from scipy.interpolate import interp1d

COLOR_DIFF_TRESH = math.sqrt(3) / 2  # TODO make a slider

code_2_color = {
    "definitelyWrongOcclusionError": "brown",  # (0, 20, 20)  # brown
    "definitelyWrongUnknown": "red",  # (0, 0, 255)  # red
    "definitelyWrongNoFuse": "maroon",  # (0, 0, 125)  # maroon
    "maybeWrongFuseColorMismatch": "magenta",  # (255, 0, 255)  # magenta
    "uncertainOcclusion": "orange",  # (0, 165, 255)  # orange
    "outOfBoundsOcclusion": "yellow",  # (0, 255, 255)  # yellow
    "maybeRight": "blue",  # (255, 0, 0)  # blue
    "definitelyRight": "green",  # (0, 255, 0)  # green
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
    plt.axes = None

    plt.show()


def pixelIsUnknown(pixelDisp):
    return pixelDisp == 0


def detect_outliers(data, threshold=3):
    outliers = []
    data_mean = np.mean(data)
    data_std = np.std(data)

    for y in data:
        z_score = (y - data_mean) / data_std
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


def plotHistogram(disps):
    plt.hist(x=disps, bins="auto", color="#0504aa", alpha=0.7, rwidth=0.85)
    plt.grid(axis="y", alpha=0.75)
    plt.xlabel("Disparity Value")
    plt.ylabel("Frequency")
    plt.title("Disparity value vs frequency")
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
    # - 50 segments & compactness = 10
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


def pixelDoesNotFuseProperly(r, c, d, leftOriginalImage, rightOriginalImage):

    m = interp1d([0, 255], [0, 1])
    c1 = leftOriginalImage[r][c]
    c2 = rightOriginalImage[r][c - d]
    distR = m(c1[0]) - m(c2[0])
    distG = m(c1[1]) - m(c2[1])
    distB = m(c1[2]) - m(c2[2])

    eDist = math.sqrt(pow(distR, 2) + pow(distG, 2) + pow(distB, 2))
    return ((c - d) < 0) or eDist > 3


def pixelIsOccludedFromBehind(r, c, leftDispMap, rightDispMap):
    P = [r, c]
    dispAtP = leftDispMap[r][c]  # d
    Q = [r, c - dispAtP]
    dispAtQ = rightDispMap[r][c - dispAtP]
    R = [r, c + dispAtQ]
    return P[1] > R[1]  # P is to the right of R


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

    definitelyWrong = (0, 0, 255)  # red
    maybeWrong = (0, 165, 255)  # orange
    maybeRight = (255, 0, 0)  # blue
    definitelyRight = (0, 255, 0)  # green

    disps = []
    for r in range(0, rows):
        for c in range(0, cols):
            curPixelDisp = leftDispMap[r][c]
            if (
                pixelIsUnknown(curPixelDisp)
                or pixelDoesNotFuseProperly(
                    r,
                    c,
                    curPixelDisp,
                    leftDispMap,
                    rightDispMap,
                    leftOriginalImage,
                    rightOriginalImage,
                )
                or pixelIsOccludedFromBehind(r, c, leftDispMap, rightDispMap)
            ):
                outputScore[r][c] = definitelyWrong
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
    # outliers = detect_outliers(disps)
    # print(outliers)
    # plotHistogram(disps)


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

    cv2.imshow("Colour-segmented image", segmentedImage)

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

    cv2.imshow("Original (left) disparity map", leftDispMap)
    cv2.imshow("Marked (left) disparity map", outputScore)
    cv2.imshow("Original (left) image", leftOriginalImage)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()
    cv2.imwrite(dispMapScoreOutputFile, outputScore)


if __name__ == "__main__":
    main()
