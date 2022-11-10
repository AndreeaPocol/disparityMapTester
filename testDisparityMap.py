import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.color import label2rgb
from matplotlib import colors


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


def processPixels(dispMap, outputScore, segments, segmentedImage, originalImage):

    rows = dispMap.shape[0]
    cols = dispMap.shape[1]

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
            curPixelDisp = dispMap[r][c]
            if pixelIsUnknown(curPixelDisp):
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
        # cv2.imshow("Segment {id}".format(id=segmentId), curSegment)
        # cv2.waitKey(0)

    # outliers = detect_outliers(disps)
    # print(outliers)
    # plotHistogram(disps)


def main():
    dispMapFile = ""
    originalImageFile = ""
    if len(sys.argv) == 4:
        dispMapFile = sys.argv[1]
        originalImageFile = sys.argv[2]
        dispMapScoreOutputFile = sys.argv[3]
    else:
        print(
            "Usage: {name} [ dispMapFile originalImage dispMapScoreOutputFile ]".format(
                name=sys.argv[0]
            )
        )
        exit

    dispMap = cv2.imread(dispMapFile, 0)  # grayscale mode
    outputScore = cv2.imread(dispMapFile, 1)  # colour mode
    originalImage = cv2.imread(originalImageFile, 1)

    segments, segmentedImage = segment(originalImage)

    # showColourDist(originalImage)
    processPixels(dispMap, outputScore, segments, segmentedImage, originalImage)

    cv2.imshow("Colour-segmented image", segmentedImage)
    cv2.imshow("Original disparity map", dispMap)
    cv2.imshow("Marked disparity map", outputScore)
    cv2.imshow("Original image", originalImage)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()
    cv2.imwrite(dispMapScoreOutputFile, outputScore)


if __name__ == "__main__":
    main()
