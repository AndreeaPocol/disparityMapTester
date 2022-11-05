import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


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


def processPixels(dispMap, outputScore):
    rows = dispMap.shape[0]
    cols = dispMap.shape[1]

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

    outliers = detect_outliers(disps)
    print(outliers)
    plotHistogram(disps)


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

    processPixels(dispMap, outputScore)

    cv2.imshow("Original disparity map", dispMap)
    cv2.imshow("Marked disparity map", outputScore)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()
    cv2.imwrite(dispMapScoreOutputFile, outputScore)


if __name__ == "__main__":
    main()
