import sys
import cv2
from matplotlib import pyplot as plt


def generateDispMaps(
    stereo,
    leftOriginalImage,
    rightOriginalImage,
    leftDispMapFileOutputFile,
    rightDispMapFileOutputFile,
):
    disparity_left = stereo.compute(leftOriginalImage, rightOriginalImage)
    disparity_right = (
        stereo.compute(rightOriginalImage[:, ::-1], leftOriginalImage[:, ::-1])
    )[:, ::-1]
    # plt.imshow(disparity_left, "gray")
    # plt.imshow(disparity_right, "gray")
    # plt.show()
    # cv2.waitKey(0)
    cv2.imwrite(img=disparity_left, filename=leftDispMapFileOutputFile)
    cv2.imwrite(img=disparity_right, filename=rightDispMapFileOutputFile)


def main():
    leftOriginalImageFile = ""
    rightOriginalImageFile = ""
    leftDispMapFileOutputFile = ""
    rightDispMapFileOutputFile = ""
    if len(sys.argv) == 5:
        leftOriginalImageFile = sys.argv[1]
        rightOriginalImageFile = sys.argv[2]
        leftDispMapFileOutputFile = sys.argv[3]
        rightDispMapFileOutputFile = sys.argv[4]
    else:
        print(
            "Usage: {name} [ leftOriginalImage rightOriginalImageFile leftDispMapFileOutputFile rightDispMapFileOutputFile]".format(
                name=sys.argv[0]
            )
        )
        exit()

    leftOriginalImage = cv2.imread(leftOriginalImageFile, 0)
    rightOriginalImage = cv2.imread(rightOriginalImageFile, 0)

    stereoBM = cv2.StereoBM_create()
    stereoSGBM = cv2.StereoSGBM_create()

    generators = {"BM": stereoBM, "SGBM": stereoSGBM}
    for name, generator in generators.items():
        generateDispMaps(
            generator,
            leftOriginalImage,
            rightOriginalImage,
            leftDispMapFileOutputFile.replace("opencv", name),
            rightDispMapFileOutputFile.replace("opencv", name),
        )


if __name__ == "__main__":
    main()
