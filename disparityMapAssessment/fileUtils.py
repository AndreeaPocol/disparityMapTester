import re
import cv2
import numpy as np
from config import COLORMAP

def roundInt(x):
    if x in [float("-inf"),float("inf")]: return 0
    return int(round(x))


def getScaleFromPMFile(dispMapFile):
    with open(dispMapFile, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if not dim_match:
            raise Exception("Malformed PFM header.")
        scale = float(pfm_file.readline().decode().rstrip()) # read disparity scale factor
        if scale < 0: # little-endian
            scale = -scale

    return scale


def convertRgbToGrayscale(color_image, side):
    # This solution is based on 
    # https://stackoverflow.com/questions/51824718/opencv-jetmap-or-colormap-to-grayscale-reverse-applycolormap
    # create an inverse from the colormap to gray values
    gray_values = np.arange(256, dtype=np.uint8)

    # NOTE: You must hardcode the colormap used to generate your RGB disparity maps.
    # See https://docs.opencv.org/4.x/d3/d50/group__imgproc__colormap.html for options.
    color_values = map(tuple, cv2.applyColorMap(gray_values, COLORMAP).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))

    # apply the inverse map to the false color image to reconstruct the grayscale image
    gray_image = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, color_image)

    # save reconstructed grayscale image
    # cv2.imwrite(f'grayscale_disp_map_{side}.png', gray_image)
    return gray_image


def visualizePfmImage(imgFile):
    img = cv2.imread(imgFile, -1)
    scale = getScaleFromPMFile(imgFile)
    img = img * scale

    cv2.imshow("Original (left) disparity map", img)
    cv2.waitKey(0)  # waits until a key is pressed
    cv2.destroyAllWindows()