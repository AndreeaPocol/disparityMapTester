import sys
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import label2rgb
from matplotlib import colors
from webcolors import name_to_rgb
from scipy.interpolate import interp1d
import seaborn as sns
import pyransac3d as pyrsc
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from skimage.segmentation import mark_boundaries
from skimage.filters import sobel
from skimage.color import rgb2gray
import re

COLOR_DIFF_TRESH = math.sqrt(3) / 2  # TODO make a slider
OUTLIER_THRESH = 3
DISPLAY = True

# segmentMethod = "segmentKMeans"
# segmentMethod = "segmentSLIC"
segmentMethod = "segmentFile"
# segmentMethod = "segmentMeanShift"
# segmentMethod = "hybrid"
# segmentMethod = "segmentOpenCVKMeans"
# segmentMethod = "segmentFelzenszwalb"
# segmentMethod = "segmentQuickshift"
# segmentMethod = "segmentWatershed"


code_2_color = {
    "definitelyWrongOcclusionError": "brown",
    "definitelyWrongUnknown": "black",
    "definitelyWrongNoFuse": "red",
    "maybeWrongFuseColorMismatch": "pink",
    "uncertainOcclusion": "orange",
    "outOfBoundsOcclusion": "yellow",
    "maybeWrongSegmentOutlier": "magenta",
    "maybeWrongGlobalOutlier": "purple",
    "maybeRight": "blue",
}


def roundInt(x):
    if x in [float("-inf"),float("inf")]: return 0
    return int(round(x))


def segmentWatershed(img):
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')
    print(segments_watershed)
    cv2.imshow("result", mark_boundaries(img, segments_watershed))
    cv2.waitKey(0)
    # exit(0)
    return segments_watershed, img


def segmentFelzenszwalb(img):
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
    print(segments_fz)
    cv2.imshow("result", mark_boundaries(img, segments_fz))
    cv2.waitKey(0)
    # exit(0)
    return segments_fz, img


def segmentQuickshift(img):
    segments_quick = quickshift(img, kernel_size=5, max_dist=10, ratio=0.5)
    print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
    print(segments_quick)
    cv2.imshow("result", mark_boundaries(img, segments_quick))
    cv2.waitKey(0)
    # exit(0)
    return segments_quick, img


def segmentMeanShift(img):
    # reduce noise
    img = cv2.medianBlur(img, 3)

    # flatten the image
    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)

    # meanshift
    bandwidth = estimate_bandwidth(flat_image, quantile=.02, n_samples=3000)
    ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flat_image)
    labeled = ms.labels_
    
    # get number of segments
    segments = np.unique(labeled)
    print('Number of segments: ', segments.shape[0])

    # get the average color of each segment
    total = np.zeros((segments.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res = avg[labeled]
    result = res.reshape((img.shape))
    labeled = labeled.reshape((img.shape[0], img.shape[1]))
    return labeled, result


def segmentKMeansColorQuant(img):
    n_colors = 20
    img = np.array(img, dtype=np.float64) / 255

    # load image and transform to a 2D numpy array.
    w, h, d = original_shape = tuple(img.shape)
    assert d == 3
    image_array = np.reshape(img, (w * h, d))

    print("Fitting model on a small sub-sample of the data")
    image_array_sample = shuffle(image_array, random_state=0, n_samples=1_000)
    kmeans = KMeans(n_clusters=n_colors, n_init="auto", random_state=0).fit(
        image_array_sample
    )
    # get labels for all points
    print("Predicting color indices on the full image (k-means)")
    labels = kmeans.predict(image_array)
    codebook_random = shuffle(image_array, random_state=0, n_samples=n_colors)
    print("Predicting color indices on the full image (random)")
    labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
    codebook = kmeans.cluster_centers_
    recreatedImg = codebook[labels].reshape(w, h, -1)
    # cv2.imshow(f"Quantized image ({n_colors} colors, K-Means)", recreatedImg)
    segments = labels_random.reshape((img.shape[0], img.shape[1]))
    return segments, recreatedImg


def segmentOpenCVKMeans(img):
    Z = img.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 20
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    resultImage = res.reshape((img.shape))
    labels = label.reshape((img.shape[0], img.shape[1]))
    return labels, resultImage


def segmentSLIC(img):
    # applying Simple Linear Iterative Clustering on the image
    segments = slic(img, n_segments=900, compactness=10)
    # converts a label image into an RGB color image for visualizing the labeled regions.
    return segments, label2rgb(segments, img, kind="avg")


def segmentsFromFile(imgSegFile):
    segments = []
    with open(imgSegFile, 'r') as f:
        for line in f.read().splitlines():
            segments.append(line.split(',')[:-1]) # omit trailing blank element
    # print(np.array(segments))
    return np.array(segments)


def segment(img):
    if segmentMethod == "segmentFile":
        return segmentsFromFile(img)
    if segmentMethod == "segmentSLIC":
        return segmentSLIC(img)
    if segmentMethod == "segmentKMeans":
        return segmentKMeansColorQuant(img)
    if segmentMethod == "segmentMeanShift":
        return segmentMeanShift(img)
    if segmentMethod == "segmentOpenCVKMeans":
        return segmentOpenCVKMeans(img)
    if segmentMethod == "hybrid":
        segments, segmentedImg = segmentMeanShift(img)
        return segmentSLIC(segmentedImg)
    if segmentMethod == "segmentFelzenszwalb":
        return segmentFelzenszwalb(img)
    if segmentMethod == "segmentQuickshift":
        return segmentQuickshift(img)
    if segmentMethod == "segmentWatershed":
        return segmentWatershed(img)


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
            # plotHistogram(segmentDispDict[segmentId])
            cv2.waitKey(0)


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
            curPixelDisp = roundInt(leftDispMap[r][c])
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


def plotHistogram(disps, lower, upper):
    # plt parameters
    plt.rcParams["figure.figsize"] = (10.0, 10.0)
    plt.rcParams["axes.grid"] = True
    plt.rcParams["patch.force_edgecolor"] = True

    p = sns.histplot(disps, stat="density")

    plt.xlabel("Disparity Value")
    plt.ylabel("Frequency")
    plt.title("Disparity value vs frequency")

    for rectangle in p.patches:
        if rectangle.get_x() > upper or rectangle.get_x() < lower:
            rectangle.set_facecolor("red")
        else:
            rectangle.set_facecolor("blue")

    plt.axvline(lower, color="black")
    plt.axvline(upper, color="black")
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

    occlusion = (R != P) and (T != Q)
    if occlusion and (P[1] < R[1]) and (T[1] < Q[1]):
        return "OCC"  # P is to the left of R, T is to the left of Q; pixel is occluded from the front (NORMAL)
    elif occlusion and (P[1] > R[1]) and (T[1] > Q[1]):
        return "OCC_ERR"  # P is to the right of R, T is to the right of Q; pixel is occluded from behind (OCCLUSION ERROR)
    else:
        return "NO_OCC"


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
            # Update segment coordinates.
            # We want to know all the pixels
            # in the segment with id segmentId
            # and their coordinates.
            segmentCoords = [[r, c]]
            if segmentId in segmentCoordsDict:
                segmentCoords = segmentCoords + segmentCoordsDict[segmentId]
            segmentCoordsDict[segmentId] = segmentCoords
    for segmentId in segmentDispDict:
        disps = list(filter(lambda x: x > 0, segmentDispDict[segmentId]))
        segmentOutliers = detectOutliersByContinuityHeuristic(disps, verbose=False)
        segmentOutliersDict[segmentId] = segmentOutliers
        for pixel in segmentCoordsDict[segmentId]:
            x = pixel[0]
            y = pixel[1]
            segmentPixelDisp = roundInt(leftDispMap[x][y])
            if segmentPixelDisp in segmentOutliers:
                outputScore[x][y] = name_to_rgb(
                    code_2_color["maybeWrongSegmentOutlier"]
                )
            else:
                outputScore[x][y] = name_to_rgb(code_2_color["maybeRight"])
    # displaySegments(segmentCoordsDict, segmentDispDict, segmentedImage)

    return segmentCoordsDict, segmentOutliersDict, globalOutliers


def fixDispMap(segmentCoordsDict, segmentOutliersDict, globalOutliers, leftDispMap, newLeftDispMap):
    for segmentId in segmentCoordsDict:
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
            best_eq, best_inliers = plane.fit(np.array(points), 0.01)
        except:
            print(f"Can't fit plane of length {len(points)}.")
            pass
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
            z = (-D - (A * x) - (B * y))/C
            segmentPixelDisp = roundInt(leftDispMap[x][y])
            if (segmentPixelDisp in segmentOutliersDict[segmentId]) or (segmentPixelDisp in globalOutliers) or (segmentPixelDisp == 0):
                newLeftDispMap[x][y] = z


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

    segmentCoordsDict, segmentOutliers, globalOutliers = markOutliers(segments, outputScore, leftDispMap, rows, cols, segmentedImage)
    fixDispMap(segmentCoordsDict, segmentOutliers, globalOutliers, leftDispMap, newLeftDispMap)

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


def convertRgbToGrayscale(color_image, side):
    # This solution is based on 
    # https://stackoverflow.com/questions/51824718/opencv-jetmap-or-colormap-to-grayscale-reverse-applycolormap
    # create an inverse from the colormap to gray values
    gray_values = np.arange(256, dtype=np.uint8)
    color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_JET).reshape(256, 3))
    color_to_gray_map = dict(zip(color_values, gray_values))

    # apply the inverse map to the false color image to reconstruct the grayscale image
    gray_image = np.apply_along_axis(lambda bgr: color_to_gray_map[tuple(bgr)], 2, color_image)

    # save reconstructed grayscale image
    cv2.imwrite(f'grayscale_disp_map_{side}.png', gray_image)
    return gray_image


def main():
    dispType = ""
    leftDispMapFile = ""
    rightDispMapFile = ""
    leftOriginalImageFile = ""
    rightOriginalImageFile = ""
    segmentFile = ""
    if len(sys.argv) == 8:
        dispType = sys.argv[1]
        leftDispMapFile = sys.argv[2]
        rightDispMapFile = sys.argv[3]
        leftOriginalImageFile = sys.argv[4]
        rightOriginalImageFile = sys.argv[5]
        dispMapScoreOutputFile = sys.argv[6]
        segmentFile = sys.argv[7]
    elif len(sys.argv) == 7:
        dispType = sys.argv[1]
        leftDispMapFile = sys.argv[2]
        rightDispMapFile = sys.argv[3]
        leftOriginalImageFile = sys.argv[4]
        rightOriginalImageFile = sys.argv[5]
        dispMapScoreOutputFile = sys.argv[6]
    else:
        print(
            "Usage: {name} [ dispType \
            leftDispMapFile \
            rightDispMapFile \
            leftOriginalImageFile \
            rightOriginalImageFile \
            dispMapScoreOutputFile \
            [optional] segmentFile]".format(
                name=sys.argv[0]
            )
        )
        exit()

    if dispType == "PGM" or dispType == "PFM":
        leftDispMap = cv2.imread(leftDispMapFile, -1)
        rightDispMap = cv2.imread(rightDispMapFile, -1)
        with open(leftDispMapFile, 'rb') as pfm_file:
            header = pfm_file.readline().decode().rstrip()
            dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
            if not dim_match:
                raise Exception("Malformed PFM header.")
            scale = float(pfm_file.readline().decode().rstrip()) # read disparity scale factor
            if scale < 0: # little-endian
                scale = -scale
        leftDispMap = leftDispMap * scale
        rightDispMap = rightDispMap * scale
        print(f"scale: {scale}")
    elif dispType == "RGB":
        leftDispMap = cv2.imread(leftDispMapFile)
        rightDispMap = cv2.imread(rightDispMapFile)
        leftDispMap = convertRgbToGrayscale(leftDispMap, "left")
        rightDispMap = convertRgbToGrayscale(rightDispMap, "right")
    else:
        leftDispMap = cv2.imread(leftDispMapFile, 0)  # grayscale mode
        rightDispMap = cv2.imread(rightDispMapFile, 0)
    outputScore = cv2.imread(leftOriginalImageFile, 1)  # colour mode
    leftOriginalImage = cv2.imread(leftOriginalImageFile, 1)
    rightOriginalImage = cv2.imread(rightOriginalImageFile, 1)
    newLeftDispMap = leftDispMap.copy()
    if segmentMethod == "segmentFile":
        segments = segment(segmentFile)
        segmentedImage = leftOriginalImage
    else:
        segments, segmentedImage = segment(leftOriginalImage)

    # showColourDist(originalImage)
    processPixels(
        leftDispMap,
        rightDispMap,
        outputScore,
        segments,
        segmentedImage,
        leftOriginalImage,
        rightOriginalImage,
        newLeftDispMap
    )

    outputScore = cv2.cvtColor(outputScore, cv2.COLOR_BGR2RGB)

    if DISPLAY:
        cv2.imshow("Original (left) disparity map", leftDispMap)
        cv2.imshow("Marked (left) disparity map", outputScore)
        cv2.imshow("Original (left) image", leftOriginalImage)
        cv2.imshow("Segmented (left) image", segmentedImage)
        cv2.imshow("Corrected (left) disparity map", newLeftDispMap)       
        # displayLegend()
        cv2.waitKey(0)  # waits until a key is pressed
        cv2.destroyAllWindows()

    cv2.imwrite(dispMapScoreOutputFile, outputScore)


if __name__ == "__main__":
    main()
