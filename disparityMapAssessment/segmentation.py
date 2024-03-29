import cv2
import numpy as np
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.color import label2rgb
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import pairwise_distances_argmin
from sklearn.utils import shuffle
from skimage.segmentation import mark_boundaries
from skimage.filters import sobel
from skimage.color import rgb2gray
from config import *


def segmentWatershed(img):
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')
    print(segments_watershed)
    cv2.imshow("result", mark_boundaries(img, segments_watershed))
    cv2.waitKey(0)
    return segments_watershed, img


def fineSegmentWatershed(img, tileId):
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')
    
    # print(segments_watershed)
    # cv2.imshow("result", mark_boundaries(img, segments_watershed))
    # cv2.waitKey(0)

    flat_image = img.reshape((-1,3))
    flat_image = np.float32(flat_image)
    labeled = segments_watershed.flatten()

    # get the average color of each segment
    total = np.zeros((labeled.shape[0], 3), dtype=float)
    count = np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label] = total[label] + flat_image[i]
        count[label] += 1
    avg = total/count
    avg = np.uint8(avg)

    res = avg[labeled]
    result = res.reshape((img.shape))
    
    newSegments = segments_watershed.astype(str)
    for row in range(segments_watershed.shape[0]):
        for col in range(segments_watershed.shape[1]):
            oldLabel = segments_watershed[row, col]
            newSegments[row, col] = str(oldLabel) + tileId
    return newSegments, result


def segmentFelzenszwalb(img):
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
    print(segments_fz)
    cv2.imshow("result", mark_boundaries(img, segments_fz))
    cv2.waitKey(0)
    return segments_fz, img


def fineSegmentFelzenszwalb(img, tileId):
    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}, tile ID: {tileId}')
    newSegments = segments_fz.astype(str)
    for row in range(segments_fz.shape[0]):
        for col in range(segments_fz.shape[1]):
            oldLabel = segments_fz[row, col]
            newSegments[row, col] = str(oldLabel) + tileId
    # cv2.imshow("result with boundaries", mark_boundaries(img, segments_fz))
    # cv2.waitKey(0)
    return newSegments, img


def segmentQuickshift(img):
    segments_quick = quickshift(img, kernel_size=5, max_dist=10, ratio=0.5)
    print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
    print(segments_quick)
    cv2.imshow("result", mark_boundaries(img, segments_quick))
    cv2.waitKey(0)
    return segments_quick, img


def fineSegmentQuickshift(img, tileId):
    segments_quick = quickshift(img, kernel_size=5, max_dist=10, ratio=0.5)
    print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')
    newSegments = segments_quick.astype(str)
    for row in range(segments_quick.shape[0]):
        for col in range(segments_quick.shape[1]):
            oldLabel = segments_quick[row, col]
            newSegments[row, col] = str(oldLabel) + tileId
    # cv2.imshow("result with boundaries", mark_boundaries(img, segments_quick))
    # cv2.waitKey(0)
    return newSegments, img


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
    segments = slic(img, n_segments=NUM_SEGS_SLIC, compactness=10, max_num_iter=30) # 171434 segments -> no segmentation, 171433 -> false alarms
    # converts a label image into an RGB color image for visualizing the labeled regions.
    return segments, label2rgb(segments, img, kind="avg")


def fineSegmentSLIC(img, tileId):
    # applying Simple Linear Iterative Clustering on the image
    segments = slic(img, n_segments=4000, compactness=10, max_num_iter=30) # 8712 segments -> no segmentation, 8711 segments -> false alarms
    newSegments = segments.astype(str)
    # converts a label image into an RGB color image for visualizing the labeled regions.
    for row in range(segments.shape[0]):
        for col in range(segments.shape[1]):
            oldLabel = segments[row, col]
            newSegments[row, col] = str(oldLabel) + tileId
    return newSegments, label2rgb(segments, img, kind="avg")


def increaseContrast(img):
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((img, enhanced_img))
    cv2.imshow('Enhanced contrast', result)
    return enhanced_img


def fineSegmentation(img, window=WINDOW_SIZE):
    rows = img.shape[0]
    cols = img.shape[1]
    segments = []
    labeledImg = np.array([["000000000"]*cols for _ in range(rows)])
    tiles = np.array([["000000000"]*cols for _ in range(rows)])
    segmentedImg = img.copy()
    tileId = 'a'

    for x in range(0,rows,window):
        for y in range(0,cols,window): # TODO: leftover chunks unsegmented...find multiple?
            tile = img[x:x+window, y:y+window]
            tiles[x:x+window, y:y+window] = tileId
            if np.size(tile) == 0:
                continue
            labels, segments = fineSegmentSLIC(tile, tileId)
            # cv2.imshow("Segment", segments)
            # cv2.waitKey(0)
            labeledImg[x:x+window, y:y+window] = labels
            segmentedImg[x:x+window, y:y+window] = segments
            tileId = chr(ord(tileId) + 1)
            assert(tileId <= 'z')
    cv2.imshow("Segmented image", segmentedImg)
    # print(f"Labels {labeledImg}")
    # cv2.imshow("result", mark_boundaries(segmentedImg, tiles))
    cv2.waitKey(0)  # waits until a key is pressed
    return labeledImg, segmentedImg


def segment(img):
    img = increaseContrast(img)
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
    if segmentMethod == "fineSegmentation":
        return fineSegmentation(img)