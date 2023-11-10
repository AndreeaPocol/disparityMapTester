import math
import cv2


OUTLIER_THRESH = 3
COLOR_DIFF_THRESH = math.sqrt(3) / 2  # TODO make a slider
DISPLAY = True
WINDOW_SIZE = 140
NUM_SEGS_SLIC = 80000
COLORMAP = cv2.COLORMAP_JET # cv2.COLORMAP_INFERNO

# segmentMethod = "segmentKMeans" # assigns pixels from completely different regions to the same segment
segmentMethod = "segmentSLIC"
# segmentMethod = "segmentMeanShift" # assigns pixels from completely different regions to the same segment
# segmentMethod = "hybrid"
# segmentMethod = "segmentOpenCVKMeans" # assigns pixels from completely different regions to the same segment
# segmentMethod = "segmentFelzenszwalb"
# segmentMethod = "segmentQuickshift"
# segmentMethod = "segmentWatershed"
# segmentMethod = "fineSegmentation"

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