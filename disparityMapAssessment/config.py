import math

OUTLIER_THRESH = 3
COLOR_DIFF_TRESH = math.sqrt(3) / 2  # TODO make a slider
DISPLAY = True

# segmentMethod = "segmentKMeans"
# segmentMethod = "segmentSLIC"
# segmentMethod = "segmentFile"
# segmentMethod = "segmentMeanShift"
# segmentMethod = "hybrid"
# segmentMethod = "segmentOpenCVKMeans"
segmentMethod = "segmentFelzenszwalb"
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