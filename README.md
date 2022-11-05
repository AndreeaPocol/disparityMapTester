To run:
```
python3 testDisparityMap.py <dispMapFile> <originalImage> <dispMapScoreOutputFile>
```
Examples: 
```
python3 testDisparityMap.py inputs/art/input/left_disparity_art_bad.png inputs/art/art_left.png inputs/art/output/left_disparity_art_bad_score.png

python3 testDisparityMap.py inputs/art/input/left_disparity_art.png inputs/art/art_left.png inputs/art/output/left_disparity_art_bad_score.png
```

Given a disparity map, `testDisparityMap` will:
1. Identify definitively wrong disparities if they result in occlusion in the wrong direction or other obvious errors. Pixels in unknown regions of the disparity map are black, and always wrong.
1. Segment the image based on both edges and colour (e.g., using EDISON).
1. Extract all of the pixels from the image and disparity map that belong to that segment.
1. Compute the global disparity distribution. If the result is a nice bell curve with some pixels off to the side, those pixels are probably outliers. Conduct a similar analysis by segment/colour to identify more outliers.
1. Assign each pixel a colour-coded score.
1. Output an image based on the disparity map where the colour of each pixel reflects the likelihood that the pixel is wrong.
<!-- 1. [optional] perform correction, using RANSAC plane fitting. -->