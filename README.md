This is a testing framework for stereo matching algorithms. *This work has been patented.*

# Examples of 3D Images
See `inputs` for an assorted test bed of 3D images. 

# Testing Disparity Maps

## Executable: `testDisparityMap.py`

### *To run:*

Set your preferred segmentation method in `config.py`. 

If your disparity maps are RGB image files (e.g., PNG, JPEG), set the `dispType` parameter to "RGB". If your disparity maps are PGM or PFM files, set `dispType` to "PGM" or "PFM". If your disparity maps are grayscale images, set `dispType` to "GRAY".
```
python3 testDisparityMap.py [dispType: RGB | PGM | PFM | GRAY] <leftDispMapFile> <rightDispMapFile> <leftOriginalImage> <rightOriginalImage> <fileOutputPrefix>
```

### *Examples:*
```
python3 disparityMapAssessment/testDisparityMap.py GRAY inputs/art/input/bad/left_disparity_art_bad.png inputs/art/input/bad/right_disparity_art_bad.png inputs/art/art_L.png inputs/art/art_R.png inputs/art/output/disparity_art_bad
```

```
python3 disparityMapAssessment/testDisparityMap.py RGB disparityMapAssessment/results/IGEV/grapes-output/left_disparity.png disparityMapAssessment/results/IGEV/grapes-output/right_disparity.png disparityMapAssessment/results/IGEV/grapes-output/grapes_L.png disparityMapAssessment/results/IGEV/grapes-output/grapes_R.png disparityMapAssessment/results/IGEV/grapes-output/grapes_
```

```
python3 testDisparityMap.py PFM inputs/artroom1/input/left_disparity_artroom1.pfm inputs/artroom1/input/right_disparity_artroom1.pfm inputs/artroom1/left_artroom1.png inputs/artroom1/right_artroom1.png inputs/artroom1/output/left_disparity_artroom1
```

Original image             |  Disparity map            |  Score                    | Legend
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
![image](inputs/art/art_L.png)  |  ![image](inputs/art/input/bad/left_disparity_art_bad.png)  |  ![image](inputs/art/output/disparity_art_bad_score_left.png) | ![image](disparityMapAssessment/legend.png)


### *Overview:*
Produces a visual score in the form of a color-coded image. 

Given a disparity map, `testDisparityMap` will:
1. Identify definitively wrong disparities if they result in definitive errors. Pixels in unknown regions of the disparity map are black, and always wrong. Pixels may encode occlusion in the wrong direction. Pixels may not fuse properly.
1. Segment the image based on both edges and color.
1. Extract all of the pixels from the image and disparity map that belong to a given segment. Consider all the disparities associated with that segment, keeping in mind that disparities should be continuous. Identify all the discrete, consecutive/contiguous disparity groups within the segment. If any group stands out from the rest, mark those disparities as potential outliers.
1. Repeat the previous step on the entire disparity map to identify potential global outliers.
1. Assign each pixel a color-coded score. Output an image based on the disparity map where the color of each pixel reflects the likelihood that the pixel is wrong.
1. Perform correction, using RANSAC plane fitting.



## Executable: `computeDisparityMapScore.py`

### *To run:*

```
python3 computeDisparityMapScore.py <dispMapScoreOutputFile>
```

### *Examples:*
```
python3 computeDisparityMapScore.py inputs/art/output/left_disparity_art_bad_score.png
```

### *Overview:*
Produces a numerical score.

# Generating Disparity Maps

## *Executable:* `generateDisparityMaps.py`

### *Example:*
```
python3 generateDisparityMaps.py inputs/aloe/aloe_L.png inputs/aloe/aloe_R.png inputs/aloe/opencv1/left_disparity_aloe.png inputs/aloe/opencv1/right_disparity_aloe.png
```

### *Overview:*

Generates a disparity map using OpenCV's StereoBM and StereoSGBM methods.

## *Executable:* `cre.ipynb`

### *Overview:*

Generates a disparity map using [CREStereo](https://openaccess.thecvf.com/content/CVPR2022/papers/Li_Practical_Stereo_Matching_via_Cascaded_Recurrent_Network_With_Adaptive_Correlation_CVPR_2022_paper.pdf). Requires CUDA. Can be run in Google Colab. Go to Edit -> Notebook Settings -> Hardware accelerator and select a GPU.

Can generate both left and right disparity maps. 

## *Executable:* `igev.ipynb`

### *Overview:*

Generates a disparity map using [IGEV-Stereo](https://arxiv.org/pdf/2303.06615.pdf). Requires CUDA. Can be run in Google Colab. Go to Edit -> Notebook Settings -> Hardware accelerator and select a GPU. 

Pre-trained models are avaiable [here]{https://github.com/gangweiX/IGEV}.

Can generate both left and right disparity maps. 


## *Executable:* `dlnr.ipynb`

### *Overview:*

Generates a disparity map using [DLNR](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf). Requires CUDA. Can be run in Google Colab. Go to Edit -> Notebook Settings -> Hardware accelerator and select a GPU. 

Contact the authors for the following required files:
* `DLNR_Middlebury.pth` (the pre-trained weights)
* `inference.py`

Cannot generate right disparity maps. 


## *Executable:* `raft.ipynb`

### *Overview:*

Generates a disparity map using [RAFT-Stereo](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf). Requires CUDA. Can be run in Google Colab. Go to Edit -> Notebook Settings -> Hardware accelerator and select a GPU. 

Pre-trained models available [here](https://github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network).

Cannot generate right disparity maps. 