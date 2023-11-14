

#!/bin/sh
# Script for regenerating all test data output files.
# Usage: ./testsuite.sh <report>/<input-name>
# Run with `report` to run on all test images and
# provide a specific image name to run once. 

clear

images=(
    art
    flowerbed
    lilies
    statue
    sunflowers
    tree
    trees
    woods
    roses
    rock
    peaches
    mug
    grapes
)

generators=(
    # CRE
    IGEV
)

training_datasets=(
    ETH3D
    # SceneFlow
)


if [ $# -ge 1 ] ; then
    if [ $1 == "report" ] ; then # GPU required
        for i in "${images[@]}"; do
            for g in "${generators[@]}"; do
                for t in "${training_datasets[@]}"; do
                    echo "Analyzing $i ($g trained on $t)..."
                    python3 disparityMapAssessment/testDisparityMap.py GRAY disparityMapAssessment/results/$g/$t/$i/left_disparity.png disparityMapAssessment/results/$g/$t/$i/right_disparity.png disparityMapAssessment/results/$g/$t/$i/${i}_L.png disparityMapAssessment/results/$g/$t/$i/${i}_R.png disparityMapAssessment/results/$g/$t/$i/${i}_score
                done
            done
        done
    elif [ $1 == "score" ] ; then
        echo "Computing scores..."
        for i in "${images[@]}"; do
            for g in "${generators[@]}"; do
                for t in "${training_datasets[@]}"; do
                    python3 disparityMapAssessment/computeDisparityMapScore.py disparityMapAssessment/results/$g/$t/$i/${i}_score_left.png
                done
            done
        done
    elif [ $1 == "maps" ] ; then
        echo "Generating disparity maps"
        mkdir $1
        for i in "${images[@]}"; do
            cd inputs/$i/
            mkdir CRE
            mkdir IGEV
            cd ../..
            python3 disparityMapGeneration/generateDisparityMaps.py inputs/$i/${i}_L.png inputs/$i/${i}_R.png inputs/$i/opencv/left_disparity_${i}.png inputs/$i/opencv/right_disparity_${i}.png
        done
    else
        echo "Running single test..."
        mkdir $1
        for g in "${generators[@]}"; do
            for t in "${training_datasets[@]}"; do
                python3 disparityMapAssessment/testDisparityMap.py RGB disparityMapAssessment/results/$g/$t/$i/left_disparity.png disparityMapAssessment/results/$g/$t/$i/right_disparity.png disparityMapAssessment/results/$g/$t/$i/${i}_L.png disparityMapAssessment/results/$g/$t/$i/${i}_R.png disparityMapAssessment/results/$g/$t/$i/${i}_score
                python3 disparityMapAssessment/computeDisparityMapScore.py $1/left_disparity_${1}_${g}_score.png
            done
        done
    fi
fi