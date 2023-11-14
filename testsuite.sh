

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
    CRE
    IGEV
)

if [ $# -ge 1 ] ; then
    if [ $1 == "report" ] ; then # GPU required
        echo "Generating a report of all tests..."
        for i in "${images[@]}"; do
            for g in "${generators[@]}"; do
                echo "Analyzing $i..."
                python3 disparityMapAssessment/testDisparityMap.py GRAY disparityMapAssessment/results/$g/$i-output/left_disparity.png disparityMapAssessment/results/$g/$i-output/right_disparity.png disparityMapAssessment/results/$g/$i-output/${i}_L.png disparityMapAssessment/results/$g/$i-output/${i}_R.png disparityMapAssessment/results/$g/$i-output/${i}_score
            done
        done
    elif [ $1 == "score" ] ; then
        echo "Computing scores..."
        for i in "${images[@]}"; do
            for g in "${generators[@]}"; do
                python3 disparityMapAssessment/computeDisparityMapScore.py disparityMapAssessment/results/$g/$i-output/${i}_score_left.png
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
            python3 disparityMapAssessment/testDisparityMap.py RGB disparityMapAssessment/results/$g/$i-output/left_disparity.png disparityMapAssessment/results/$g/$i-output/right_disparity.png disparityMapAssessment/results/$g/$i-output/${i}_L.png disparityMapAssessment/results/$g/$i-output/${i}_R.png disparityMapAssessment/results/$g/$i-output/${i}_score
            python3 disparityMapAssessment/computeDisparityMapScore.py $1/left_disparity_${1}_${g}_score.png
        done
    fi
fi