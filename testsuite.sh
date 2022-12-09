

#!/bin/sh
# Script for regenerating all test data output files.
# Usage: ./testsuite.sh <report>/<input-name>
# Run with `report` to run on all test images and
# provide a specific image name to run once. 

clear

images=(
    aloe
    art
    baby
    bowling1
    bowling2
    lines
    teddy
    box
    cat
    cup
    cups
    globe1
    globe2
    grapes
    lilies
    peaches
    reflections
    rock
    roses
    sail
    sphere
    woods
)

generators=(
    # anu
    # asw
    # bp
    # dp
    # gc
    # rh
    # scanline
    BM
    SGBM
)

if [ $# -ge 1 ] ; then
    if [ $1 == "report" ] ; then
        echo "Generating a report of all outputs..."
        mkdir report
        for i in "${images[@]}"; do
            for j in "${generators[@]}"; do
                python3 testDisparityMap.py inputs/$i/$j/left_disparity_${i}.png inputs/$i/$j/right_disparity_${i}.png inputs/$i/${i}_L.png inputs/$i/${i}_R.png report/left_disparity_${i}_${j}_score.png
            done
        done
    elif [ $1 == "maps" ] ; then
        echo "Generating disparity maps"
        mkdir $1
        for i in "${images[@]}"; do
            cd inputs/$i/
            mkdir BM
            mkdir SGBM
            cd ../..
            python3 generateDisparityMaps.py inputs/$i/${i}_L.png inputs/$i/${i}_R.png inputs/$i/opencv/left_disparity_${i}.png inputs/$i/opencv/right_disparity_${i}.png
        done
    else
        echo "Running single test and writing output to test_data..."
        mkdir $1
        for g in "${generators[@]}"; do
            python3 testDisparityMap.py inputs/$1/$g/left_disparity_$1.png inputs/$1/$g/right_disparity_$1.png inputs/$1/$1_L.png inputs/$1/$1_R.png $1/left_disparity_$1_${g}_score.png
            python3 computeDisparityMapScore.py $1/left_disparity_$1_${g}_score.png
        done
    fi
fi