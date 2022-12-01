

#!/bin/sh
# Script for regenerating all test data output files.
# Usage: ./testsuite.sh <report>/<input-name>
# Run with `report` to run on all test images and
# provide a specific image name to run once. 

clear

images=(
    teddy
    baby
    aloe
)

generators=(
    anu
    asw
    bp
    dp
    gc
    rh
    scanline
)

if [ $# -ge 1 ] ; then
    if [ $1 == "report" ] ; then
        echo "Generating a report of all outputs..."
        mkdir report
        for i in "${images[@]}"; do
            for j in "${generators[@]}"; do
                python3 testDisparityMap.py inputs/$i/$j/left_disparity_$i.png inputs/$i/$j/right_disparity_$i.png inputs/$i/$i_L.png inputs/$i/$i_R.png report/left_disparity_$i_$j_score.png
            done
        done
    else
        echo "Running single test and writing output to test_data..."
        mkdir $1
        for j in "${generators[@]}"; do
            python3 testDisparityMap.py inputs/$1/$j/left_disparity_$1.png inputs/$1/$j/right_disparity_$1.png inputs/$1/$1_L.png inputs/$1/$1_R.png $1/left_disparity_$1_${j}_score.png
            python3 computeDisparityMapScore.py $1/left_disparity_$1_${j}_score.png
        done
    fi
fi