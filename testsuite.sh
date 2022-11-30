

#!/bin/sh
# Script for regenerating all test data output files.
# Usage: ./testsuite.sh <report>/<input-name>
# Run with `report` to run on all test images and
# provide a specific image name to run once. 

images=(
    teddy
    art
)

if [ $# -ge 1 ] ; then
    if [ $1 == "report" ] ; then
        echo "Generating a report of all outputs..."
        mkdir report
        for i in "${images[@]}"; do
        python3 testDisparityMap.py inputs/$i/input/left_disparity_$i_bad.png inputs/$i/input/right_disparity_$i_bad.png inputs/$i/$i_left.png inputs/$i/$i_right.png inputs/$i/output/left_disparity_$i_bad_score.png
        done
    else
        echo "Running single test and writing output to test_data..."
        python3 testDisparityMap.py inputs/$1/input/left_disparity_$1_bad.png inputs/$1/input/right_disparity_$1_bad.png inputs/$1/$1_left.png inputs/$1/$1_right.png inputs/$1/output/left_disparity_$1_bad_score.png
fi