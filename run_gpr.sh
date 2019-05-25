#!/bin/bash

NAME=$(date +"%Y%m%d%H%M%S")

declare -a ratios=(70 60 50 40 30 20)

for ratio in ${ratios[@]}
do
  python gpr.py --csv case1 --MC --aug_num 5 --ratio $ratio --folder $NAME
  python gpr.py --csv case2 --MC --aug_num 5 --ratio $ratio --folder $NAME
  python gpr.py --csv case3 --MC --aug_num 5 --ratio $ratio --folder $NAME
done
