#!/bin/bash

NAME=$(date +"%Y%m%d%H%M%S")

declare -a ratios=(20 30 40 50)

for ratio in ${ratios[@]}
do
  python model.py --csv case1 --ratio $ratio --folder $NAME
  python model.py --csv case2 --ratio $ratio --folder $NAME
  python model.py --csv case3 --ratio $ratio --folder $NAME
done
