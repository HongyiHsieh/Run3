#!/bin/bash

file="2022_02_postEE.txt"
for line in $(dasgoclient -query="/RSGravitonTo2G_kMpl-02_M-*_TuneCP5_13p6TeV_pythia8/Run3Summer22EENanoAODv13-133X_mcRun3_2022_realistic_postEE_ForNanov13_*/NANOAODSIM");
do
    prefix=$(echo "$line" | awk -F'/' '{split($2, a, "_"); print a[1] "_" a[2] "_" a[3]}')
    mass=$(echo "$prefix" | awk -F'-' '{print $3+0}')  # 提取M-后的数值

    if [ "$mass" -ge 1000 ]; then
        echo "$prefix $line"
    fi
done > $file

fetch.py --input $file -w Yolo