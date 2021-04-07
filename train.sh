#!/usr/bin/env bash

work_dir=$(cd $(dirname $0); pwd)
echo $work_dir


rm run.log
nohup python3 train.py > run.log &