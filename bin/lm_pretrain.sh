#!/usr/bin/env bash

home_dir=$(cd $(dirname $0); cd ..; pwd)
echo $home_dir

rm ${home_dir}/lm.log
nohup python3 ${home_dir}/rnnt/lm_pretrain.py > lm.log &