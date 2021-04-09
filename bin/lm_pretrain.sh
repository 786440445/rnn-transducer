#!/usr/bin/env bash

home_dir=$(cd $(dirname $0); cd ..; pwd)
echo $home_dir

if [ -f ${home_dir}/lm.log ]; then
    rm ${home_dir}/lm.log
fi
nohup python3 ${home_dir}/rnnt/lm_pretrain.py > lm.log &