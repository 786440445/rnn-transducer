#!/usr/bin/env bash

home_dir=$(cd $(dirname $0); cd ..; pwd)
echo $home_dir

rm ${home_dir}/ctc.log
nohup python3 ${home_dir}/ctc_pretrain.py > ctc.log &