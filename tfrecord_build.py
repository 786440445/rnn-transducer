import os
import sys
import argparse
import math
import yaml
import torch
from torch import nn as nn
import numpy as np
import tensorflow as tf
from tqdm import tqdm

home_dir = os.getcwd()
from rnnt.dataset import AudioDataset
from rnnt.utils import AttrDict

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def build_tfrecord(tf_file, type):
    dataset = AudioDataset(config.data, type)
    writer = tf.python_io.TFRecordWriter(tf_file)
    for features, origin_length, inputs_length, targets, targets_length in tqdm(dataset):
        feature_list = {'features': _bytes_feature(features.astype(np.float32).tostring()),
                        'targets': _bytes_feature(targets.astype(np.int32).tostring()),
                        'origin_length': _int64_feature(int(origin_length)),
                        'inputs_length': _int64_feature(int(inputs_length)),
                        'targets_length': _int64_feature(int(targets_length))}
                            
        item = tf.train.Example(features = tf.train.Features(feature=feature_list))
        writer.write(item.SerializeToString())    
    writer.close()
    print ("TFRecord文件已保存。") 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    opt = parser.parse_args()
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    tf_record_dir = '/media/fengchengli/UUUU/tf_records/aishell'
    if not os.path.exists(tf_record_dir):
        os.makedirs(tf_record_dir, exist_ok=True)

    train_tf_record_file = os.path.join(tf_record_dir, 'train_aishell.tfrecord')
    valid_tf_record_file = os.path.join(tf_record_dir, 'valid_aishell.tfrecord')

    build_tfrecord(train_tf_record_file, 'train')
    build_tfrecord(valid_tf_record_file, 'dev')