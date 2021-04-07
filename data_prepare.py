#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn-transducer -> data_prepare
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/6 8:20 PM
@Desc   ：
=================================================='''
import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

from random import shuffle
from rnnt.dataset import AudioDataset
import argparse
from rnnt.utils import AttrDict
import yaml
import numpy as np
import math
import pickle
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class AudioData():
    def __init__(self):
        self.train_dataset = AudioDataset(config.data, 'train')
        self.wav_ids = self.train_dataset.wav_ids
        self.lengths = self.train_dataset.lengths
        shuffle(self.wav_ids)

    def pickle_all_data(self):
        store_dict = {}
        for index in tqdm(self.wav_ids):
            audio_path = self.train_dataset.audio_path_dict[index]
            targets_ids = np.array(self.train_dataset.target_ids[index])
            features = self.train_dataset.get_fbank_features(audio_path, config.data.features_dim)
            # features = self.train_dataset.concat_frame(features)
            # features = self.train_dataset.subsampling(features)
            features = self.train_dataset.build_LFR_features(features, 4, 3)
            # print(features.shape)
            origin_length = np.array(math.ceil(features.shape[0]) if features.shape[0] <= 500 else 500).astype(np.int64)
            inputs_length = np.array(math.ceil(features.shape[0] / 2) if features.shape[0] <= 500 else 250).astype(np.int64)
            targets_length = np.array(targets_ids.shape[0] if targets_ids.shape[0] <= 50 else 50).astype(np.int64)
            features = self.train_dataset.pad(features).astype(np.float32)
            targets = self.train_dataset.pad(targets_ids).astype(np.int64).reshape(-1)
            feature_cli = [features, origin_length, inputs_length, targets, targets_length]
            store_dict[index] = feature_cli
        pickle.dump(store_dict, os.path.join(home_dir, 'pkl/features.pkl'), protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    opt = parser.parse_args()
    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))

    dataset = AudioData()
    dataset.pickle_all_data()