#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：RNNT-pytorch -> dataloader
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/10/26 5:59 PM
@Desc   ：
=================================================='''
import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import codecs
import json
from collections import Counter
import parser

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 生成wav-label path路径
def generate_path_txt(category, dict):
    data = ''
    count = 0
    for path, _, files in os.walk(os.path.join(data_dir, 'wav', category)):
        for file in files:
            cur_path = path + r'/' + file
            id = file.split('.')[0]
            label = dict.get(id)
            if label:
                count += 1
                data += cur_path + '\t' + label + '\n'
    with open(os.path.join(home_dir, 'wav_txt', 'aishell.' + category), 'w', encoding='utf-8') as f:
        f.writelines(data)


# xxx
def get_label_dict():
    dict = {}
    with open(os.path.join(data_dir, 'transcript/aishell_transcript_v0.8.txt'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip().split()
            id = line[0]
            label = ' '.join(line[1:])
            dict[id] = label
    return dict


# 生成vocab文件
def get_all_label(label):
    data = []
    for value in label.values():
        value = ''.join(value)
        data.extend(value)
    vocab = Counter(data)
    vocab_list = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    save_vocab = [item for (item, index) in vocab_list if item != ' ']
    save_vocab = ['<PAD>', '<SOS>', '<EOS>', '<UNK>', '<BLANK>'] + save_vocab
    print(len(save_vocab))
    with open(os.path.join(home_dir, 'labels_ch.txt'), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(save_vocab)[:-1])


# 构建预训练文本数据集，验证集，测试集
def obtain_language_txt():
    dev_data = ''
    train_data = ''
    for path, dirs, files in os.walk(os.path.join(home_dir, 'wav_txt')):
        for file in files:
            if file[-4:] == '.dev':
                with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip().split()
                        label = ''.join(line[1:])
                        dev_data += label + '\n'
            else:
                with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        line = line.strip().split()
                        label = ''.join(line[1:])
                        train_data += label + '\n'
    with open(os.path.join(home_dir, 'lm_txt/train_lm.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_data[:-1])
    with open(os.path.join(home_dir, 'lm_txt/dev_lm.txt'), 'w', encoding='utf-8') as f:
        f.writelines(dev_data[:-1])


if __name__ == '__main__':
    parser = parser.ArgumentParser()
    parser.add_argument('--data_dir', '/media/fengchengli/UUUU/speech_data/data_aishell')

    args = parser.parse_args()
    data_dir = args.data_dir

    dict = get_label_dict()
    generate_path_txt('train', dict)
    generate_path_txt('test', dict)
    generate_path_txt('dev', dict)
    get_all_label(dict)
    obtain_language_txt()
