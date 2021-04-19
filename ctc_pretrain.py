#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn-transducer -> ctc_pretrain
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/2 6:59 PM
@Desc   ：
=================================================='''

import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import math
import yaml
import torch
from torch import nn as nn
import argparse
import numpy as np
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import warnings
from random import shuffle
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from tensorboardX import SummaryWriter

from rnnt.encoder import build_encoder
from rnnt.utils import AttrDict, count_parameters
from rnnt.dataset import AudioDataset, data_prefetcher
import torch.nn.functional as F
from rnnt.func_warm_up import *
from rnnt.utils import computer_cer

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

warnings.filterwarnings('ignore')

# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]


def eval():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    # 是否继续训练
    continue_train = True
    epochs = 100
    batch_size = 64
    learning_rate = 0.0001
    device = torch.device('cuda')

    opt = parser.parse_args()
    configfile = open(opt.config)

    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    # ==========================================
    # NETWORK SETTING
    # ==========================================
    # load model
    model = build_encoder(config.model)
    if continue_train:
        print('load ctc pretrain model')
        ctc_path = os.path.join(home_dir, 'ctc_model/44_0.1983_enecoder_model')
        model.load_state_dict(torch.load(ctc_path), strict=False)

    print(model)
    model = model.cuda(device)

    # 数据提取
    ctc_loss = torch.nn.CTCLoss()
    train_dataset = AudioDataset(config.data, 'train')
    training_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=config.data.shuffle, num_workers=32, pin_memory=True)

    dev_dataset = AudioDataset(config.data, 'dev')
    dev_data = torch.utils.data.DataLoader(
        dev_dataset, batch_size=batch_size,
        shuffle=False, num_workers=16, pin_memory=True)

    steps = len(train_dataset)

    # 优化器设置
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    param_group = opt.param_groups
    n_params, enc, dec = count_parameters(model)
    print('# the number of parameters in the whole model: %d' % n_params)

    summary_writer = SummaryWriter(os.path.join(home_dir, 'logs/'))
    old_wer = 100
    for epoch in range(epochs):
        i = 0
        total_loss = 0
        nums_batchs = len(train_dataset.wav_ids) // batch_size
        prefetcher = data_prefetcher(training_data)
        while True:
            i += 1
            inputs_x, targets_y, origin_length, ctc_length, targets_length = prefetcher.next()
            if inputs_x is None:
                break
            # print('inputs_x', inputs_x.shape)
            # print('targets_y', targets_y.shape)
            # print('origin_length', origin_length.shape)
            # print('ctc_length', ctc_length.shape)
            # print('targets_length', targets_length.shape)
            # Get mini-batch inputs and targets
            max_inputs_length = origin_length.cpu().numpy().max().item()
            max_targets_length = targets_length.cpu().numpy().max().item()
            inputs_x = inputs_x[:, :max_inputs_length, :]
            targets_y = targets_y[:, :max_targets_length]

            inputs_x, ctc_length = inputs_x.cuda(), ctc_length.cuda()
            targets_y, targets_length = targets_y.cuda(), targets_length.cuda()
            print('ctc_length', ctc_length.shape)
            logits, outputs, hidden = model(inputs_x, ctc_length)
            logits = logits.transpose(1, 0).log_softmax(2).requires_grad_()
            model.zero_grad()
            loss = ctc_loss(logits, targets_y, ctc_length, targets_length)
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            total_loss += loss.item()
            if i % 5 == 0:
                average_loss = total_loss / (i + 1)
                print('Epoch [{}/{}], Step[{}/{}], Lr: {:.6f}, total_loss: {:.5f}, CurLoss: {:.5f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, epochs, i, nums_batchs, param_group[0]['lr'], average_loss, loss.item(), np.exp(loss.item())))
                if i % 50 == 0:
                    summary_writer.add_scalar('loss', average_loss, (nums_batchs*epoch + i + 1))
        # def get_beam_result(preds, beam_size):
        #     result = []
        #     for pred in preds:
        #         length = pred.size(0)
        #         for index in range(length):
        #             logits = pred[index, :]
        #             local_best_scores, local_best_ids = torch.topk(logits, beam_size, dim=1)

        #     pass

        if average_loss < 2:
            # CTC 预测过程
            dev_prefetcher = data_prefetcher(dev_data)
            total_diff = 0
            total_nums = 0

            while True:
                inputs_x, targets_y, origin_length, inputs_length, targets_length = dev_prefetcher.next()
                if inputs_x is None:
                    break
                logits, _, _ = model(inputs_x, inputs_length)
                preds = F.softmax(logits, dim=2).detach()
                # preds = get_beam_result(preds, beam_size=10)
                preds = torch.argmax(preds, dim=2)
                preds = preds.cpu().numpy()
                targets_y = targets_y.cpu().numpy()
                def remove_blank(labels, blank=0):
                    new_labels = []
                    # 合并相同的标签
                    previous = None
                    for l in labels:
                        if l != previous:
                            new_labels.append(l)
                            previous = l
                    # 删除blank
                    new_labels = [l for l in new_labels if l != blank]
                    return new_labels
                pred_outs = [remove_blank(pred) for pred in preds]
                targets_y = [remove_blank(label) for label in targets_y]
                print(''.join(dev_dataset.index2word.get(index) for index in pred_outs[0]))
                print(''.join(dev_dataset.index2word.get(index) for index in targets_y[0]))
                diff, total = computer_cer(pred_outs, targets_y)
                # print(diff)
                # print(total)
                total_diff += diff
                total_nums += total
            wer = total_diff / total_nums * 100
            print('ctc model wer : {}%'.format(wer))
            if wer < old_wer:
                old_wer = wer
                print('complete trained model save!')
                save_path = os.path.join(home_dir, 'ctc_model')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                torch.save(model.state_dict(), 'ctc_model/{}_{:.4f}_enecoder_model'.format(epoch+1, total_loss / nums_batchs))
