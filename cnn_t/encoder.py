#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn-transducer -> encoder
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/18 2:43 PM
@Desc   ：
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F

from rnnt.utils import init_parameters


class BaseEncoder(nn.Module):
    def __init__(self, input_size, projection_size, hidden_size, output_size, vocab_size, n_layers, dropout=0.2, bidirectional=True):
        super(BaseEncoder, self).__init__()
        input_size = 400
        hidden_size = 1024
        projection_size = 300
        dropout = 0.3
        self.conv2d_1 = nn.Conv2d(1, 1, (3, 3), stride=1, padding=(1, 1))
        self.conv2d_2 = nn.Conv2d(1, 1, (5, 5), stride=1, padding=(2, 2))
        self.conv2d_3 = nn.Conv2d(1, 1, (7, 7), stride=1, padding=(3, 3))
        # self.conv2d_4 = nn.Conv2d(1, 1, (3, 3), stride=1, padding=(1, 1))
        # self.conv2d_5 = nn.Conv2d(1, 1, (3, 3), stride=1, padding=(1, 1))

        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(1)
        self.bn3 = nn.BatchNorm2d(1)
        # self.bn4 = nn.BatchNorm2d(1)
        # self.bn5 = nn.BatchNorm2d(1)

        self.softmax_linear = nn.Linear(400, vocab_size)
        self.linear = nn.Linear(400, 320)

    def forward(self, inputs, input_lengths):
        # 输入帧为10-20帧
        assert inputs.dim() == 3
        inputs = inputs.unsqueeze(1)
        inputs = self.bn1(inputs)
        outputs1 = self.conv2d_1(inputs)
        outputs1 = F.relu6(outputs1)

        outputs2 = self.bn1(outputs1)
        outputs2 = self.conv2d_2(outputs2)
        outputs2 = F.relu6(outputs2)

        outputs3 = self.bn1(outputs2)
        outputs3 = self.conv2d_3(outputs3)
        outputs3 = F.relu6(outputs3)
        outputs = outputs3.squeeze(1)

        logits = self.softmax_linear(outputs)
        outputs = self.linear(outputs)

        return logits, outputs


def build_encoder(config):
    if config.enc.type == 'lstm':
        return BaseEncoder(
            input_size=config.feature_dim,
            projection_size=config.enc.project_size,
            hidden_size=config.enc.hidden_size,
            output_size=config.enc.output_size,
            vocab_size=config.vocab_size,
            n_layers=config.enc.n_layers,
            dropout=config.dropout,
            bidirectional=config.enc.bidirectional
        )
    else:
        raise NotImplementedError
