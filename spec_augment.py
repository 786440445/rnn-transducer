#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn-transducer -> spec_augment
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/6 5:26 PM
@Desc   ：
=================================================='''

import random


def run_sepcagument(data_input, F=10, W=10):
    # 开始对得到的特征应用SpecAugment
    mode = random.randrange(0, 100)
    h_width = random.randint(0, W-1)
    h_start = random.randint(0, data_input.shape[0]-h_width-1)

    v_width = random.randint(0, F-1)
    v_start = random.randint(0, data_input.shape[1]-v_width-1)

    if (mode < 70):  # 正常特征 60%
        pass
    elif (mode >= 70 and mode < 80):  # 横向遮盖15帧
        if h_width == 0:
            pass
        else:
            data_input[h_start: h_start + h_width, :] = 0
    elif (mode >= 80 and mode < 90):  # 纵向遮盖 15帧
        if v_width == 0:
            pass
        else:
            data_input[:, v_start: v_start + v_width] = 0
    else:  # 两种遮盖叠加 10%
        if h_width == 0:
            pass
        else:
            data_input[h_start: h_start + h_width, :] = 0

        if v_width == 0:
            pass
        else:
            data_input[:, v_start: v_start + v_width] = 0
    return data_input