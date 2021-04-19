#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn-transducer -> lm_pretrain
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/2 10:08 AM
@Desc   ：
=================================================='''
import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import yaml
import torch
from torch import nn as nn
import argparse
import numpy as np
from torch.nn.utils import clip_grad_norm_
import warnings

from rnnt.decoder import build_decoder
from rnnt.utils import AttrDict, count_parameters
from rnnt.dataset import AudioDataset

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
warnings.filterwarnings('ignore')


# Truncated backpropagation
def detach(states):
    return [state.detach() for state in states]


def get_pretrain_data(train_dataset, config, batch_size):
    ids = []
    for uuid, _ in train_dataset.sorted_list:
        value = train_dataset.target_ids.get(uuid)
        for id in value + [config.data.EOS_INDEX]:
            ids.append(id)
    ids = torch.LongTensor(ids)
    num_batches = ids.size(0) // batch_size
    ids = ids[:num_batches * batch_size]
    return ids.view(batch_size, -1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='train')
    parser.add_argument('-epochs', type=str, default=150)
    parser.add_argument('-batch_size', type=str, default=256)

    device = torch.device('cuda')

    opt = parser.parse_args()
    configfile = open(opt.config)

    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    # ==========================================
    # NETWORK SETTING
    # ==========================================
    # load model
    epochs = opt.epochs
    batch_size = opt.batch_size

    model = build_decoder(config.model)
    lm_dir = os.path.join(home_dir, 'lm_model')

    if not os.path.exists(lm_dir):
        os.makedirs(lm_dir)
    # 是否继续训练
    continue_train = False
    if continue_train:
        print('load lm pretrain model')
        lm_path = os.path.join(lm_dir,'decoder_LM_model')
        model.load_state_dict(torch.load(lm_path), strict=False)

    model.cuda()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    print(model)
    train_dataset = AudioDataset(config.data, 'train')
    inputs_x = get_pretrain_data(train_dataset, config, batch_size)

    seq_length = 40
    num_batchs = inputs_x.size(1) // seq_length

    n_params, enc, dec = count_parameters(model)
    print('# the number of parameters in the whole model: %d' % n_params)

    for epoch in range(epochs):
        total_loss = 0
        for i in range(0, inputs_x.size(1) - seq_length, seq_length):
            # Get mini-batch inputs and targets
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            inputs = inputs_x[:, i: i + seq_length].to(device)
            targets = inputs_x[:, i + 1: i + 1 + seq_length].to(device)
            logits, outputs, hidden = model(inputs, length=None, hidden=None)
            loss = criterion(logits, targets.reshape(-1))

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            step = (i + 1) // seq_length

            if step % 10 == 0:
                average_loss = total_loss / (step + 1)
                print('Epoch [{}/{}], Step[{}/{}], total_loss: {:.4f}, CurLoss: {:.4f}, Perplexity: {:5.2f}'
                      .format(epoch + 1, epochs, step, num_batchs, average_loss, loss.item(), np.exp(loss.item())))


    # Test the model
    with torch.no_grad():
        with open(os.path.join(os.getcwd(), 'sample.txt'), 'w', encoding='utf-8') as f:
            # Set intial hidden ane cell states
            state = (torch.zeros(1, 1, 512).to(device),
                     torch.zeros(1, 1, 512).to(device))

            # Select one word id randomly
            prob = torch.ones(config.model.vocab_size)
            input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)

            for i in range(1000):
                # Forward propagate RNN
                logits, outputs, hidden = model(input, length=None, hidden=state)
                # Sample a word id
                prob = logits.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()
                # Fill input with sampled word id for the next time step
                input.fill_(word_id)

                # File write
                word = train_dataset.index2word[word_id]
                word = '\n' if word == '<EOS>' else word
                f.write(word)

                if (i + 1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i + 1, 1000, 'sample.txt'))

    # Save the model checkpoints
    print('complete trained model save!')
    torch.save(model.state_dict(), os.path.join(lm_dir, 'decoder_LM_model'))