#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn-transducer -> test
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/4 11:55 AM
@Desc   ：
=================================================='''
import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import torch
# torch.cuda.set_device(0)

import yaml
import shutil
import argparse

from rnnt.model import Transducer
from rnnt.utils import AttrDict, init_logger, count_parameters, computer_cer
from rnnt.dataset import AudioDataset

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

def test(config, model, test_dataset, validate_data, logger):
    model.eval()
    total_dist = 0
    total_word = 0
    batch_steps = len(validate_data)
    for step, (inputs, targets, origin_length, inputs_length, targets_length) in enumerate(validate_data):
        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        max_inputs_length = inputs_length.max().item()
        max_targets_length = targets_length.max().item()
        # inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]
        preds = model.recognize(inputs, inputs_length)
        transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()]
                       for i in range(targets.size(0))]
        print(''.join([test_dataset.index2word.get(index) for index in preds[0]]))
        print(''.join([test_dataset.index2word.get(index) for index in transcripts[0]]))

        dist, num_words = computer_cer(preds, transcripts)
        total_dist += dist
        total_word += num_words

        cer = total_dist / total_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Validation-Epoch:%d(%.5f%%), CER: %.5f %%' % (1, process, cer))

    logger.info('-Validation-Epoch:%4d, AverageCER: %.5f %%' %(1, cer))
    return cer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    
    exp_name = os.path.join('aishell/rnnt-model')
    if not os.path.isdir(exp_name):
        os.makedirs(exp_name)
    logger = init_logger(os.path.join(exp_name, opt.log))

    shutil.copyfile(opt.config, os.path.join(exp_name, 'config.yaml'))
    logger.info('Save config info.')

    # 测试数据集
    test_dataset = AudioDataset(config.data, 'test')
    validate_data = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.data.batch_size,
        shuffle=False, num_workers=4)
    logger.info('Load Dev Set!')

    if config.training.num_gpu > 0:
        torch.cuda.manual_seed(config.training.seed)
        torch.backends.cudnn.deterministic = True
    else:
        torch.manual_seed(config.training.seed)
    logger.info('Set random seed: %d' % config.training.seed)

    model = Transducer(config.model)

    if config.training.load_model:
        checkpoint = torch.load(config.training.load_model)
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.joint.load_state_dict(checkpoint['joint'])
        logger.info('Loaded model from %s' % config.training.load_model)
    elif config.training.load_encoder or config.training.load_decoder:
        if config.training.load_encoder:
            checkpoint = torch.load(config.training.load_encoder)
            model.encoder.load_state_dict(checkpoint['encoder'])
            logger.info('Loaded encoder from %s' %
                        config.training.load_encoder)
        if config.training.load_decoder:
            checkpoint = torch.load(config.training.load_decoder)
            model.decoder.load_state_dict(checkpoint['decoder'])
            logger.info('Loaded decoder from %s' %
                        config.training.load_decoder)

    if config.training.num_gpu > 0:
        model = model.cuda()
        if config.training.num_gpu > 1:
            device_ids = list(range(config.training.num_gpu))
            model = torch.nn.DataParallel(model, device_ids=device_ids)
        logger.info('Loaded the model to %d GPUs' % config.training.num_gpu)

    n_params, enc, dec = count_parameters(model)
    logger.info('# the number of parameters in the whole model: %d' % n_params)
    logger.info('# the number of parameters in the Encoder: %d' % enc)
    logger.info('# the number of parameters in the Decoder: %d' % dec)
    logger.info('# the number of parameters in the JointNet: %d' %
                (n_params - dec - enc))
    
    cer = test(config, model, test_dataset, validate_data, logger)
    logger.info('# Test CER: %.5f%%' % (cer))


if __name__ == '__main__':
    main()
