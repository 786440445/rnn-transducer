import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import shutil
import argparse
import warnings
import yaml
import time
import torch
import torch.nn as nn
import torch.utils.data
from ctc_pretrain import data_prefetcher
from rnnt.model import Transducer
from rnnt.optim import Optimizer
from rnnt.dataset import AudioDataset
from tensorboardX import SummaryWriter
from rnnt.utils import AttrDict, init_logger, count_parameters, save_model, computer_cer

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

warnings.filterwarnings('ignore')

store_loss = []


def train(epoch, config, model, training_data, training_preferdata, optimizer, logger, visualizer=None):

    model.train()
    start_epoch = time.process_time()
    total_loss = 0
    optimizer.epoch()
    batch_steps = len(training_data)
    step = 0
    while True:
        step += 1
        inputs, targets, origin_length, inputs_length, targets_length = training_preferdata.next()
        if inputs is None:
            break
        max_inputs_length = origin_length.cpu().numpy().max().item()
        max_targets_length = targets_length.cpu().numpy().max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]
        if config.optim.step_wise_update:
            optimizer.step_decay_lr()

        optimizer.zero_grad()
        start = time.process_time()
        loss = model(inputs, inputs_length, targets, targets_length)

        if config.training.num_gpu > 1:
            loss = torch.mean(loss)

        loss.backward()

        total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm)

        optimizer.step()

        if visualizer is not None:
            visualizer.add_scalar(
                'train_loss', loss.item(), optimizer.global_step)
            visualizer.add_scalar(
                'learn_rate', optimizer.lr, optimizer.global_step)

        avg_loss = total_loss / (step + 1)
        store_loss.append(avg_loss)
        if optimizer.global_step % config.training.show_interval == 0:
            end = time.process_time()
            process = step / batch_steps * 100
            logger.info('-Training-Epoch:%d(%.5f%%), Global Step:%d, Learning Rate:%.6f, Grad Norm:%.5f, Loss:%.5f, '
                        'AverageLoss: %.5f, Run Time:%.3f' % (epoch, process, optimizer.global_step, optimizer.lr,
                                                              grad_norm, loss.item(), avg_loss, end-start))
        # break
    end_epoch = time.process_time()
    logger.info('-Training-Epoch:%d, Average Loss: %.5f, Epoch Time: %.3f' %
                (epoch, total_loss / (step+1), end_epoch-start_epoch))
    return total_loss / batch_steps


def eval(epoch, config, model, validating_data, logger, dev_dataset, visualizer=None):
    model.eval()
    total_loss = 0
    total_dist = 0
    total_word = 0
    batch_steps = len(validating_data)
    for step, (inputs, targets, origin_length, inputs_length, targets_length) in enumerate(validating_data):

        if config.training.num_gpu > 0:
            inputs, inputs_length = inputs.cuda(), inputs_length.cuda()
            targets, targets_length = targets.cuda(), targets_length.cuda()

        max_inputs_length = origin_length.max().item()
        max_targets_length = targets_length.max().item()
        inputs = inputs[:, :max_inputs_length, :]
        targets = targets[:, :max_targets_length]

        preds = model.recognize(inputs, inputs_length)
        transcripts = [targets.cpu().numpy()[i][:targets_length[i].item()] for i in range(targets.size(0))]
        
        print(''.join([dev_dataset.index2word.get(index) for index in preds[0]]))
        print(''.join([dev_dataset.index2word.get(index) for index in transcripts[0]]))
        dist, num_words = computer_cer(preds, transcripts)
        total_dist += dist
        total_word += num_words

        cer = total_dist / total_word * 100
        if step % config.training.show_interval == 0:
            process = step / batch_steps * 100
            logger.info('-Validation-Epoch:%d(%.5f%%), CER: %.5f %%' % (epoch, process, cer))

    val_loss = total_loss / batch_steps
    logger.info('-Validation-Epoch:%4d, AverageLoss:%.5f, AverageCER: %.5f %%' %
                (epoch, val_loss, cer))

    if visualizer is not None:
        visualizer.add_scalar('cer', cer, epoch)

    return cer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', type=str, default='config/aishell.yaml')
    parser.add_argument('-log', type=str, default='train.log')
    parser.add_argument('-mode', type=str, default='train')
    opt = parser.parse_args()

    configfile = open(opt.config)
    config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
    config.data.short_first = False

    model_path = os.path.join(config.data.name, config.training.save_model)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
    logger = init_logger(os.path.join(model_path, opt.log))

    shutil.copyfile(opt.config, os.path.join(model_path, 'config.yaml'))
    logger.info('Save config info.')

    train_dataset = AudioDataset(config.data, 'train')
    training_data = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.data.batch_size * config.training.num_gpu,
        shuffle=config.data.shuffle, num_workers=16, pin_memory=True)
    logger.info('Load Train Set!')

    dev_dataset = AudioDataset(config.data, 'dev')
    validate_data = torch.utils.data.DataLoader(
        dev_dataset, batch_size=64,
        shuffle=False, num_workers=8, pin_memory=True)
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

    optimizer = Optimizer(model.parameters(), config.optim)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warm_up_with_step_lr)

    logger.info('Created a %s optimizer.' % config.optim.type)

    if opt.mode == 'continue':
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        logger.info('Load Optimizer State!')
    else:
        start_epoch = 0

    # create a visualizer
    if config.training.visualization:
        visualizer = SummaryWriter(os.path.join(model_path, 'log'))
        logger.info('Created a visualizer.')
    else:
        visualizer = None

    old_wer = 100
    for epoch in range(start_epoch, config.training.epochs):
        training_preferdata = data_prefetcher(training_data)

        train_loss = train(epoch, config, model, training_data, training_preferdata,
              optimizer, logger, visualizer)

        if config.training.eval_or_not and train_loss < 100:
            if train_loss < 100:
                wer = eval(epoch, config, model, validate_data, logger, dev_dataset, visualizer)
                if wer < old_wer:
                    save_name = os.path.join(model_path, 'epoch%d_%.4f.chkpt' % (epoch, train_loss))
                    save_model(model, optimizer, config, save_name)
                logger.info('Epoch %d model has been saved.' % epoch)

            if epoch >= config.optim.begin_to_adjust_lr:
                optimizer.decay_lr()
                # early stop
                if optimizer.lr < 1e-7:
                    logger.info('The learning rate is too low to train.')
                    break
                logger.info('Epoch %d update learning rate: %.6f' %
                            (epoch, optimizer.lr))

    logger.info('The training process is OVER!')


if __name__ == '__main__':
    main()