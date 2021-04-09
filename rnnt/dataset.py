import os, sys
home_dir = os.getcwd()
sys.path.append(home_dir)

import codecs
import math
import copy
import random
import warnings
import torchaudio
import numpy as np
import torch
import soundfile as sf
from sklearn import preprocessing
from tempfile import NamedTemporaryFile

from spec_augment import run_sepcagument
from python_speech_features import logfbank, mfcc
from specAugment.sparse_image_warp_zcaceres import sparse_image_warp

warnings.filterwarnings('ignore')

class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.features, self.targets, self.origin_length, self.inputs_length, self.targets_length = next(self.loader)
        except StopIteration:
            self.features = None
            self.targets = None
            self.origin_length = None
            self.inputs_length = None
            self.targets_length = None
            return

        with torch.cuda.stream(self.stream):
            self.features = self.features.cuda(non_blocking=True)
            self.targets = self.targets.cuda(non_blocking=True)
            self.origin_length = self.origin_length.cuda(non_blocking=True)
            self.inputs_length = self.inputs_length.cuda(non_blocking=True)
            self.targets_length = self.targets_length.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        features = self.features
        targets = self.targets
        origin_length = self.origin_length
        inputs_length = self.inputs_length
        targets_length = self.targets_length
        self.preload()
        return features, targets, origin_length, inputs_length, targets_length


class Dataset:
    def __init__(self, config, type):

        self.type = type
        self.name = config.name
        self.left_context_width = config.left_context_width
        self.right_context_width = config.right_context_width
        self.frame_rate = config.frame_rate
        self.apply_cmvn = config.apply_cmvn

        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        self.vocab = config.vocab
        self.word2index, self.index2word = self.get_vocab_map()
        self.config = config

        if self.type == 'train':
            self.audio_path_dict, self.transcripts_dict, self.wav_ids, self.target_ids = \
                self.get_audio_path(config.train)
        elif self.type == 'dev':
            self.audio_path_dict, self.transcripts_dict, self.wav_ids, self.target_ids = \
                self.get_audio_path(config.dev)
        elif self.type == 'test':
            self.audio_path_dict, self.transcripts_dict, self.wav_ids, self.target_ids = \
                self.get_audio_path(config.test)
        else:
            pass

    def __len__(self):
        raise NotImplementedError

    def get_vocab_map(self):
        vocab = []
        with codecs.open(self.vocab, 'r', encoding='utf-8') as fid:
            for line in fid:
                line = line.strip()
                if line != '':
                    vocab.append(line)
        word2index = dict([(word, index) for index, word in enumerate(vocab)])
        index2word = dict([(index, word) for index, word in enumerate(vocab)])
        return word2index, index2word

    def pad(self, inputs, max_length=550):
        dim = len(inputs.shape)
        if dim == 1:
            if max_length is None:
                max_length = self.max_target_length
            pad_zeros_mat = np.zeros([1, max_length - inputs.shape[0]], dtype=np.int32)
            padded_inputs = np.column_stack([inputs.reshape(1, -1), pad_zeros_mat])
        elif dim == 2:
            if inputs.shape[0] <= max_length:
                feature_dim = inputs.shape[1]
                pad_zeros_mat = np.zeros([max_length - inputs.shape[0], feature_dim])
                padded_inputs = np.row_stack([inputs, pad_zeros_mat])
            else:
                padded_inputs = inputs[:max_length, :]
        else:
            raise AssertionError(
                'Features in inputs list must be one vector or two dimension matrix! ')
        return padded_inputs

    def get_audio_path(self, transcript_file):
        audio_path = {}
        transcripts = {}
        wav_ids = []
        target_ids = {}
        with open(os.path.join(home_dir, transcript_file), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip()
                if line != '':
                    line = line.split('\t')
                    path = line[0]
                    id = path.split('/')[-1][:-4]
                    transcript = ''.join(line[1].split(' '))
                    target_id = [self.word2index.get(item, self.config.UNK_INDEX) for item in transcript]
                    audio_path[id] = path
                    transcripts[id] = transcript
                    target_ids[id] = target_id
                    wav_ids.append(id)
        return audio_path, transcripts, wav_ids, target_ids

    def cmvn(self, mat, stats):
        mean = stats[0, :-1] / stats[0, -1]
        variance = stats[1, :-1] / stats[0, -1] - np.square(mean)
        return np.divide(np.subtract(mat, mean), np.sqrt(variance))

    def concat_frame(self, features):
        time_steps, features_dim = features.shape
        concated_features = np.zeros(
            shape=[time_steps, features_dim *
                   (1 + self.left_context_width + self.right_context_width)],
            dtype=np.float32)
        # middle part is just the uttarnce
        concated_features[:, self.left_context_width * features_dim:
                          (self.left_context_width + 1) * features_dim] = features

        for i in range(self.left_context_width):
            # add left context
            concated_features[i + 1:time_steps,
                              (self.left_context_width - i - 1) * features_dim:
                              (self.left_context_width - i) * features_dim] = features[0:time_steps - i - 1, :]

        for i in range(self.right_context_width):
            # add right context
            concated_features[0:time_steps - i - 1,
                              (self.right_context_width + i + 1) * features_dim:
                              (self.right_context_width + i + 2) * features_dim] = features[i + 1:time_steps, :]

        return concated_features

    def subsampling(self, features):
        if self.frame_rate != 10:
            interval = int(self.frame_rate / 10)
            temp_mat = [features[i]
                        for i in range(0, features.shape[0], interval)]
            subsampled_features = np.row_stack(temp_mat)
            return subsampled_features
        else:
            return features


class AudioDataset(Dataset):
    def __init__(self, config, type):
        super(AudioDataset, self).__init__(config, type)
        self.config = config
        self.text = os.path.join(config.__getattr__(type), 'text')

        self.short_first = config.short_first
        self.features_dim = config.features_dim
        self.merge_num = config.merge_num
        self.flat_num = config.flat_num

        self.max_input_length = config.max_input_length
        self.max_target_length = config.max_target_length
        # if self.config.encoding:
        # self.targets_dict = self.get_targets_dict()

        if self.short_first and type == 'train':
            self.sorted_list = sorted(self.transcripts_dict.items(), key=lambda x: len(x[1]), reverse=False)
            print('-----sort data-----')
        else:
            self.sorted_list = None
            random.shuffle(self.wav_ids)

        self.check_speech_and_text()
        self.lengths = len(self.wav_ids)

    def __getitem__(self, index):
        if self.sorted_list is not None:
            utt_id = self.sorted_list[index][0]
        else:
            utt_id = self.wav_ids[index]

        audio_path = self.audio_path_dict[utt_id]
        targets_ids = self.target_ids[utt_id]
        targets_ids = np.array(targets_ids)
        if self.type == 'train':
            features = self.get_fbank_features(audio_path, self.features_dim)
        else:
            features = self.get_fbank_features_dev(audio_path, self.features_dim)
        # 实现拼接四帧，重合一帧的操作
        # [L, 80]
        # features = self.concat_frame(features)
        # [L, 320]
        # features = self.subsampling(features)
        if self.merge_num != 0 or self.flat_num != 0:
            features = self.build_LFR_features(features, self.merge_num, self.flat_num)

        origin_length = features.shape[0] if features.shape[0] <= self.max_input_length else self.max_input_length
        inputs_length = math.ceil(features.shape[0] if features.shape[0] <= self.max_input_length else self.max_input_length)
        inputs_length = inputs_length // 4
        targets_length = targets_ids.shape[0] if targets_ids.shape[0] <= self.max_target_length else self.max_target_length
        
        features = self.pad(features, self.max_input_length).astype(np.float32)
        targets = self.pad(targets_ids, self.max_target_length).astype(np.int64).reshape(-1)
        return features, targets, origin_length, inputs_length, targets_length

    def __len__(self):
        return self.lengths

    def build_LFR_features(self, inputs, m, n):
        """
        Actually, this implements stacking frames and skipping frames.
        if m = 1 and n = 1, just return the origin features.
        if m = 1 and n > 1, it works like skipping.
        if m > 1 and n = 1, it works like stacking but only support right frames.
        if m > 1 and n > 1, it works like LFR.
        Args:
            inputs_batch: inputs is T x D np.ndarray
            m: number of frames to stack
            n: number of frames to skip
        """
        LFR_inputs = []
        T = inputs.shape[0]
        T_lfr = int(np.ceil(T / n))
        for i in range(T_lfr):
            if m <= T - i * n:
                LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
            else:
                num_padding = m - (T - i * n)
                frame = np.hstack(inputs[i * n:])
                for _ in range(num_padding):
                    frame = np.hstack((frame, inputs[-1]))
                LFR_inputs.append(frame)
        return np.vstack(LFR_inputs)

    def check_speech_and_text(self):
        assert len(self.wav_ids) == len(self.transcripts_dict)
        assert len(self.wav_ids) == len(self.audio_path_dict)

    def get_fbank_features(self, audio_path, nfilt=200, normalize=True):
        """
        Fbank特征提取, 结果进行零均值归一化操作
        :param wav_file: 文件路径
        :return: feature向量y
        """
        # signal, sample_rate = sf.read(audio_path)
        signal = self.load_randomly_augmented_audio(audio_path)
        feature = logfbank(signal, 16000, nfilt=nfilt)
        feature = np.array(feature)
        feature = run_sepcagument(feature)
        if normalize:
            preprocessing.scale(feature)
        return feature

    def get_fbank_features_dev(self, audio_path, nfilt=200, normalize=True):
        """
        Fbank特征提取, 结果进行零均值归一化操作
        :param wav_file: 文件路径
        :return: feature向量y
        """
        signal = self.load_audio(audio_path)
        # signal, sample_rate = sf.read(audio_path)
        feature = logfbank(signal, 16000, nfilt=nfilt)
        feature = np.array(feature)
        # feature = run_sepcagument(feature)
        if normalize:
            preprocessing.scale(feature)
        return feature

    # 语音增益 与 语速增益
    def load_randomly_augmented_audio(self, path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                      gain_range=(-6, 8)):
        """
        Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
        Returns the augmented utterance.
        """
        low_tempo, high_tempo = tempo_range
        tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
        low_gain, high_gain = gain_range
        gain_value = np.random.uniform(low=low_gain, high=high_gain)
        audio = self.augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                            tempo=tempo_value, gain=gain_value)
        return audio

    def augment_audio_with_sox(self, path, sample_rate, tempo, gain):
        """
        Changes tempo and gain of the recording with sox and loads it.
        """
        with NamedTemporaryFile(suffix=".wav") as augmented_file:
            augmented_filename = augmented_file.name
            sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
            sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                          augmented_filename,
                                                                                          " ".join(sox_augment_params))
            os.system(sox_params)
            y = self.load_audio(augmented_filename)
            return y

    def load_audio(self, path):
        sound, _ = torchaudio.load(path)
        sound = sound.numpy().T
        if len(sound.shape) > 1:
            if sound.shape[1] == 1:
                sound = sound.squeeze()
            else:
                sound = sound.mean(axis=1)  # multiple channels, average
        return sound


def time_warp(spec, W=3):
    # [1, L, feature_dim]
    spec = spec.reshape([1, spec.shape[0], spec.shape[1]])
    num_rows = spec.shape[1]
    spec_len = spec.shape[2]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]
    assert len(horizontal_line_at_ctr) == spec_len

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len - W)]
    assert isinstance(point_to_warp, torch.Tensor)

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts, dest_pts = torch.tensor([[[y, point_to_warp]]]), torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)


def freq_mask(spec, F=15, num_masks=1, replace_with_zero=True):
    cloned = spec.clone()
    num_mel_channels = cloned.shape[1]
    for i in range(0, num_masks):
        f = random.randrange(0, F)
        f_zero = random.randrange(0, num_mel_channels - f)
        # avoids randrange error if values are equal and range is empty
        if (f_zero == f_zero + f): return cloned
        mask_end = random.randrange(f_zero, f_zero + f)
        if (replace_with_zero):
            cloned[0][f_zero:mask_end] = 0
        else:
            cloned[0][f_zero:mask_end] = cloned.mean()

    return cloned


def time_mask(spec, T=15, num_masks=1, replace_with_zero=True):
    cloned = spec.clone()
    len_spectro = cloned.shape[2]

    for i in range(0, num_masks):
        t = random.randrange(0, T)
        t_zero = random.randrange(0, len_spectro - t)
        # avoids randrange error if values are equal and range is empty
        if (t_zero == t_zero + t): return cloned
        mask_end = random.randrange(t_zero, t_zero + t)
        if (replace_with_zero):
            cloned[0][:, t_zero:mask_end] = 0
        else:
            cloned[0][:, t_zero:mask_end] = cloned.mean()
    return cloned