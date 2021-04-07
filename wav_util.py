#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：rnn-transducer -> wav_util
@IDE    ：PyCharm
@Author ：chengli
@Date   ：2020/11/18 6:19 PM
@Desc   ：
=================================================='''
import numpy as np

class wav_tool():
    def __init__(self, sample_rate=16000, frame_duration=0.025, frame_shift=0.010,
                 preemphasis=0.97, num_mel=40, low_freq=0, high_freq=None,
                 mean_norm_feat=True, mean_norm_wav=True, compute_stats=False):
        self.sample_rate = sample_rate
        self.win_size = int(np.floor(frame_duration * sample_rate))
        self.win_shift = int(np.floor(frame_shift * sample_rate))

        self.low_freq = low_freq
        if high_freq is None:
            self.high_freq = sample_rate // 2
        else:
            self.high_freq = high_freq

        self.preemphasis = preemphasis
        self.num_mel = num_mel
        self.fft_size = 2
        while self.fft_size < self.win_size:
            self.fft_size *= 2

        self.hamwin = np.hamming(self.win_size)

        self.make_mel_filterbank()

    def line2mel(self, freq):
        return 2595 * np.log10(1 + freq / 700)

    def mel2line(self, mel):
        return (10 ** (mel / 2595) - 1) * 700

    def make_mel_filterbank(self):
        # 把lo_freq和hi_freq从频率变成mel
        low_mel = self.line2mel(self.low_freq)
        # 8000->2840
        high_mel = self.line2mel(self.high_freq)

        mel_freqs = np.linspace(low_mel, high_mel, self.num_mel + 2)

        bin_width = self.sample_rate / self.fft_size

        mel_bins = np.floor(self.mel2line(mel_freqs) / bin_width)

        num_bins = self.fft_size // 2 + 1

        self.mel_filterbank = np.zeros([self.num_mel, num_bins])

        for i in range(self.num_mel):
            left_bin = int(mel_bins[i])
            center_bin = int(mel_bins[i+1])
            right_bin = int(mel_bins[i+2])

            up_slope = 1 / (center_bin - left_bin)

            for j in range(left_bin, center_bin):
                self.mel_filterbank[i, j] = (j - left_bin) * up_slope
            # 第二个点和第三个点是往下走的直线
            down_slope = -1 / (right_bin - center_bin)
            for j in range(center_bin, right_bin):
                self.mel_filterbank[i, j] = (j - right_bin) * down_slope



    def process_utterance(self, utterance):
        wav = self.dither(utterance)
        wav = self.pre_emphasize(wav)
        frames = self.wav_to_frames(wav)
        magspec = self.frames_to_magspec(frames)
        fbank = self.magspec_to_fbank(magspec)
        if (self.mean_normalize):
            fbank = self.mean_norm_fbank(fbank)

        if (self.compute_global_stats):
            self.accumulate_stats(fbank)

        return fbank


