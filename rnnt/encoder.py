import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rnnt.utils import init_parameters
from rnnt.swish import Swish
from rnnt.subsampling import Conv2dSubsampling4

class BaseEncoder(nn.Module):
    def __init__(self, input_size, projection_size, hidden_size, output_size, vocab_size, n_layers, dropout=0.2, bidirectional=True):
        super(BaseEncoder, self).__init__()
        self.swish = Swish()
        self.subsampling = Conv2dSubsampling4(80, 256, 0.1)
        # self.convs = nn.ModuleList([
        #     nn.Sequential(
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(
        #             in_channels=1, 
        #             out_channels=256,
        #             kernel_size=3,
        #             stride=2,
        #         ),
        #         self.swish),
        #     nn.Sequential(
        #         nn.ReflectionPad2d(1),
        #         nn.Conv2d(
        #             in_channels=256,
        #             out_channels=256, 
        #             kernel_size=3,
        #             stride=2,
        #         ),
        #         self.swish)
        #     ]
        # )
        # [B, L, 80, 1] -> [B, L/2, 80/2, 256] -> [B, L/4, 80/4, 256]
        # self.out = torch.nn.Sequential(nn.Linear(256 * input_size // 4, 256))
        # self.lstm_layers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.LSTM(
        #             input_size=input_size,
        #             hidden_size=hidden_size,
        #             num_layers=1,
        #             batch_first=True,
        #             dropout=dropout,
        #             bidirectional=bidirectional),
            
        #     )
        # ])

        self.lstm1 = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.lstm2 = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.lstm3 = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.lstm4 = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.lstm5 = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.lstm6 = nn.LSTM(
            input_size=projection_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.layer_norm1 = nn.LayerNorm(256)
        self.layer_norm2 = nn.LayerNorm(projection_size)
        self.layer_norm3 = nn.LayerNorm(projection_size)
        self.layer_norm4 = nn.LayerNorm(projection_size)
        self.layer_norm5 = nn.LayerNorm(projection_size)
        self.layer_norm6 = nn.LayerNorm(projection_size)
        if bidirectional:
            linear_hidden_size = 2 * hidden_size
        else:
            linear_hidden_size = hidden_size
        self.proj_layer1 = nn.Linear(linear_hidden_size, projection_size)
        self.proj_layer2 = nn.Linear(linear_hidden_size, projection_size)
        self.proj_layer3 = nn.Linear(linear_hidden_size, projection_size)
        self.proj_layer4 = nn.Linear(linear_hidden_size, projection_size)
        self.proj_layer5 = nn.Linear(linear_hidden_size, projection_size)
        self.proj_layer6 = nn.Linear(linear_hidden_size, projection_size)

        self.output_proj = nn.Linear(projection_size, output_size)
        self.linear_vocab = nn.Linear(output_size, vocab_size)

    def time_reduction(self, inputs):
        size = inputs.size()
        if int(size[1]) % 2 == 1:
            padded_inputs = F.pad(inputs, [0, 0, 0, 1, 0, 0])
            sequence_length = int(size[1]) + 1
        else:
            padded_inputs = inputs
            sequence_length = int(size[1])

        odd_ind = torch.arange(1, sequence_length, 2, dtype=torch.long)
        even_ind = torch.arange(0, sequence_length, 2, dtype=torch.long)

        odd_inputs = padded_inputs[:, odd_ind, :]
        even_inputs = padded_inputs[:, even_ind, :]

        outputs_stacked = torch.cat([even_inputs, odd_inputs], 2)

        return outputs_stacked

    def pad_sequence(self, inputs, input_lengths):
        sorted_seq_lengths, indices = torch.sort(input_lengths, descending=True)
        inputs = inputs[indices]
        inputs = nn.utils.rnn.pack_padded_sequence(inputs, sorted_seq_lengths, batch_first=True)
        return inputs, indices

    def pack_sequence(self, inputs, indices):
        _, desorted_indices = torch.sort(indices, descending=False)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(inputs, batch_first=True)
        outputs = outputs[desorted_indices]
        return outputs

    def do_lstm(self, lstm, input1, input_lengths):
        if input_lengths is not None:
            lstm.flatten_parameters()
            output, indices = self.pad_sequence(input1, input_lengths)
            output, hidden = lstm(output)
            output = self.pack_sequence(output, indices)
            return output, hidden
        else:
            return input1, None

    def forward(self, inputs, input_lengths):
        assert inputs.dim() == 3
        inputs = self.subsampling(inputs)
        # outputs = outputs.squeeze(1)
        # inputs = inputs.unsqueeze(1)
        # print('inputs', inputs.shape)
        # features = []
        # for module in self.convs:
            # inputs = module(inputs)
            # features.append(inputs)
        # x = features[-1]
        # b, c, t, f = x.size()

        # outputs = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        outputs1 = self.layer_norm1(inputs)
        outputs1, _ = self.do_lstm(self.lstm1, outputs1, input_lengths)
        outputs1 = self.swish(self.proj_layer1(outputs1))

        # [B, L, 256]
        outputs2 = self.layer_norm2(outputs1)
        outputs2, _ = self.do_lstm(self.lstm2, outputs2, input_lengths)
        # outputs2, _ = self.lstm2(outputs2)
        outputs2 = 1/2 * self.swish(self.proj_layer2(outputs2)) + 1/2 * outputs1
        # outputs2 = self.time_reduction(outputs2)

        # [B, L//2, 512]
        outputs3 = self.layer_norm3(outputs2)
        # self.lstm3.flatten_parameters()
        outputs3, _ = self.do_lstm(self.lstm3, outputs3, input_lengths)
        # outputs3, _ = self.lstm3(outputs3)
        outputs3 = self.swish(self.proj_layer3(outputs3))

        # [B, L//2, 256]
        outputs4 = self.layer_norm4(outputs3)
        # self.lstm4.flatten_parameters()
        outputs4, _ = self.do_lstm(self.lstm4, outputs4, input_lengths)
        # outputs4, _ = self.lstm4(outputs4)
        outputs4 = 1/2 * self.swish(self.proj_layer4(outputs4)) + 1/2 * outputs3
        # outputs4 = self.time_reduction(outputs4)
        
        # [B, L//2, 256]
        outputs5 = self.layer_norm5(outputs4)
        # self.lstm5.flatten_parameters()
        # outputs5, _ = self.lstm5(outputs5)
        outputs5, _ = self.do_lstm(self.lstm5, outputs5, input_lengths)
        outputs5 = self.swish(self.proj_layer5(outputs5))
        # [B, L//4, 256]

        outputs6 = self.layer_norm6(outputs5)
        # self.lstm6.flatten_parameters()
        # outputs6, hidden = self.lstm6(outputs6)
        outputs6, hidden = self.do_lstm(self.lstm6, outputs6, input_lengths)
        outputs6 = self.swish(self.proj_layer6(outputs6))

        out = self.output_proj(outputs6)
        logits = self.linear_vocab(out)

        return logits, out, hidden


def build_encoder(config):
    if config.enc.type == 'lstm':
        return BaseEncoder(
            input_size=config.input_size,
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
