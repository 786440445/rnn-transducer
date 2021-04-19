import torch
import torch.nn as nn
import torch.nn.functional as F
from rnnt.swish import Swish

class BaseDecoder(nn.Module):
    def __init__(self, project_size, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, share_weight=False, bidirectional=False):
        super(BaseDecoder, self).__init__()

        self.swish = Swish()
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.lstm2 = nn.LSTM(
            input_size=project_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        self.norm_layer1 = nn.LayerNorm(hidden_size)
        self.norm_layer2 = nn.LayerNorm(project_size)

        self.project_layer1 = nn.Linear(hidden_size, project_size)
        self.project_layer2 = nn.Linear(hidden_size, project_size)

        self.output_proj = nn.Linear(hidden_size, output_size)
        self.softmax_linear = nn.Linear(output_size, vocab_size)
        if share_weight:
            self.embedding.weight = self.output_proj.weight

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

    def do_lstm(self, lstm, input1, length, hidden=None):
        if length is not None:
            lstm.flatten_parameters()
            output, indices = self.pad_sequence(input1, length)
            output, hidden = lstm(output, hidden)
            output = self.pack_sequence(output, indices)
            return output, hidden
        else:
            output, hidden = lstm(input1, hidden)
            return output, hidden

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.embedding(inputs)

        if length is not None:
            sorted_seq_lengths, indices = torch.sort(length, descending=True)
            embed_inputs = embed_inputs[indices]
            embed_inputs = nn.utils.rnn.pack_padded_sequence(
                embed_inputs, sorted_seq_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, hidden = self.lstm(embed_inputs, hidden)

        if length is not None:
            _, desorted_indices = torch.sort(indices, descending=False)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[desorted_indices]

        # embed_inputs = self.embedding(inputs)

        # embed_inputs = self.norm_layer1(embed_inputs)
        # outputs, _ = self.lstm1(embed_inputs, hidden)
        # outputs = F.relu(self.project_layer1(outputs))

        # outputs = self.norm_layer2(outputs)
        # outputs, hidden = self.lstm2(outputs)
        # outputs = F.relu(self.project_layer2(outputs))

        outputs = self.output_proj(outputs)
        # sfomax logits
        logits = outputs.reshape(outputs.size(0) * outputs.size(1), outputs.size(2))
        logits = self.softmax_linear(logits)

        return logits, outputs, hidden
        # embed_inputs = self.embedding(inputs)

        # embed_inputs = self.norm_layer1(embed_inputs)
        # outputs, _ = self.do_lstm(self.lstm1, embed_inputs, length, hidden)
        # outputs = F.relu(self.project_layer1(outputs))

        # outputs = self.norm_layer2(outputs)
        # outputs, hidden = self.do_lstm(self.lstm2, outputs, length)
        # outputs = F.relu(self.project_layer2(outputs))

        # outputs = self.output_proj(outputs)
        # logits = outputs.reshape(outputs.size(0) * outputs.size(1), outputs.size(2))
        # logits = self.softmax_linear(logits)
        # return logits, outputs, hidden

        # embed_inputs = self.embedding(inputs)
        # outputs = self.norm_layer1(embed_inputs)
        # if length is not None:
        #     sorted_seq_lengths, indices = torch.sort(length, descending=True)
        #     embed_inputs = embed_inputs[indices]
        #     embed_inputs = nn.utils.rnn.pack_padded_sequence(
        #         embed_inputs, sorted_seq_lengths, batch_first=True)
        #     self.lstm1.flatten_parameters()

        # outputs, _ = self.lstm1(outputs, hidden)

        # if length is not None:
        #     _, desorted_indices = torch.sort(indices, descending=False)
        #     outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #     outputs = outputs[desorted_indices]
        
        # outputs = self.swish(self.project_layer1(outputs))
        # outputs = self.norm_layer2(outputs)
        # if length is not None:
        #     sorted_seq_lengths, indices = torch.sort(length, descending=True)
        #     outputs = outputs[indices]
        #     outputs = nn.utils.rnn.pack_padded_sequence(
        #         outputs, sorted_seq_lengths, batch_first=True)
        #     self.lstm2.flatten_parameters()

        # outputs, hidden = self.lstm2(outputs, hidden)

        # if length is not None:
        #     _, desorted_indices = torch.sort(indices, descending=False)
        #     outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        #     outputs = outputs[desorted_indices]
        
        # outputs = self.swish(self.project_layer2(outputs))

        # outputs = self.output_proj(outputs)
        # logits = outputs.reshape(outputs.size(0) * outputs.size(1), outputs.size(2))
        # logits = self.softmax_linear(logits)
        # return logits, outputs, hidden


def build_decoder(config):
    if config.dec.type == 'lstm':
        return BaseDecoder(
            project_size=config.dec.project_size,
            hidden_size=config.dec.hidden_size,
            vocab_size=config.vocab_size,
            output_size=config.dec.output_size,
            n_layers=config.dec.n_layers,
            dropout=config.dropout,
            share_weight=config.share_weight,
            bidirectional=config.dec.bidirectional
        )
    else:
        raise NotImplementedError
