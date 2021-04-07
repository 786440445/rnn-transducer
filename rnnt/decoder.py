import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseDecoder(nn.Module):
    def __init__(self, project_size, hidden_size, vocab_size, output_size, n_layers, dropout=0.2, share_weight=False):
        super(BaseDecoder, self).__init__()
        # hidden_size = 300
        # project_size = 128
        hidden_size = 512
        project_size = 300
        dropout = 0.2
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.lstm1 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )
        self.lstm2 = nn.LSTM(
            input_size=project_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=dropout
        )
        self.norm_layer = nn.LayerNorm(hidden_size)
        self.norm_layer1 = nn.LayerNorm(project_size)

        self.project_layer1 = nn.Linear(hidden_size, project_size)
        self.project_layer2 = nn.Linear(hidden_size, project_size)

        self.output_proj = nn.Linear(project_size, output_size)
        self.softmax_linear = nn.Linear(output_size, vocab_size)
        if share_weight:
            self.embedding.weight = self.output_proj.weight

    def forward(self, inputs, length=None, hidden=None):
        embed_inputs = self.embedding(inputs)

        embed_inputs = self.norm_layer(embed_inputs)
        outputs, _ = self.lstm1(embed_inputs, hidden)
        outputs = F.relu(self.project_layer1(outputs))

        outputs = self.norm_layer1(outputs)
        outputs, hidden = self.lstm2(outputs)
        outputs = F.relu(self.project_layer2(outputs))

        outputs = self.output_proj(outputs)
        # sfomax logits
        logits = outputs.reshape(outputs.size(0) * outputs.size(1), outputs.size(2))
        logits = self.softmax_linear(logits)

        return logits, outputs, hidden


def build_decoder(config):
    if config.dec.type == 'lstm':
        return BaseDecoder(
            project_size=config.dec.project_size,
            hidden_size=config.dec.hidden_size,
            vocab_size=config.vocab_size,
            output_size=config.dec.output_size,
            n_layers=config.dec.n_layers,
            dropout=config.dropout,
            share_weight=config.share_weight
        )
    else:
        raise NotImplementedError
