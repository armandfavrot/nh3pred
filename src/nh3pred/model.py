import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence


class AmmoniaRNN(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        nonlinearity,
        bidirectional,
        dropout=0,
        positive_cumulative=False,
        positive_cumulative_bias=-8.0,
        cat_dims=None,
        embedding_dims=None,
    ):

        super().__init__()

        D = 2 if bidirectional else 1
        self.positive_cumulative = positive_cumulative

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(num_embeddings=cat_dim, embedding_dim=embed_dim)
                for cat_dim, embed_dim in zip(cat_dims, embedding_dims)
            ]
        )

        input_size = input_size - len(cat_dims) + sum(embedding_dims)

        self.rnn = nn.RNN(
            input_size,
            hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc1 = nn.Linear(hidden_size * D, 6)
        self.fc2 = nn.Linear(6, output_size)

        if self.positive_cumulative:
            nn.init.normal_(self.fc2.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc2.bias, positive_cumulative_bias)

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, list_x_train):

        list_x = []

        for x in list_x_train:
            x_continuous = x[0]
            x_categoricals = x[1]

            x_embeds = [
                embed(x_cat) for embed, x_cat in zip(self.embeddings, x_categoricals)
            ]

            x_final = torch.cat([x_continuous] + x_embeds, dim=-1)

            list_x.append(x_final)

        packed_x = pack_sequence(list_x, enforce_sorted=False)

        packed_h, _ = self.rnn(packed_x)

        h, lengths = pad_packed_sequence(packed_h, batch_first=True)

        out = self.fc1(h)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        if self.positive_cumulative:
            increments = torch.nn.functional.softplus(out)
            out = torch.cumsum(increments, dim=1)

        return out, lengths
