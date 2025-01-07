import torch
import torch.nn as nn

class DNNLinearCombinedModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=70,
        dnn_hidden_units=[512, 32],
        dropout_rate=0.5
    ):
        super().__init__()
        # Linear part: EmbeddingBag to simulate a linear model on categorical features
        # Using a single-dim embedding as a linear weight per token
        self.linear_embedding = nn.EmbeddingBag(vocab_size, 1, mode='sum', sparse=True)

        # DNN part: normal embedding followed by MLP
        self.dnn_embedding = nn.Embedding(vocab_size, embedding_dim)

        layers = []
        input_dim = embedding_dim
        for units in dnn_hidden_units:
            layers.append(nn.Linear(input_dim, units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            input_dim = units
        layers.append(nn.Linear(input_dim, 1))
        self.dnn = nn.Sequential(*layers)

    def forward(self, tokens):
        # tokens: (batch_size, seq_len)
        batch_size, seq_len = tokens.size()
        offsets = torch.arange(0, batch_size * seq_len, step=seq_len, device=tokens.device)
        flat_tokens = tokens.reshape(-1)

        # Linear out
        linear_out = self.linear_embedding(flat_tokens, offsets)  # (batch_size, 1)

        # DNN out
        embedded = self.dnn_embedding(tokens)  # (batch_size, seq_len, embedding_dim)
        avg_embedded = embedded.mean(dim=1)    # (batch_size, embedding_dim)
        dnn_out = self.dnn(avg_embedded)       # (batch_size, 1)

        logits = linear_out + dnn_out
        return logits
