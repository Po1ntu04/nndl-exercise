import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        weight_shape = list(module.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        module.weight.data.uniform_(-w_bound, w_bound)
        module.bias.data.fill_(0)
        print("inital linear weight")


class word_embedding(nn.Module):
    def __init__(self, vocab_length, embedding_dim):
        super(word_embedding, self).__init__()
        embedding_init = np.random.uniform(-1, 1, size=(vocab_length, embedding_dim)).astype(np.float32)
        self.word_embedding = nn.Embedding(vocab_length, embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(embedding_init))

    def forward(self, input_sentence):
        """
        :param input_sentence: tensor containing word indices.
        :return: tensor containing word embeddings.
        """
        return self.word_embedding(input_sentence)


class RNN_model(nn.Module):
    def __init__(
        self,
        batch_sz,
        vocab_len,
        word_embedding,
        embedding_dim,
        lstm_hidden_dim,
        model_type="lstm",
    ):
        super(RNN_model, self).__init__()

        self.word_embedding_lookup = word_embedding
        self.batch_size = batch_sz
        self.vocab_length = vocab_len
        self.word_embedding_dim = embedding_dim
        self.hidden_dim = lstm_hidden_dim
        self.model_type = model_type.lower()
        self.num_layers = 2

        if self.model_type == "lstm":
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
            )
        elif self.model_type == "rnn":
            self.recurrent = nn.RNN(
                input_size=embedding_dim,
                hidden_size=lstm_hidden_dim,
                num_layers=self.num_layers,
                nonlinearity="tanh",
                batch_first=True,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.fc = nn.Linear(lstm_hidden_dim, vocab_len)
        self.apply(weights_init)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sentence, is_test=False):
        batch_input = self.word_embedding_lookup(sentence)
        if batch_input.dim() == 2:
            batch_input = batch_input.unsqueeze(0)
        batch_size = batch_input.size(0)

        if self.model_type == "lstm":
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=batch_input.device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=batch_input.device)
            output, _ = self.recurrent(batch_input, (h0, c0))
        else:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=batch_input.device)
            output, _ = self.recurrent(batch_input, h0)

        out = output.contiguous().view(-1, self.hidden_dim)
        out = F.relu(self.fc(out))
        out = self.softmax(out)

        if is_test:
            return out[-1, :].view(1, -1)
        return out
