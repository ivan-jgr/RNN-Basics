import torch.nn as nn
import torch

gpu = torch.cuda.is_available()


class RNN(nn.Module):

    def __init__(self, alphabet, hidden_size=256, n_layers=2, drop_prob=0.5):
        """
        Arguments
        ---------
        alphabet:       conjunto de caracteres
        hidden_size:    hidden features
        n_layers:       numero celdas LSTM
        drop_prob:      Dropout prob.
        """
        super(RNN, self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        # Diccionario de caracteres
        self.chars = alphabet
        self.int_to_char = dict(enumerate(self.chars))
        self.char_to_int = {char: index for index, char in self.int_to_char.items()}

        self.lstm = nn.LSTM(len(self.chars), hidden_size, n_layers, dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(hidden_size, len(self.chars))

    def forward(self, x, hidden):
        out, hidden = self.lstm(x, hidden)

        out = self.dropout(out)
        out = out.contiguous().view(-1, self.hidden_size)
        out = self.fc(out)

        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data

        if gpu:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda()
            )
        else:
            hidden = (
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_()
            )

        return hidden
