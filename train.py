import torch
from dataloader.dataloader import data_loader
from dataloader.dataloader import read_text
import torch.nn as nn
import numpy as np

import settings

from torch.nn.functional import one_hot

from model.model import RNN

gpu = torch.cuda.is_available()


def train(model, dataset):
    """
    Arguments
    ---------
    model:      modelo a entrenar
    dataset:    torch.Tensor con el dataset
    """
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=settings.lr)
    criterion = nn.CrossEntropyLoss()

    # train y val dataset
    train_size = int(len(dataset) * settings.train_frac)
    dataset, val_data = dataset[:train_size], dataset[train_size:]

    if gpu:
        model.cuda()

    batch_count = 0
    n_chars = len(model.chars)

    for e in range(settings.epochs):
        # hidden state
        h = model.init_hidden(settings.batch_size)

        for x, targets in data_loader(dataset, settings.batch_size, settings.seq_length):
            batch_count += 1

            # Codificacón one-hot de los caracteres de entrada
            inputs = one_hot(x.long(), n_chars).float()

            if gpu:
                inputs, targets = inputs.cuda(), targets.cuda()

            # Para evitar propagar los gradientes
            h = tuple([each.data for each in h])

            model.zero_grad()

            output, h = model(inputs, h)

            # loss y backprop
            loss = criterion(output, targets.long())
            loss.backward()

            # Para evitar exploding gradient
            nn.utils.clip_grad_norm_(model.parameters(), settings.clip)
            optimizer.step()

            if batch_count % 10 == 0:
                val_h = model.init_hidden(settings.batch_size)
                val_losses = []

                model.eval()

                for x, targets in data_loader(val_data, settings.batch_size, settings.seq_length):
                    # Codificacón one-hot de los caracteres de entrada
                    inputs = one_hot(x.long(), n_chars)

                    # Para evitar propagar los gradientes
                    val_h = tuple([each.data for each in val_h])

                    if gpu:
                        inputs, targets = inputs.cuda().float(), targets.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output, targets.long())

                    val_losses.append(val_loss.item())

                model.train()

                print("Epoch: {}/{}, número de batch: {}, loss: {:.4f}, val loss: {:.4f}".format(
                    e + 1, settings.epochs, batch_count, loss.item(), np.mean(val_losses)))

    torch.save({
        'state_dict': model.state_dict(),
        'chars': model.chars},
        './checkpoint/dict_rnn_model.pth'
    )


if __name__ == '__main__':
    alphabet, encoded = read_text('./misc/guerra_y_paz.txt')

    model = RNN(alphabet, settings.hidden_size, settings.n_layers)
    print(model)
    train(model, encoded)
