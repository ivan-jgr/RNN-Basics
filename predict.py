import torch
from dataloader.dataloader import  read_text
import numpy as np
from model.model import  RNN
import torch.nn.functional as F
import settings


# Creamos el modelo y cargamos el checkpoint
ckp = torch.load('./checkpoint/dict_rnn_model.pth')

alphabet = ckp['chars']

model = RNN(alphabet, settings.hidden_size, settings.n_layers)

model.load_state_dict(ckp['state_dict'])
model.to("cuda")


def predict(char):
    """
    Arguments
    ---------
    model:  modelo
    char:   caracter actual

    Returns
    -------
    next_char:  siguiente caracter
    """
    alph_size = len(model.chars)
    x = torch.tensor([model.char_to_int[char]]).view(1, 1)

    x = F.one_hot(x, num_classes=alph_size).float().cuda()

    h = model.init_hidden(1)
    h = tuple([each.data for each in h])

    out, h = model(x, h)

    p = F.softmax(out, dim=1).data.cpu()
    p, top_char = p.topk(5) # consideramos los 5 caracteres más probables
    top_char = top_char.numpy().squeeze()
    p = p.numpy().squeeze()
    
    next_char = np.random.choice(top_char, p=p/p.sum())
    next_char = model.int_to_char[next_char]

    return next_char

def sample(model, size, base='El '):
    """
    Genera texto con size caracteres

    Arguments
    ---------
    modelo: modelo
    size:   número de caracteres en el texto a generar
    base:   texto base
    """

    model.eval()

    chars = [ch for ch in base]

    for c in base:
        # predecimos el siguiente caracter
        next_char = predict(c)

    chars.append(next_char)

    for _ in range(size):
        next_char = predict(chars[-1]) # predecimos el siguiente caracter
        chars.append(next_char)
    
    return ''.join(chars)


# Generar texto con 200 caracteres
print(sample(model, 200))