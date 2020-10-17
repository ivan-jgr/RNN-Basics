import torch


def read_text(text_path):
    """
    Lee un archivo de texto y genera un Tensor con los caracteres codificados

    Arguments
    ---------
    text_path: path del archivo .txt a leer

    Returns
    -------
    encoded_text:   Tensor con todos los caracteres codificados del archivo de texto
    alph_size:      Tamaño del alfabeto
    int_to_char:    dict que usaremos como lookup table para generar predicciones
    """
    with open(text_path, 'r') as f:
        text = f.read()

    # Obtenemos el alfabeto
    alphabet = tuple(set(text))
    alph_size = len(alphabet)

    # caracter -> int
    int_to_char = dict(enumerate(alphabet))
    # int-> caracter
    char_to_int = {char: val for val, char in int_to_char.items()}

    # Codificamos todo el texto
    encoded_text = torch.tensor([char_to_int[c] for c in text])

    return alphabet, encoded_text


def data_loader(dataset, batch_size, seq_length):
    # Caracteres totales en cada batch
    chars_per_batch = batch_size * seq_length
    # Número de batches
    n_batches = len(dataset) // chars_per_batch

    # Conservar batches de tamaño completo
    dataset = dataset[:n_batches * chars_per_batch]
    dataset = dataset.view(batch_size, -1)

    # Una secuencia a la vez
    for n in range(0, dataset.size(1), seq_length):
        x = dataset[:, n:n + seq_length]
        # Los "targes" los obtenemos desplazando x una posición
        y = torch.zeros_like(x)
        y[:, :-1] = x[:, 1:]

        try:
            y[:, -1] = dataset[:, n + seq_length]
        except IndexError:
            y[:, -1] = dataset[:, 0]

        yield x, y.view(batch_size * seq_length)
