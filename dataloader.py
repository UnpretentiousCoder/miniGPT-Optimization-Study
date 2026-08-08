import requests
import torch
import tiktoken

def get_dataset(url):
    """
    Fetches the dataset from the given URL and returns the text content.
    """
    return requests.get(url).text


def tokenise_by_character(text):
    """
    Tokenises the input text into unique characters and provides encoding and decoding functions.
    """
    unique_chars = sorted(list(set(text)))
    encode = {char: i for i, char in enumerate(unique_chars)}
    decode = {i: char for i, char in enumerate(unique_chars)}

    decoder = lambda l: [decode[i] for i in l]
    encoder = lambda s: [encode[char] for char in s]

    encoded_text = encoder(text)
    return encoded_text, encoder, decoder

def tokenise_by_tiktoken(text, model_name="gpt2"):
    """
    Tokenises the input text using the tiktoken library and provides encoding and decoding functions.
    """
    enc = tiktoken.get_encoding(model_name)
    encoded_text = enc.encode(text)
    
    decoder = lambda l: enc.decode(l)
    encoder = lambda s: enc.encode(s)

    return encoded_text, encoder, decoder

def train_eval_split(data, train_rate=0.9):
    """
    Splits the data into training and evaluation sets based on the specified train_rate.
    """
    n = int(len(data) * train_rate)
    train = data[:n]
    ev = data[n:]
    return train, ev


def get_batches(data, batch_size, block_size, device):
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y