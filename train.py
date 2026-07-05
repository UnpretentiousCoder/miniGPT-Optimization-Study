import requests
import torch
from bigram_language_model import BigramLanguageModel
batch_size = 4
block_size = 8

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

    encoder = lambda s: [encode[char]for char in s]

    encoder_text = encoder(text)
    return encoder_text, encoder

def train_eval_split(data, train_rate=0.9):
    """
    Splits the data into training and evaluation sets based on the specified train_rate.
    """
    n = int(len(data) * train_rate)
    train = data[:n]
    ev = data[n:]
    return train, ev

def get_batches(data):
    ix = torch.randint(0, len(data)- block_size, (batch_size,))#returns a tensor of 4 random numbers between 0 and len(data)-block_size
    x = torch.stack([data[i:i +block_size] for i in ix]) #returns a tensor of shape 4 by 8  
    y = torch.stack([data[i+1: i+block_size + 1] for i in ix])
    return x, y

def main():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = get_dataset(url)
    
    print(torch.__version__)          
    print(torch.cuda.is_available())  
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    encoded_text, encoder = tokenise_by_character(text)
    data = torch.tensor(encoded_text, dtype = torch.long)
    train_data, eval_data = train_eval_split(data)
    m = BigramLanguageModel(vocab_size=len(set(text)))
    optimiser = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for step in range(100):
        
        xb, yb = get_batches(train_data)
        logits, loss = m(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        print(f"Step {step}: loss = {loss.item()}")


if __name__ == "__main__":
    main()