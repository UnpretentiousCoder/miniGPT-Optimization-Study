import requests
import torch
from bigram_language_model import BigramLanguageModel
batch_size = 32
block_size = 8
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 32

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
    encoder = lambda s: [encode[char]for char in s]

    encoder_text = encoder(text)
    return encoder_text, encoder, decoder

def train_eval_split(data, train_rate=0.9):
    """
    Splits the data into training and evaluation sets based on the specified train_rate.
    """
    n = int(len(data) * train_rate)
    train = data[:n]
    ev = data[n:]
    return train, ev

def get_batches(data):
    ix = torch.randint(0, len(data)- block_size, (batch_size,))#returns a tensor of 32 random numbers between 0 and len(data)-block_size
    x = torch.stack([data[i:i +block_size] for i in ix]) #returns a tensor of shape 32 by 8  
    y = torch.stack([data[i+1: i+block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def main():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = get_dataset(url)

    encoded_text, encoder, decoder = tokenise_by_character(text)
    data = torch.tensor(encoded_text, dtype = torch.long)
    train_data, eval_data = train_eval_split(data)
    m = BigramLanguageModel(vocab_size=len(set(text)))
    m = m.to(device)
    optimiser = torch.optim.AdamW(m.parameters(), lr=1e-3)
    for step in range(4000):
        
        xb, yb = get_batches(train_data)
        logits, loss = m(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        print(f"Step {step}: loss = {loss.item()}") if step % 200 == 0 else None

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print("".join(decoder(m.generate(context, max_new_tokens= 300)[0].tolist()))) #this generates 300 new tokens from the model and decodes them to characters


if __name__ == "__main__":
    main()