import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import get_batches
import requests
torch.manual_seed(42)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) #this maps each token to a vector of size "vocab_size"
        #num_embeddings = vocab_size, embedding_dim = vocab_size

    def forward(self, idx, targets):
        logits = self.token_embedding_table(idx) #this returns a tensor of shape (batch_size, block_size, vocab_size)
        return logits #B,T,C
    

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
unique_chars = sorted(list(set(text))) #sorting it because we want to have a consistent order of characters
print(f"Unique characters: {len(unique_chars)}")

#Tokenising each character to a number
encode = {char:i for i, char in enumerate(unique_chars)}
decode = {i: char for i, char in enumerate(unique_chars)}

encoder = lambda s: [encode[char] for char in s]
decoder = lambda l: [decode[i] for i in l]

encoded_text = encoder(text)
data = torch.tensor(encoded_text, dtype=torch.long) #stores the encoded text as a tensor of long integers
xt, yt = get_batches(split="train")
m = BigramLanguageModel(vocab_size=65)
out = m(xt, yt) #runs the forward pass of the model, returns a tensor of shape (batch_size, block_size, vocab_size)
print(out.shape) #this should print torch.Size([4, 8, 65]) which is (batch_size, block_size, vocab_size)
