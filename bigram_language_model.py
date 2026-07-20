import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import get_batches
import requests
torch.manual_seed(42)

n_embed = 32
block_size = 8

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #this maps each token to a vector of size "n_embed" #semantic meaning of the token is captured in the vector representation, which is learned during training
        self.position_embedding_table = nn.Embedding(block_size, n_embed) #T by C, maps the position of each token in the input sequence to a vector of size "n_embed" #this allows the model to capture the order of the tokens in the input sequence, which is important for language modeling
        self.lm_head = nn.Linear(n_embed, vocab_size) #num_embeddings = vocab_size, embedding_dim = n_embed, returns B,T,C (a score for every single character in the vocabulary for every position in the input sequence)
        #vector of size n_embed for each token in the input sequence, which is then used to predict the next token in the sequence

    def forward(self, idx, targets= None): 
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx) #this returns a tensor of shape (batch_size, block_size, n_embed)
        pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device)) #this returns a tensor of shape (block_size, n_embed)
        x = token_embed + pos_embed #this adds the token embeddings and position embeddings together, shape (batch_size, block_size, n_embed)
        logits = self.lm_head(x) #this returns a tensor of shape (batch_size, block_size, vocab_size). Y = X@W.T + b
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) #converts to a 2D tensor of shape (B*T, C) for the loss function
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss #B,T,C
    
    def generate(self, idx, max_new_tokens): #idx is B,T
        for _ in range(max_new_tokens):
            logits, loss = self(idx) #runs the forward
            logits = logits[:, -1, :] #keep the last time step(column) of the logits, shape (B, C)
            probs = F.softmax(logits, dim = -1)#converts the logits to probabilities, shape (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) #returns a tensor of shape (B, 1) with the index of the next token sampled from the probabilities
            idx = torch.cat((idx, idx_next), dim =1) #this concatenates the new token to the input tensor, shape (B, T+1)
        return idx
    

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
logits, loss = m(xt, yt) #runs the forward pass of the model, returns a tensor of shape (batch_size, block_size, vocab_size)
print(logits.shape) #this should print torch.Size([4, 8, 65]) which is (batch_size, block_size, vocab_size)
print(loss)
print(decoder(m.generate(torch.zeros((1,1), dtype= torch.long), max_new_tokens= 100)[0].tolist())) #this generates 100 new tokens from the model and decodes them to characters
