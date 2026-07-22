import torch
import torch.nn as nn
from torch.nn import functional as F
from dataloader import get_batches
import requests
torch.manual_seed(42)

n_embed = 32
block_size = 8
head_size = 16

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #this maps each token to a vector of size "n_embed" #semantic meaning of the token is captured in the vector representation, which is learned during training
        self.position_embedding_table = nn.Embedding(block_size, n_embed) #T by C, maps the position of each token in the input sequence to a vector of size "n_embed" #this allows the model to capture the order of the tokens in the input sequence, which is important for language modeling
        self.sa_head = MultiHeadAttention(num_heads=4, n_embed=n_embed, head_size=head_size) #this is a single head of self-attention, which allows the model to capture the relationships between the tokens in the input sequence, regardless of their position in the sequence
        self.ffwd = FeedForward(n_embed) #this is a feedforward neural network that takes the output of the self-attention head and projects it back to the embedding size, which allows the model to capture more complex relationships between the tokens in the input sequence
        self.lm_head = nn.Linear(n_embed, vocab_size) #the attention block projects back to the embedding size before predicting the next token
        #vector of size n_embed for each token in the input sequence, which is then used to predict the next token in the sequence

    def forward(self, idx, targets= None): 
        B, T = idx.shape
        token_embed = self.token_embedding_table(idx) #this returns a tensor of shape (batch_size, block_size, n_embed)
        pos_embed = self.position_embedding_table(torch.arange(T, device=idx.device)) #this returns a tensor of shape (block_size, n_embed)
        x = token_embed + pos_embed #this adds the token embeddings and position embeddings together, shape (batch_size, block_size, n_embed)
        x = self.sa_head(x)
        x = self.ffwd(x)
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
            idx_cond = idx[:, -block_size:] #this ensures that the input to the model is always of shape (B, block_size), which is the maximum context size that the model can handle
            logits, loss = self(idx_cond) #runs the forward
            logits = logits[:, -1, :] #keep the last time step(column) of the logits, shape (B, C)
            probs = F.softmax(logits, dim = -1)#converts the logits to probabilities, shape (B, C)
            idx_next = torch.multinomial(probs, num_samples=1) #returns a tensor of shape (B, 1) with the index of the next token sampled from the probabilities
            idx = torch.cat((idx, idx_next), dim =1) #this concatenates the new token to the input tensor, shape (B, T+1)
        return idx
    

class Head(nn.Module):
    """one head of attention"""
    def __init__(self, n_embed, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias = False)
        self.query = nn.Linear(n_embed, head_size, bias = False)
        self.value = nn.Linear(n_embed, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) #lower triangular matrix of shape (block_size, block_size). this ensures only the previous tokens can affect the curr token

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        v = self.value(x) # (B, T, head_size)

        wei = q @ k.transpose(-2, -1)/ head_size**0.5 # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)  
        out = wei @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""
    def __init__(self, num_heads, n_embed, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embed, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embed)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out
    
class FeedForward(nn.Module):
    """linear layer followed by a non-lienar activation function"""
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)
    
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
