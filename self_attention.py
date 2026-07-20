import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)

B,T,C = 4, 8, 32
x = torch.randn(B, T, C)

head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) #(B,T,head_size)
q = query(x) #(B,T,16)
wei = q @ k.transpose(-2,-1) / head_size**0.5  # (B,T,16) @ (B,16,T) = (B,T,T)

tril = torch.tril(torch.ones(T, T)) #lower triangular matrix of shape (T, T). this ensures only the previous tokens can affect the curr token
wei = wei.masked_fill(tril == 0, float('-inf')) #this sets the upper triangular part of the matrix to -inf, which will be used to mask out the future tokens in the attention mechanism
wei = F.softmax(wei, dim = -1) #wei is shape (T, T), this applies the softmax function to the last dimension of the matrix, which converts the logits to probabilities. The softmax function is applied to each row of the matrix, which means that the sum of each row will be equal to 1. This is important because it allows us to interpret the values in the matrix as probabilities, which can be used to weight the input tokens when computing the output of the attention mechanism.
v = value(x) #v is the thing i want to aggregate, shape (B,T,16)
out = wei @ v #this is the same as torch.matmul(wei, x) or torch.bmm(wei, x) but for 3D tensors
print(wei[0])
print(wei.shape)
print(x.shape)
print(out.shape) #this should print torch.Size([4, 8, 32]) which is (batch_size, block_size, n_embed)