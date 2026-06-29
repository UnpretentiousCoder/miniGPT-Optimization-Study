import requests
import torch

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
batch_size = 4
block_size = 8
torch.manual_seed(42)

unique_chars = sorted(list(set(text))) #sorting it because we want to have a consistent order of characters
print(f"Unique characters: {len(unique_chars)}")

#Tokenising each character to a number
encode = {char:i for i, char in enumerate(unique_chars)}
decode = {i: char for i, char in enumerate(unique_chars)}

encoder = lambda s: [encode[char] for char in s]
decoder = lambda l: [decode[i] for i in l]

encoded_text = encoder(text)
data = torch.tensor(encoded_text, dtype=torch.long) #stores the encoded text as a tensor of long integers
print(f"Data shape: {data.shape}, Data type: {data.dtype}")

train_data, eval_data = data[:int(0.9*len(data))], data[int(0.9*len(data)):] #splitting the data into training and evaluation sets

train_dataset = train_data[:block_size+ 1]

x = train_data[:block_size]
y = train_data[1:block_size + 1]

for i in range(block_size):
    context = x[:i+1]
    target = y[i]
    print(f"Context: {context.tolist()} Target: {target}")

ix = torch.randint(0, 13, (4,))
for i in ix:
    print(i) #generates a random tensor of shape (4,) with values between 0 and 12

def get_batches(split):
    data = train_data if split =="train" else eval_data
    ix = torch.randint(0, len(data)- block_size, (batch_size,))#returns a tensor of 4 random numbers between 0 and len(data)-block_size
    x = torch.stack([data[i:i +block_size] for i in ix]) #returns a tensor of shape 4 by 8  
    y = torch.stack([data[i+1: i+block_size + 1] for i in ix])
    return x, y

xb, yb = get_batches("train") #xb and yb are 4,8 tensors
print(f"input shape: {xb.shape}, target shape: {yb.shape}") #prints the shape of the input and target batches

for b in range(batch_size):
    for i in range(block_size):
        context = xb[b, :i+1]
        target = yb[b, i]
        print("Context:", context.tolist(), "Target:", target.item())
print("Batch input:", xb)
print("Batch target:", yb)