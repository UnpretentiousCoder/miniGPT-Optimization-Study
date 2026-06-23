import requests
import torch

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(torch.__version__)          # should be 2.0+
print(torch.cuda.is_available())  # should be True
print(torch.cuda.get_device_name(0))  # should show RTX 4060

unique_chars = sorted(list(set(text))) #sorting it because we want to have a consistent order of characters
print(f"Unique characters: {len(unique_chars)}")

#Tokenising each character to a number
encode = {char:i for i, char in enumerate(unique_chars)}
decode = {i: char for i, char in enumerate(unique_chars)}

encoder = lambda s: [encode[char] for char in s]
decoder = lambda l: [decode[i] for i in l]

encoded_text = encoder(text)
data = torch.tensor(encoded_text, dtype=torch.long) #stores the encoded text as a tensor of long integers
print(data.shape, data.dtype) #this is a 1D tensor of shape (1115394,) and dtype torch.int64
print(data[:500])