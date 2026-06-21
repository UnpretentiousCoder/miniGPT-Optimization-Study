import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

unique_chars = sorted(list(set(text))) #sorting it because we want to have a consistent order of characters
print(f"Unique characters: {len(unique_chars)}")

#Tokenising each character to a number
encode = {char:i for i, char in enumerate(unique_chars)}
decode = {i: char for i, char in enumerate(unique_chars)}

encoder = lambda s: [encode[char] for char in s]
decoder = lambda l: [decode[i] for i in l]

print(f" Encoded text: {encoder('hello world')}")
print(f"Decoded: {decoder(encoder('hello world'))}")
print(f"Decoded text: { ''.join(decoder(encoder('hello world')))}")