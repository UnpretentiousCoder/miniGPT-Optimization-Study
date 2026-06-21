import requests

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text
print(text[:1000])

print(f"Total characters: {len(text)}")
print(f"Unique characters: {len(set(text))}")
