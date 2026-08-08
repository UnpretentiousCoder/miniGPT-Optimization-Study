from datasets import load_dataset

# Load WikiText-2 raw
ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")

# Join the lines into a single raw text string
full_text = "\n".join(ds["text"])

# Slice to ~1.1 MB (roughly 1,100,000 characters, same size as Shakespeare)
text = full_text[:1_100_000]

# Write to input.txt just like Karpathy's script expects
with open("input.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Saved input.txt with size: {len(text) / (1024*1024):.2f} MB")