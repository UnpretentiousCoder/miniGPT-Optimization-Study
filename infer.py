from xml.parsers.expat import model

import torch
import tiktoken

from bigram_language_model import BigramLanguageModel
from train import checkpoint_path, device, load_checkpoint


def build_decoder(tokenizer_name="gpt2"):
    enc = tiktoken.get_encoding(tokenizer_name)
    return lambda token_ids: enc.decode(token_ids)


def decode_tokens(token_ids, decoder):
    return decoder(token_ids)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() 
                   if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Model size: {total * 4 / 1024**2:.1f} MB (float32)")
    return total, trainable



def main():
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Train the model first."
        )

    model, tokenizer_name, _ = load_checkpoint()
    decoder = build_decoder(tokenizer_name)
    model.eval()

    count_parameters(model)
    prompt = input("Enter a prompt (or leave empty for random generation): \n")
    if prompt:
        enc = tiktoken.get_encoding(tokenizer_name)
        context_ids = enc.encode(prompt)
        context = torch.tensor([context_ids], dtype=torch.long, device=device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)

    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=300)[0].tolist()

    print(decode_tokens(generated, decoder))


if __name__ == "__main__":
    main()
