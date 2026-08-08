import torch
from pathlib import Path
from bigram_language_model import BigramLanguageModel
from dataloader import tokenise_by_tiktoken, train_eval_split, get_batches
from datasets import load_dataset

batch_size = 8
block_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 192
num_heads = 4
max_grad_norm = 1.0
checkpoint_path = Path("checkpoints/bigram_language_model.pt")
tokenizer_name = "gpt2"


def save_checkpoint(model, optimiser, tokenizer_name, vocab_size, n_embed, num_heads, block_size):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimiser.state_dict(),
            "tokenizer_name": tokenizer_name,
            "vocab_size": vocab_size,
            "n_embed": n_embed,
            "num_heads": num_heads,
            "block_size": block_size,
        },
        checkpoint_path,
    )


def load_checkpoint():
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer_name = checkpoint.get("tokenizer_name", "gpt2")
    model = BigramLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        n_embed=checkpoint.get("n_embed", 384),
        num_heads=checkpoint.get("num_heads", 6),
        block_size=checkpoint.get("block_size", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model, tokenizer_name, checkpoint

def main():

    ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    full_text = "\n".join(ds["text"])

    text = full_text[:1_100_000]

    encoded_text, encoder, decoder = tokenise_by_tiktoken(text, model_name=tokenizer_name)
    data = torch.tensor(encoded_text, dtype=torch.long)
    vocab_size = max(encoded_text) + 1
    train_data, _ = train_eval_split(data)
    m = BigramLanguageModel(vocab_size=vocab_size, n_embed=n_embed, num_heads=num_heads, block_size=block_size)
    m = m.to(device)
    optimiser = torch.optim.AdamW(m.parameters(), lr=3e-4)
    print(f"Training on {device} with batch_size={batch_size}, block_size={block_size}, n_embed={n_embed}")
    for step in range(5000):
        
        xb, yb = get_batches(train_data, batch_size=batch_size, block_size=block_size, device=device)
        logits, loss = m(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
        optimiser.step()
        print(f"Step {step}: loss = {loss.item()}") if step % 200 == 0 else None

    save_checkpoint(m, optimiser, tokenizer_name, vocab_size, n_embed, num_heads, block_size)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = m.generate(context, max_new_tokens=300)[0].tolist()
    print(decoder(generated_tokens))


if __name__ == "__main__":
    main()