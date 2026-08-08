import torch
from pathlib import Path
import matplotlib.pyplot as plt
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

def run_experiment(optim, vocab_size, train_data, seed=42):
    torch.manual_seed(seed)

    m = BigramLanguageModel(vocab_size=vocab_size, n_embed=n_embed, num_heads=num_heads, block_size=block_size)
    m = m.to(device)

    if optim == 'SGD':
        optimizer = torch.optim.SGD(
            m.parameters(), lr=0.01)
    elif optim == 'AdamW':
        optimizer = torch.optim.AdamW(
            m.parameters(), lr=3e-4, weight_decay=0.1)
    elif optim == 'Muon':
        muon_params = [p for name, p in m.named_parameters() 
                      if p.ndim == 2 ]
        adam_params = [p for name, p in m.named_parameters()
                      if p.ndim != 2 
                      or 'embedding' in name 
                      or 'lm_head' in name]
        optim_muon = torch.optim.Muon(muon_params, lr=0.02, momentum=0.95)
        optim_adam = torch.optim.AdamW(adam_params, lr=3e-4, weight_decay=0.1)

    print(f"Training on {device} with batch_size={batch_size}, block_size={block_size}, n_embed={n_embed}")
    if optim != "Muon":
        grad_norms = []
        losses = []
        for step in range(8000):
            xb, yb = get_batches(train_data, batch_size=batch_size, block_size=block_size, device=device)
            logits, loss = m(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
            grad_norms.append(total_norm.item())
            losses.append(loss.item())
            optimizer.step()
            print(f"Step {step}: loss = {loss.item()}") if step % 200 == 0 else None
    else:
        grad_norms = []
        losses = []
        for step in range(8000):
            xb, yb = get_batches(train_data, batch_size=batch_size, block_size=block_size, device=device)
            logits, loss = m(xb, yb)
            optim_muon.zero_grad(set_to_none=True)
            optim_adam.zero_grad(set_to_none=True)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
            grad_norms.append(total_norm.item())
            losses.append(loss.item())
            optim_muon.step()
            optim_adam.step()
            print(f"Step {step}: loss = {loss.item()}") if step % 200 == 0 else None
    return losses, grad_norms

def plot_metrics(results):
    fig, (ax_loss, ax_grad) = plt.subplots(1, 2, figsize=(14, 5))

    for optim_name, metrics in results.items():
        ax_loss.plot(metrics["losses"], label=optim_name, linewidth=1.5)
        ax_grad.plot(metrics["grad_norms"], label=optim_name, linewidth=1.5)

    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Step")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend()

    ax_grad.set_title("Gradient Norm")
    ax_grad.set_xlabel("Step")
    ax_grad.set_ylabel("Global grad norm")
    ax_grad.grid(True, alpha=0.3)
    ax_grad.legend()

    fig.tight_layout()
    output_path = Path("checkpoints/optim_benchmark_metrics.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved metrics graph to: {output_path}")
    plt.show()

def main():
    ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    
    full_text = "\n".join(ds["text"])

    text = full_text[:1_100_000]

    encoded_text, encoder, decoder = tokenise_by_tiktoken(text, model_name=tokenizer_name)
    data = torch.tensor(encoded_text, dtype=torch.long)
    vocab_size = max(encoded_text) + 1
    train_data, _ = train_eval_split(data)

    results = {}
    for opt in ['SGD', 'AdamW', 'Muon']:
        print(f"\n{'='*50}")
        print(f"Running {opt}...")
        print(f"{'='*50}")

        losses, grad_norms = run_experiment(opt, vocab_size, train_data)
        results[opt] = {"losses": losses, "grad_norms": grad_norms}

    plot_metrics(results)

if __name__ == "__main__":
    main()