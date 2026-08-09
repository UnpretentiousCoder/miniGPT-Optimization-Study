import csv
import torch
import subprocess
import time
from pathlib import Path
from bigram_LM_FA import BigramLanguageModel_FA
from dataloader import tokenise_by_tiktoken, train_eval_split, get_batches
from datasets import load_dataset

batch_size = 8
block_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 192
num_heads = 4
max_grad_norm = 1.0
checkpoint_path = Path("checkpoints/bigram_language_model.pt")
metrics_path = Path("logs/training_metrics_FA.csv")
tokenizer_name = "gpt2"
profile_every = 200


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
    model = BigramLanguageModel_FA(
        vocab_size=checkpoint["vocab_size"],
        n_embed=checkpoint.get("n_embed", 384),
        num_heads=checkpoint.get("num_heads", 6),
        block_size=checkpoint.get("block_size", 256),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model, tokenizer_name, checkpoint


def get_gpu_utilization():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None

    value = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    return int(value) if value.isdigit() else None


def init_metrics_file():
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "step",
                    "loss",
                    "step_time_sec",
                    "elapsed_sec",
                    "tokens_processed",
                    "tokens_per_sec",
                    "vram_used_mb",
                    "vram_peak_mb",
                    "gpu_util_percent",
                ],
            )
            writer.writeheader()


def append_metrics_row(row):
    with metrics_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        writer.writerow(row)


def print_metrics_table(row):
    headers = ["step", "loss", "step_time", "elapsed", "tok/s", "vram", "peak", "gpu"]
    values = [
        str(row["step"]),
        f'{row["loss"]:.4f}',
        f'{row["step_time_sec"]:.3f}s',
        f'{row["elapsed_sec"]:.1f}s',
        f'{row["tokens_per_sec"]:.1f}',
        "-" if row["vram_used_mb"] is None else f'{row["vram_used_mb"]:.1f}MB',
        "-" if row["vram_peak_mb"] is None else f'{row["vram_peak_mb"]:.1f}MB',
        "-" if row["gpu_util_percent"] is None else f'{row["gpu_util_percent"]}%',
    ]
    widths = [max(len(h), len(v)) for h, v in zip(headers, values)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    data = " | ".join(v.ljust(w) for v, w in zip(values, widths))
    print(line)
    print(sep)
    print(data)

def main():

    ds = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split="train")

    full_text = "\n".join(ds["text"])

    text = full_text[:1_100_000]

    encoded_text, encoder, decoder = tokenise_by_tiktoken(text, model_name=tokenizer_name)
    data = torch.tensor(encoded_text, dtype=torch.long)
    vocab_size = max(encoded_text) + 1
    train_data, _ = train_eval_split(data)
    m = BigramLanguageModel_FA(vocab_size=vocab_size, n_embed=n_embed, num_heads=num_heads, block_size=block_size)
    m = m.to(device)
    optimiser = torch.optim.AdamW(m.parameters(), lr=3e-4)
    print(f"Training on {device} with batch_size={batch_size}, block_size={block_size}, n_embed={n_embed}")
    init_metrics_file()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    training_start = time.perf_counter()
    for step in range(8000):
        if device == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()

        xb, yb = get_batches(train_data, batch_size=batch_size, block_size=block_size, device=device)
        logits, loss = m(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_grad_norm)
        optimiser.step()

        if step % profile_every == 0:
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - training_start #time
            step_elapsed = time.perf_counter() - step_start
            tokens_processed = (step + 1) * batch_size * block_size
            tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0.0
            gpu_util = get_gpu_utilization() if device == "cuda" else None
            vram_used = torch.cuda.memory_allocated() / 1024**2 if device == "cuda" else None
            vram_peak = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else None

            metrics_row = {
                "step": step,
                "loss": float(loss.item()),
                "step_time_sec": step_elapsed,
                "elapsed_sec": elapsed,
                "tokens_processed": tokens_processed,
                "tokens_per_sec": tokens_per_sec,
                "vram_used_mb": vram_used,
                "vram_peak_mb": vram_peak,
                "gpu_util_percent": gpu_util,
            }
            append_metrics_row(metrics_row)
            print_metrics_table(metrics_row)

    save_checkpoint(m, optimiser, tokenizer_name, vocab_size, n_embed, num_heads, block_size)

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = m.generate(context, max_new_tokens=300)[0].tolist()
    print(decoder(generated_tokens))


if __name__ == "__main__":
    main()