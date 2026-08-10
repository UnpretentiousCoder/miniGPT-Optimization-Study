import argparse
import time

import matplotlib.pyplot as plt
import torch

from bigram_LM_FA import BigramLanguageModel_FA as BigramLanguageModelNoKV
from bigram_LM_FA_KV import BigramLanguageModel_FA as BigramLanguageModelKV


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def generate_without_kv_cache(model: BigramLanguageModelNoKV, idx: torch.Tensor, max_new_tokens: int):
    step_times_ms = []
    generated = idx.clone()
    for _ in range(max_new_tokens):
        _sync_if_cuda(generated.device)
        start = time.perf_counter()
        idx_cond = generated[:, -model.block_size :]
        logits, _ = model(idx_cond)
        idx_next = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat((generated, idx_next), dim=1)
        _sync_if_cuda(generated.device)
        step_times_ms.append((time.perf_counter() - start) * 1000.0)
    return generated, step_times_ms


@torch.inference_mode()
def generate_with_kv_cache(model: BigramLanguageModelKV, idx: torch.Tensor, max_new_tokens: int):
    step_times_ms = []
    generated = idx.clone()
    past_key_values = None
    logits_for_next = None

    for step_idx in range(max_new_tokens):
        _sync_if_cuda(generated.device)
        start = time.perf_counter()

        if step_idx == 0:
            logits, _, past_key_values = model(generated, past_key_values=None, use_cache=True)
            logits_for_next = logits[:, -1, :]
        else:
            logits, _, past_key_values = model(
                generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits_for_next = logits[:, -1, :]

        idx_next = torch.argmax(logits_for_next, dim=-1, keepdim=True)
        generated = torch.cat((generated, idx_next), dim=1)

        _sync_if_cuda(generated.device)
        step_times_ms.append((time.perf_counter() - start) * 1000.0)

    return generated, step_times_ms


def run_test(max_new_tokens: int, warmup_steps: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = 100

    model_no_kv = BigramLanguageModelNoKV(vocab_size=vocab_size).to(device)
    model_kv = BigramLanguageModelKV(vocab_size=vocab_size).to(device)

    # Keep weights aligned for a fair timing comparison.
    model_kv.load_state_dict(model_no_kv.state_dict(), strict=True)

    model_no_kv.eval()
    model_kv.eval()

    idx = torch.randint(0, vocab_size, (1, 16), device=device)
    print("device:", device)
    print("input:", idx.tolist())

    if warmup_steps > 0:
        _ = generate_without_kv_cache(model_no_kv, idx, warmup_steps)
        _ = generate_with_kv_cache(model_kv, idx, warmup_steps)

    out_no_cache, no_cache_times = generate_without_kv_cache(model_no_kv, idx, max_new_tokens)
    out_kv_cache, kv_cache_times = generate_with_kv_cache(model_kv, idx, max_new_tokens)

    no_cache_total = sum(no_cache_times)
    kv_cache_total = sum(kv_cache_times)
    speedup = no_cache_total / kv_cache_total

    print("generated (no cache):", out_no_cache.tolist())
    print("generated (kv cache):", out_kv_cache.tolist())
    print(f"total time no cache: {no_cache_total:.3f} ms")
    print(f"total time kv cache: {kv_cache_total:.3f} ms")
    print(f"speedup (no_cache / kv_cache): {speedup:.3f}x")

    steps = list(range(1, max_new_tokens + 1))
    plt.figure(figsize=(10, 5))
    plt.plot(steps, no_cache_times, marker="o", label="No KV cache")
    plt.plot(steps, kv_cache_times, marker="o", label="With KV cache")
    plt.xlabel("Generated token step")
    plt.ylabel("Step time (ms)")
    plt.title("Generation Speed Comparison: KV cache vs No KV cache")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plot_path = "kv_cache_timing_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare generation speed with and without KV cache.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Number of new tokens to generate.")
    parser.add_argument("--warmup-steps", type=int, default=4, help="Warmup generation steps.")
    args = parser.parse_args()
    run_test(max_new_tokens=args.max_new_tokens, warmup_steps=args.warmup_steps)
