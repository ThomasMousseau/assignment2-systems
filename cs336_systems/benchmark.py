import math

import cs336_basics
from cs336_basics.model import BasicsTransformerLM
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from einops import rearrange, einsum
from cs336_basics.nn_utils import softmax
from torch import Tensor
from jaxtyping import Float, Bool, Int


if not torch.cuda.is_available():
    print("CUDA is required to run this benchmark!")


def cuda_timer():
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    return start, end

def elapsed_time_cuda_events(start_event, end_event):
    start_event.synchronize()
    end_event.synchronize()
    return start_event.elapsed_time(end_event)   

def build_model_and_batch(device: torch.device):
    vocab_size     = 10_000
    context_length = 128
    batch_size     = 8
    d_model        = 768
    num_layers     = 12
    num_heads      = 12
    d_ff           = 3072
    rope_theta     = 10_000.0

    model = BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    ).to(device)

    dummy_input_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    dummy_targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)

    return model, dummy_input_ids, dummy_targets

def benchmark_forward(model, input_ids, warmup=5, iters=10):
    model.eval() #! just in case there is dropout or something
    times = [] #!(time (ms), is_before_warmup) pairs
    with torch.no_grad():
        for i in range(warmup + iters):
            start_evt, end_evt = cuda_timer()
            start_evt.record()
            _ = model(input_ids)
            end_evt.record()
            
            times.append((elapsed_time_cuda_events(start_evt, end_evt), i>=warmup))

    avg_excl_warmup = sum(t for t, is_after_warmup in times if is_after_warmup) / iters
    assert len(times) == warmup + iters
    avg_incl_warmup = sum(t for t, _ in times) / len(times)
    print(f"[Forward]   avg over {warmup+iters} iters (incl. warmup): {avg_incl_warmup:.2f} ms")
    print(f"[Forward]   avg over {iters} iters: {avg_excl_warmup:.2f} ms")
    print(f"[Forward]   std dev over {iters} iters: {torch.std(torch.tensor([t for t, is_after_warmup in times if is_after_warmup])):.2f} ms")
    
    return avg_excl_warmup


def benchmark_backward(model, input_ids, targets, warmup=5, iters=10):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times = []

    for i in range(warmup + iters):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        start_evt, end_evt = cuda_timer()
        start_evt.record()
        loss.backward() #! gradients are computes but not applied yet                             
        end_evt.record()

        times.append((elapsed_time_cuda_events(start_evt, end_evt), i>=warmup))

    avg_excl_warmup = sum(t for t, is_after_warmup in times if is_after_warmup) / iters
    assert len(times) == warmup + iters
    avg_incl_warmup = sum(t for t, _ in times) / len(times)
    print(f"[Backward]  avg over {iters} iters: {avg_excl_warmup:.2f} ms")
    print(f"[Backward]  avg over {warmup+iters} iters (incl. warmup): {avg_incl_warmup:.2f} ms")
    print(f"[Backward]  std dev over {iters} iters: {torch.std(torch.tensor([t for t, is_after_warmup in times if is_after_warmup])):.2f} ms")
    return avg_excl_warmup


def benchmark_e2e(model, input_ids, targets, warmup=5, iters=10):
    """Forward + backward + optimizer step."""
    model.train()
    loss_fn  = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    times = []

    for i in range(warmup + iters):
        optimizer.zero_grad(set_to_none=True)
        start_evt, end_evt = cuda_timer()
        start_evt.record()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step() #! gradients are applied and weights are updated here
        end_evt.record()
        
        times.append((elapsed_time_cuda_events(start_evt, end_evt), i>=warmup))

    avg_excl_warmup = sum(t for t, is_after_warmup in times if is_after_warmup) / iters
    avg_incl_warmup = sum(t for t, _ in times) / len(times)
    print(f"[E2E]       avg over {iters} iters: {avg_excl_warmup:.2f} ms")
    print(f"[E2E]       avg over {warmup+iters} iters (incl. warmup): {avg_incl_warmup:.2f} ms")
    print(f"[E2E]       std dev over {iters} iters: {torch.std(torch.tensor([t for t, is_after_warmup in times if is_after_warmup])):.2f} ms")
    return avg_excl_warmup

def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention.

    This function implements Eq. 1 of the Transformer paper.

    Args:
        Q: Tensor of queries, may have any number of leading dimensions.
        K: Tensor of keys, sharing leading dimensions with Q.
        V: Tensor of values, sharding leading dimensions with Q and K.
        mask: An (optional) mask of shape (..., seq_len, seq_len).
            Attention scores for positions with a mask value of `False` should
            be masked out, i.e., not affect the softmaxed attention probabilities.

    Returns:
        torch.FloatTensor of shape (..., seq_len, value_dimension)
        with the output of running your scaled dot product attention
        implementation with the provided key, query, and value tensors.
    """

    d_k = K.shape[-1]
    with nvtx.range("Computing scaled dot product attention"):
        #TODO: compare eisum vs transpose + matmul for computing attention scores (Thomas)
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("Computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("Computing matmul with values (V matrix)"):
        result = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
        
    return result

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    model, input_ids, targets = build_model_and_batch(device)

    benchmark_forward(model, input_ids)
    benchmark_backward(model, input_ids, targets)
    benchmark_e2e(model, input_ids, targets)


