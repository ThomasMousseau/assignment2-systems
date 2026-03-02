from cs336_basics.model import BasicsTransformerLM
import torch
import torch.nn as nn

if not torch.cuda.is_available():
    print("CUDA is required to run this benchmark!")

# def sync():
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()

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

def benchmark_forward(model, input_ids, warmup=3, iters=10):
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
    print(f"[Forward]   avg over {iters} iters: {avg_excl_warmup:.2f} ms")
    print(f"[Forward]   avg over {warmup+iters} iters (incl. warmup): {avg_incl_warmup:.2f} ms")
    return avg_excl_warmup


def benchmark_backward(model, input_ids, targets, warmup=3, iters=10):
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
    return avg_excl_warmup


def benchmark_e2e(model, input_ids, targets, warmup=3, iters=10):
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
    print(f"[E2E]       avg over {iters} iters: {avg_excl_warmup:.2f} ms  (last loss: {loss.item():.4f})")
    print(f"[E2E]       avg over {warmup+iters} iters (incl. warmup): {avg_incl_warmup:.2f} ms")
    return avg_excl_warmup

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model, input_ids, targets = build_model_and_batch(device)

    benchmark_forward(model, input_ids)
    benchmark_backward(model, input_ids, targets)
    benchmark_e2e(model, input_ids, targets)


