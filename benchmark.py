"""
Parallel Compute Sample — sem GPU NVIDIA
Funciona em:
  - MacBook Air M4  → backend: MPS (Metal Performance Shaders)
  - ThinkPad T14 Ubuntu → backend: CPU (ou ROCm se tiver GPU AMD)

Instalar dependências:
  pip install torch numpy matplotlib

Rodar:
  python benchmark.py
"""

import time
import platform
import numpy as np
import torch

# ── Detecção de backend ──────────────────────────────────────────────────────

def detect_device() -> torch.device:
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        name = "Apple Silicon MPS (Metal)"
    else:
        dev = torch.device("cpu")
        name = "CPU"
    return dev, name


device, device_name = detect_device()

print("=" * 55)
print(f"  Sistema  : {platform.system()} {platform.machine()}")
print(f"  Python   : {platform.python_version()}")
print(f"  PyTorch  : {torch.__version__}")
print(f"  Backend  : {device_name}")
print("=" * 55)


# ── 1. Matrix Multiplication Benchmark ──────────────────────────────────────
# Análogo a um kernel CUDA de GEMM

def benchmark_matmul(size: int = 4096, runs: int = 10):
    print(f"\n[1] Matrix Multiplication  ({size}x{size}, {runs} runs)")

    A = torch.randn(size, size, device=device, dtype=torch.float32)
    B = torch.randn(size, size, device=device, dtype=torch.float32)

    # warm-up
    _ = A @ B

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        C = A @ B
        # MPS/CUDA são assíncronos — sincroniza antes de medir
        if device.type in ("cuda", "mps"):
            torch.mps.synchronize() if device.type == "mps" else torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    # FLOPs = 2 * N^3
    gflops = (2 * size ** 3) / (np.mean(times) * 1e9)
    print(f"  Tempo médio : {avg_ms:.2f} ms")
    print(f"  Performance : {gflops:.1f} GFLOP/s")


# ── 2. Parallel Reduction (soma de vetor grande) ─────────────────────────────
# Análogo a kernels de redução em CUDA

def benchmark_reduction(n: int = 50_000_000):
    print(f"\n[2] Parallel Reduction  (soma de {n:,} elementos float32)")

    x = torch.randn(n, device=device, dtype=torch.float32)

    # warm-up
    _ = x.sum()

    start = time.perf_counter()
    result = x.sum()
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    bandwidth_gb = (x.element_size() * n) / (elapsed_ms / 1000 * 1e9)
    print(f"  Resultado   : {result.item():.4f}")
    print(f"  Tempo       : {elapsed_ms:.2f} ms")
    print(f"  Bandwidth   : {bandwidth_gb:.1f} GB/s")


# ── 3. Simulação de Embedding (caso de uso HubbleHR) ─────────────────────────
# Simula a parte de similaridade de cosseno que acontece num ATS semântico

def benchmark_cosine_similarity(n_docs: int = 100_000, dim: int = 768):
    print(f"\n[3] Cosine Similarity  ({n_docs:,} docs × dim={dim})")
    print("    (simula busca semântica em corpus de currículos)")

    corpus = torch.randn(n_docs, dim, device=device, dtype=torch.float32)
    query  = torch.randn(1, dim, device=device, dtype=torch.float32)

    # normaliza (equivalente ao nomic-embed-text normalizado)
    corpus = torch.nn.functional.normalize(corpus, dim=1)
    query  = torch.nn.functional.normalize(query,  dim=1)

    start = time.perf_counter()
    scores = (corpus @ query.T).squeeze()   # (n_docs,)
    top_k  = torch.topk(scores, k=10)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"  Top-10 scores : {top_k.values.tolist()[:3]} ...")
    print(f"  Tempo total   : {elapsed_ms:.2f} ms")
    print(f"  Throughput    : {n_docs / (elapsed_ms / 1000):,.0f} docs/s")


# ── 4. Mini Neural Net forward pass ──────────────────────────────────────────

def benchmark_inference(batch: int = 512):
    print(f"\n[4] MLP Forward Pass  (batch={batch})")

    model = torch.nn.Sequential(
        torch.nn.Linear(768, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 128),
    ).to(device)
    model.eval()

    x = torch.randn(batch, 768, device=device)

    # warm-up
    with torch.no_grad():
        _ = model(x)

    runs = 50
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            out = model(x)
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / runs

    print(f"  Output shape  : {tuple(out.shape)}")
    print(f"  Latência média: {elapsed_ms:.2f} ms/batch")
    print(f"  Throughput    : {batch / (elapsed_ms / 1000):,.0f} samples/s")


# ── Execução ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    benchmark_matmul()
    benchmark_reduction()
    benchmark_cosine_similarity()
    benchmark_inference()

    print("\n" + "=" * 55)
    print("  Concluído! Mesmo código rodaria em CUDA trocando")
    print("  apenas o device — a API PyTorch é idêntica.")
    print("=" * 55)
