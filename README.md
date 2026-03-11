# Parallel Compute Sample

Benchmark de computação paralela sem GPU NVIDIA.  
Detecta automaticamente o melhor backend disponível.

## Compatibilidade

| Máquina                        | Backend usado              |
|-------------------------------|----------------------------|
| MacBook Air M4                | MPS (Metal Performance Shaders) |
| ThinkPad T14 Gen 2 (Ubuntu)   | CPU (ou ROCm se tiver GPU AMD) |
| Qualquer máquina com NVIDIA   | CUDA                       |

## Instalação

```bash
pip install torch numpy
```

> **M4**: PyTorch já inclui suporte MPS — nenhuma config extra necessária.  
> **Ubuntu**: `pip install torch` instala a versão CPU por padrão.

## Rodando

```bash
python benchmark.py
```

## O que é testado

1. **Matrix Multiplication** — equivalente a um kernel GEMM em CUDA
2. **Parallel Reduction** — soma de 50M floats (testa bandwidth de memória)
3. **Cosine Similarity** — simula busca semântica em 100k documentos (caso real de ATS)
4. **MLP Forward Pass** — inferência de rede neural pequena

## Resultados — MacBook Air M4

> Sistema: Darwin arm64 · Python 3.13.11 · PyTorch 2.10.0 · Backend: Apple Silicon MPS (Metal)

| # | Teste | Resultado | Métrica |
|---|-------|-----------|---------|
| 1 | Matrix Multiplication (4096×4096, 10 runs) | 64.02 ms/run | **2,146.7 GFLOP/s** |
| 2 | Parallel Reduction (50M floats) | 4.60 ms | **43.4 GB/s** |
| 3 | Cosine Similarity (100k docs × dim=768) | 1,035 ms | **96,586 docs/s** |
| 4 | MLP Forward Pass (batch=512, 50 runs) | 1.00 ms/batch | **511,861 samples/s** |

## Por que isso importa para CUDA?

A API PyTorch é **idêntica** para CPU, MPS e CUDA.  
O único trecho que muda ao migrar para NVIDIA é:

```python
# CPU / desenvolvimento
device = torch.device("cpu")

# M4 / desenvolvimento
device = torch.device("mps")

# Produção CUDA
device = torch.device("cuda")
```

Todo o restante do código permanece igual — o que torna esse setup
ideal para desenvolver localmente e fazer deploy em GPU na nuvem.
