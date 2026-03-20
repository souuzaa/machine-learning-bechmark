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

## Resultados — ThinkPad T14 Gen 2 (Ubuntu)

> Sistema: Linux x86_64 · Python 3.12.3 · PyTorch 2.10.0+cu128 · Backend: CPU

| # | Teste | Resultado | Métrica |
|---|-------|-----------|---------|
| 1 | Matrix Multiplication (4096×4096, 10 runs) | 437.11 ms/run | **314.4 GFLOP/s** |
| 2 | Parallel Reduction (50M floats) | 4.87 ms | **41.1 GB/s** |
| 3 | Cosine Similarity (100k docs × dim=768) | 8.96 ms | **11,160,603 docs/s** |
| 4 | MLP Forward Pass (batch=512, 50 runs) | 5.82 ms/batch | **88,041 samples/s** |

## Resultados — Google Colab (Tesla T4)

> Sistema: Linux x86_64 · Python 3.12.12 · PyTorch 2.10.0+cu128 · Backend: Tesla T4

| # | Teste | Resultado | Métrica |
|---|-------|-----------|---------|
| 1 | Matrix Multiplication (4096×4096, 10 runs) | 37.07 ms/run | **3,707.4 GFLOP/s** |
| 2 | Parallel Reduction (50M floats) | 1.52 ms | **131.4 GB/s** |
| 3 | Cosine Similarity (100k docs × dim=768) | 122.23 ms | **818,104 docs/s** |
| 4 | MLP Forward Pass (batch=512, 50 runs) | 0.29 ms/batch | **1,745,816 samples/s** |

## Resultados — Server (RTX 3060)

> Sistema: Linux x86_64 · Python 3.12.3 · PyTorch 2.10.0+cu128 · Backend: NVIDIA GeForce RTX 3060

| # | Teste | Resultado | Métrica |
|---|-------|-----------|---------|
| 1 | Matrix Multiplication (4096×4096, 10 runs) | 19.38 ms/run | **7,091.8 GFLOP/s** |
| 2 | Parallel Reduction (50M floats) | 1.16 ms | **171.8 GB/s** |
| 3 | Cosine Similarity (100k docs × dim=768) | 24.59 ms | **4,066,094 docs/s** |
| 4 | MLP Forward Pass (batch=512, 50 runs) | 0.25 ms/batch | **2,083,520 samples/s** |

## Comparativo de Resultados

| Teste | MacBook Air M4 (MPS) | ThinkPad T14 (CPU) | Google Colab (T4) | Server (RTX 3060) | Melhor |
|-------|---------------------|--------------------|--------------------|--------------------------|--------|
| **Matrix Multiplication** | 2,146.7 GFLOP/s | 314.4 GFLOP/s | 3,707.4 GFLOP/s | 7,091.8 GFLOP/s | **RTX 3060 (3.3× M4)** |
| **Parallel Reduction** | 43.4 GB/s | 41.1 GB/s | 131.4 GB/s | 171.8 GB/s | **RTX 3060 (4× M4)** |
| **Cosine Similarity** | 96,586 docs/s | 11,160,603 docs/s | 818,104 docs/s | 4,066,094 docs/s | CPU (115× M4) |
| **MLP Forward Pass** | 511,861 samples/s | 88,041 samples/s | 1,745,816 samples/s | 2,083,520 samples/s | **RTX 3060 (4× M4)** |

### Observações

- **RTX 3060 domina em compute puro**: supera a T4 em todos os benchmarks de GPU — 7,091 GFLOP/s vs 3,707 GFLOP/s na multiplicação de matrizes.
- **T4 vs RTX 3060**: a RTX 3060 tem quase o dobro da performance da T4 em matmul e bandwidth, mostrando a evolução das arquiteturas NVIDIA (Ampere vs Turing).
- **CPU surpreende em cosine similarity**: o ThinkPad T14 (CPU) teve throughput 115× maior que o M4 e muito acima das GPUs nesse teste. A operação é uma única multiplicação matriz-vetor seguida de `topk` — para esse padrão, as instruções AVX/SSE do x86 são extremamente eficientes, enquanto a GPU sofre com overhead de lançamento de kernel.
- **M4 é um bom meio-termo para desenvolvimento**: performance sólida em matmul (2,146 GFLOP/s) e inferência (511k samples/s), com consumo de energia muito inferior.
- **A portabilidade do PyTorch se confirma**: o mesmo código rodou nos 4 ambientes sem nenhuma alteração.

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
