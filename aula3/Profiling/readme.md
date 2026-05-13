# Perfilamento de Kernel CUDA

# Arrasta o arquivo .nsys-rep pro NCU

##
1.
```bash
nvcc -o 00 00\ nvtx_matmul.cu -lnvToolsExt
nsys profile --stats=true ./00
```

2.
```bash
nvcc -o 01 01_naive_matmul.cu
nsys profile --stats=true ./01
```

3.
```bash
nvcc -o 02 02_tiled_matmul.cu
nsys profile --stats=true ./02
```

## Ferramentas CLI
- Ferramentas para visualizar o uso da GPU
- `nvitop`
- `nvidia-smi` ou `watch -n 0.1 nvidia-smi`


# Nsight Systems & Compute
- `nvprof` está obsoleto, então usaremos `nsys` e `ncu`
- Nsight Systems & Compute ⇒ `nsys profile --stats=true ./main`

- `compute-sanitizer ./main` para detectar vazamentos de memória
- Interface gráfica de desempenho de kernel ⇒ `ncu-ui` (pode precisar de `sudo apt install libxcb-cursor0`)

## Perfilamento de Kernel
- [Guia de Perfilamento do Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- `ncu --kernel-name matrixMulKernelOptimized --launch-skip 0 --launch-count 1 --section    Occupancy "./nvtx_matmul"`
- As ferramentas de perfilamento da NVIDIA podem não fornecer tudo que você precisa para otimizar kernels de deep learning: [Referência](https://stackoverflow.com/questions/2204527/how-do-you-profile-optimize-cuda-kernels)

## Perfilamento de Adição de Vetores
- Ao perfilar as 3 variantes a seguir com uma adição de vetores de 32 milhões de elementos (2^25):
    - Básico, sem blocos OU threads
    - Com threads
    - Com threads e blocos
- Fonte original: https://developer.nvidia.com/blog/even-easier-introduction-cuda/


## Perfilamento com NVTX
```bash
# Compilar o código
nvcc -o matmul matmul.cu -lnvToolsExt

# Rodar o programa com o Nsight Systems
nsys profile --stats=true ./matmul
```
- `nsys stats report.qdrep` para ver as estatísticas


## CUPTI
- Permite construir suas próprias ferramentas de perfilamento
- A *CUDA Profiling Tools Interface* (CUPTI) permite a criação de ferramentas de perfilamento e rastreamento para aplicações CUDA. A CUPTI fornece as seguintes APIs: *Activity API*, *Callback API*, *Event API*, *Metric API*, *Profiling API*, *PC Sampling API*, *SASS Metric API* e *Checkpoint API*. Com essas APIs, é possível desenvolver ferramentas que dão visibilidade ao comportamento de CPU e GPU em aplicações CUDA. A CUPTI é entregue como uma biblioteca dinâmica em todas as plataformas suportadas pelo CUDA.
- https://docs.nvidia.com/cupti/overview/overview.html
- Como a CUPTI tem uma curva de aprendizado maior, mantemos o uso das outras ferramentas de perfilamento neste curso.
