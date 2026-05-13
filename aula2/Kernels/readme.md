# Kernels

## Parametros na execução de Kernel

- Os tipos de variáveis dim3 são tipos 3d para grids e blocos de thread que depois você manda pra configurar a execução
do kernel.

- Dim3 aceita eleementos como vetores, matrizes ou "volumes" (tensores) # Esse volumes foi a tradução que eu fiz de ingles -> pt não sei até que ponto é volume mesmo que fala kkkk

```cpp
dim3 gridDim(4, 4, 1); // 4 blocks in x, 4 block in y, 1 block in z
dim3 blockDim(4, 2, 2); // 4 threads in x, 2 thread in y, 2 thread in z
```

- other type is `int` which specifies a 1D vector

```cpp
int gridDim = 16; // 16 blocks
int blockDim = 32; // 32 threads per block
<<<gridDim, blockDim>>>
// these aren't dim3 types but they are still valid if the indexing scheme is 1D
```

- 
- gridDim ⇒ gridDim.x * gridDim.y * gridDim.z = # De blocos sendo executados

- blockDim ⇒ blockDim.x * blockDim.y * blockDim.z = # Threads/Bloco

- total threads = (threads per block) \* # De blocos

- A configuração de execução de uma função global é feita por `<<<gridDim, blockDim, Ns, S>>>`, onde:
 - gridDim -> dimensão/tamanho de cada grid
 - blockDim -> dimensão/tamanho de cada bloco
 Ns (size_t) -> número de bytes na memória compartilhada que é dinamicamente alocada por 
 chamada de bloco, além da alocação de memória estática ( parametro geralmente omitido )
 - S ( cudaStream_t ) especifica o stream associado, parametro opcional que por padrão é 0

> Ref -> https://stackoverflow.com/questions/26770123/understanding-this-cuda-kernels-launch-parameters

## Sincronização de Threads

- `cudaDeviceSyncronize();` -> Pense nisso como uma barreira, ela dá a certeza que não teve nenhum no problema nos kernels

- `__syncthreads();` poem uma barreira **dentro** do kernel. Util para quando você esta mexendo nos mesmos locais de memória,
e precisa que todas as tarefas terminem primeiro antes de começar a fazer alterações em um determinado ponto.
Exemplo: Um (worker) processo/thread pode estar no meio do trabalho modificando um local da memória, outro (worker) pode já ter
terminado a tarefa que o primeiro está executando. Isso pode dar instabilidade por isso o syncthreads corrige 

- `__syncwarps();` sincroniza todas as threads com o warp

- Sincronizamos as threads pq as threads são assincronas e podem ser executadas em qualquer ordem. Porém alguns precisam ser em ordens pq dependem do output dos outros 
Exemplo: somar 2 vetores A + B e armazenar em C depois somar 1 em C, cada elemento vai ser somado paralelamente, pode haver algum
erro e algum dos resultados de C ser só 1 enves de c[i] = a[i] + b[i] + 1




## Thread Safety

- [Is CUDA thread-safe?](https://forums.developer.nvidia.com/t/is-cuda-thread-safe/2262/2)
- quando um pedaço do código é seguro para 1 thread ele pode ser executado por multiplos threads
ao mesmo tempo sem ter que lidar com race conditions ou outros problemas

- Race Conditions são prevenidas pelo comando `cudaDeviceSynchronize()`

- if you are wondering about calling multiple GPU kernels with different CPU threads,
  refer to the link above.

## SIMD/SIMT (Single Instruction, Multiple Threads)

- Não é tão importante por agora, será retomado depois

- [Can CUDA use SIMD instructions?](https://stackoverflow.com/questions/5238743/can-cuda-use-simd-extensions)

> [Warp Level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)

- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy 

## Math intrinsics
- Instruções no dispositivo fundamentais para operações matemáticas
- https://docs.nvidia.com/cuda/cuda-math-api/index.html
- you can use host designed operations like `log()` (host) {CPU} instead of `logf()` {GPU} (device) but they will run slower. these math essentials allow very math math operations on the device/GPU. you can pass in `-use_fast_math` to the nvcc compiler to convert to these device only ops at the cost of barely noticeable precision error.
- `--fmad=true` for fused multiply-add