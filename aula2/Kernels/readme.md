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

- For example, if we want to vector add the two arrays `a = [1, 2, 3, 4]`, `b = [5, 6, 7, 8]` and store the result in `c`, then add 1 to each element in `c`, we need to ensure all the multiply operations catch up before moving onto adding (following PEDMAS). If we don't sync threads here, there is a possibility that we may get an incorrect output vector where a 1 is added before a multiply.

- A more clear but less common example would be when we parallelize a bit shift. If we have a bit shift operation that is dependent on the previous bit shift operation, we need to make sure that the previous bit shift operation is done before we move onto the next one.
  ![](../assets/bitshift1.png)

![](../assets/barrier.png)
