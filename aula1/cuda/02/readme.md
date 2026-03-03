
### Kernel

Kernel é uma função que executa na sua GPU, envés da CPU, dá instruções para um grande número de trabalhadores ( GPU Threads ) que fazem ao mesmo tempo o trabalho. Você marca um kernel com '__global__', e só retorna void!

Exemplo:
```
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}
```

d_A e h_A se referem a device (GPU) e host (CPU)

### Grid

O grid representa todas as threads executadas para uma unica execução de kernel. Coleção de blocos de thread. 
Quando você executa um kernel, você especifica a dimensão do grid, essencialmente definindo quantos blocos serão criados ( pode ser 1D, 2D, 3D { Cubo, Linha, ponto}). 

- Exemplo: Quando processamos uma imagem grande, cada bloco cuida de um pedaço da imagem

### Bloco
Um bloco é um grupo de threads que copera e compartilha dados. Podem ser 1D, 2D, 3D

  - Compartilham memoria
  - Sincronizam umas as outras
  - Coperam em tarefas

### Threads

Unidade, de menor execução em CUDA,cada thread executa o codigo do kernel independentemente. Tem um ID especifico. 

### Cuda Thread Indexing
Em CUDA, cada thread tem um id unico que pode ser usado para determinar a posição do grid e bloco

As seguintes variaveis são para esse proposito:

1. **`threadIdx`**:  
   - Um vetor com 3 componentes(`threadIdx.x`, `threadIdx.y`, `threadIdx.z`) da a posição da thread no bloco
   

2. **`blockDim`**:  
   - Um vetor com 3 componentes (`blockDim.x`, `blockDim.y`, `blockDim.z`) que especifica as dimensões do bloco

3. **`blockIdx`**:  
   - Um vetor com 3 componentes (`blockIdx.x`, `blockIdx.y`, `blockIdx.z`) te da a posição do bloco no grid

4. **`gridDim`**:  
   - Um vetor com 3 componentes (`gridDim.x`, `gridDim.y`, `gridDim.z`) da as dimensões do grid

## Gerenciamento de memória

- `cudaMalloc` Aloca memória na VRAM ( chamado de global memory)

```
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, N*N*sizeof(float));
    cudaMalloc(&d_b, N*N*sizeof(float));
    cudaMalloc(&d_c, N*N*sizeof(float));
```

- `cudaMemcpy` copia do dispositivo para o host, host para o dispositivo, ou disp -> dispo.
    - host to device ⇒ CPU to GPU
    - device to host ⇒ GPU to CPU
    - device to device ⇒ GPU location to different GPU location
    - **`cudaMemcpyHostToDevice`**, **`cudaMemcpyDeviceToHost`**, or **`cudaMemcpyDeviceToDevice`**
- `cudaFree` libera a memória

## Hierarquia do CUDA

1. Um **kernel** executa em várias **threads**
2. Threads são agrupadas em **Thread Blocks** (ou simplesmente **Blocks**)
3. Blocks são agrupados em uma **Grid**
4. Um **kernel** é executado como uma **Grid** de **Blocks** de **Threads**

---

### Estrutura Hierárquica

```
Grid
 ├── Block (0,0)
 │    ├── Thread (0,0,0)
 │    ├── Thread (1,0,0)
 │    └── ...
 ├── Block (1,0)
 │    ├── Thread (0,0,0)
 │    ├── Thread (1,0,0)
 │    └── ...
 └── ...
```

## Threads

- Cada **thread** possui memória local (registradores) e é privada da própria thread.
- Se quisermos somar:
  
  `a = [1, 2, 3, ... N]`  
  `b = [2, 4, 6, ... N]`

  Cada **thread** executa apenas **uma soma**:

  - Thread 0 → `a[0] + b[0]`
  - Thread 1 → `a[1] + b[1]`
  - Thread 2 → `a[2] + b[2]`
  - ...
  - Thread N → `a[N] + b[N]`

Ou seja, o paralelismo acontece distribuindo o trabalho entre várias threads, onde cada uma processa um índice diferente do vetor.

---
## Warps

- O termo **warp** vem da tecelagem:
  - Na tecelagem, o *warp* é o conjunto de fios longitudinais esticados no tear antes da inserção do *weft* (trama transversal).
  - Em CUDA, a ideia é semelhante: um grupo fixo de elementos que executam juntos.

---

### Definição em CUDA

- Um **warp** é um grupo de **32 threads**.
- Todo warp está **dentro de um Block**.
- A execução paralela real do hardware acontece no nível do **warp**, não da thread individual.
- As **instruções são emitidas para o warp**, e o warp executa a instrução simultaneamente nas 32 threads.
- Não existe como "ignorar" warps — eles fazem parte da arquitetura do hardware NVIDIA.
- O **warp scheduler** é responsável por decidir qual warp executa a cada ciclo.
- Cada **SM (Streaming Multiprocessor)** possui **4 warp schedulers**.

---

### Como o hardware enxerga

Quando você escreve:

```cpp
c[i] = a[i] + b[i];
```

Você pensa em threads.

Mas o hardware executa algo como:

```
Warp 0 → executa instrução ADD nas 32 threads
Warp 1 → executa instrução ADD nas 32 threads
Warp 2 → ...
```

Ou seja:
- 32 threads recebem **a mesma instrução**
- Cada uma opera em dados diferentes
- Isso é modelo **SIMT (Single Instruction, Multiple Threads)**

---

### Observação importante: Divergência

Se dentro de um warp você fizer:

```cpp
if (i % 2 == 0) {
    // caminho A
} else {
    // caminho B
}
```

O warp precisa executar:
1. Caminho A para as threads pares
2. Caminho B para as threads ímpares

Isso reduz eficiência → chamado de **warp divergence**.

---

### Hierarquia atualizada

```
Grid
 └── Block
      └── Warp (32 threads)
           ├── Thread 0
           ├── Thread 1
           ├── ...
           └── Thread 31
```

---