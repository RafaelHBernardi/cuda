
### Kernel

Kernel é uma função que executa na sua GPU, envés da CPU, dá instruções para um grande número de trabalhadores ( GPU Threads ) que fazem ao mesmo tempo o trabalho. Você marca um kernel com '__global__', e só retorna void!

Exemplo:
```
__global__ void addNumbers(int *a, int *b, int *result) {
    *result = *a + *b;
}
```

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

