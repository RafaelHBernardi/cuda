# Streams

# Intuição

Streams são como correntes de rio ( River Streams ) onde as operações fluem em uma só direção no tempo. Por exemplo: copiar alguns dados ( etapa 1), depois fazer alguma operação sobre eles ( etapa 2), depois copiar os dados de volta ( etapa 3 ). Essa é a ideia basica de streams.

Nos podemos ter multiplos streams de uma só vez em CUDA, cada stream tendo sua propria timeline.
Isso nos ajuda a sobrepor operações e aproveitar melhor a GPU 

Treinando uma LLM massiva, seria um desperdicio gastar muito tempo carregando todos os tokens pra dentro e fora da GPU. Os streams permite mover dados enquanto fazemos operações o tempo todo. Eles introduzem uma abstração de software chamada "prefetching", que é uma forma de mover dados antes que eles sejam necessários, ocultando a latência das transferências


# Code
- Stream padrão = stream 0 = null stream
```cpp
// Esse lançamento de kernel usa o null stream 0
myKernel<<<gridSize, blockSize>>>(args);

// Equivalente a:
myKernel<<<gridSize, blockSize, 0, 0>>>(args);
```

Revisão seção de Kernels:

 - A configuração de execução de uma função é especificada inserindo uma expressão no formato <<<gridDim, blockDim, Ns, S>>> onde:
     -> Dg(dim3): Dimensão/tamanho grid
     -> Db(dim3): Dimensão/tam de cada bloco
     -> Ns(size_t): num bytes em memoria compartilhada alocada dinamicamente por bloco na chamada
     -> S (cudaStream_t): especifica o stream associado, parametro opcional, cujo padrão é 0

  - O Stream 1 e o stream 2 são criados com prioridades diferentes significa que são executados em uma certa ordem de tempo de execução, nos dando mais controle sobre a execução concorrente dos kernels

```cpp
// Cria streams com prioridades diferentes
int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));
```

# Exemplos:

1. streams_basics.cu -> Ilustra o básico do uso de stream com transferencias de memoria assincrona e lançamento de kernels 

# Compilação

nvcc -o 01 01_stream_basics.cu
nvcc -o 02 02_stream_advanced.cu





# Extra

Prefetching é basicamente não deixar a GPU ociosa, esperando os dados, é ao mesmo tempo que ela processa o dado D1, ja ta carregando o D2, para na proxima etapa processar o D2 e carregar D3 por exempl.

Ai nos streams CUDA o Stream A - faz as tranferencias
o Stream B vai realizando as computações/operações. 

Num modelo grande os tokens/embeddings não cabem na VRAM de uma vez, sem o prefetching ficaria assim o treino:

espera os dados -> treina -> espera -> treina

O gargalo de I/O some praticamente com o prefetching