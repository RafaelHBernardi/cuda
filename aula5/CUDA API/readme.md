# CUDA API
> cuBLAS, cuDNN, cuBLASmp

## Opaque Struct Types ( CUDA API ):

- Você não consegue ver/acessar as implementações internas, apenas a interface externa, como nomes, argumentos etc (Assim como o Keras). O arquivo .so ( shared object ) é tratado como um binário opaco, usado apenas para executar as funções compiladas com alto desempenho

- cuFFT, cuDNN ou qualquer outra extensão cuta, elas serão disponibilizados como API. A impossibilidade de enxergar o código Assembly/C/C++ por trás é justamente o que significa o termo "opaco"

- Structs em C são apenas um mecanismo geral da linguagem que permite a NVIDIA construir seu ecossistema de forma organizada. Por exemplo cublasLtHandle_t é um tipo opaco contendo o contexto necessário para uma operação da bibilioteca cuBLASLt.

- cuBLAS (CUDA Basic Linear Algebra Subprograms)
- cuDNN (CUDA Deep Neural Network)
- cuFFT (CUDA Fast Fourier Transform)

## Error Checking (Específico da API)

- Exemplo cuBLAS 

```cpp
#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)\
```

- Exemplo cuDNN

```cpp
#define CUDNN_CHECK(call) \
    do { \
        cudnnStatus_t status = call; \
        if (status != CUDNN_STATUS_SUCCESS) { \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__, 
                    cudnnGetErrorString(status)); \
            exit(EXIT_FAILURE); \
        } 
    } while(0)\
```
- A necessidade da verificação de erros funciona assim: você configura o contexto para uma chamada da API Cuda, executa a operação e verifica o status dela passando a chamada da API para o campo call da macro

- Se a operação for concluida, o programa continuará executando normalmente, caso contrário, manda uma mensagem de erro descritiva envés de um erro qualquer.

- Existem outras macros de verificação de erros para diferentres APIs mas nesse curso só veremos as mais comuns

- https://leimao.github.io/blog/Proper-CUDA-Error-Checking/

- Resumo: por isso envés de chamar cudaMalloc(&ptr, size) nós chamamos CHECK_CUDA(cudaMalloc(&ptr, size))


## Multiplicação de Matriz

- cuDNN tem suporte implicito a multiplicaçãod e matrizes por meio de operações especificas de convolução e outras, mas esse não é o principal recurso da cuDNN

- Para multiplicação de matrizes, a melhor opção é usar as operações de algebra linear pra aprendizado do cuBLAS, é amplamente utilizado e otimizado p/ essas operações com alto desempenho

