#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
// esse #VAL é um parametro de macro que transforma em string literal
// #define stringify(x) #x
// STRINGIFY(cudaMalloc) -> (Retorna) ->"cudaMalloc"
// Serve pra pintar o nome da função que falhou ali, o __file__ é o arquivo __line__ é a linha q quebrou


// esse template <typename t>  ele escreve uma função que
// funciona pra qualquer tipo
template <typename T>

void check(T err, const char* const func, const char* const file, const int line){
    if(err != cudaSuccess){
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void adicaoVetor(const float *A, const float *B, float *C, int numElements){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < numElements){
        C[i] = A[i] + B[i];
    }
    // faz a adição em paralello
}

int main(void){
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    cudaStream_t stream1, stream2;

    // Aloca memoria no host ( CPU)
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Inicia os arrays no Host
    for (int i = 0; i < numElements; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    // Aloca memoria na GPU
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));

    // Cria streams
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));

    // COPIA ASSINCRONAMENTE do dispositivo a GPU
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));
    
    // p/ ter certeza que d_B está copiado antes de lançar o kernel que usa ele
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    
    adicaoVetor<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, numElements);

    // Copia os resultados devolta pra CPU assincrono
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1));

    // Sincroniza as streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // Verify result
    for (int i = 0; i < numElements; ++i) {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
            fprintf(stderr, "Falha no elelemento: %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("\nRealizou a adição de vetores corretamente!\n");

    // Limpa
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}