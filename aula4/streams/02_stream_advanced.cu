#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel1(float *data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        data[idx] *= 2.0f;
    }
}

__global__ void kernel2(float *data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){
        data[idx] += 1.0f;
    }
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData){
    printf("Streams callback: Operação feita!\n");
}

int main(void){
    const int N = 1000000;
    size_t size = N * sizeof(float);
    float *h_data, *d_data;
    cudaStream_t stream1, stream2;
    cudaEvent_t event;
    std::cout << event << std::endl;

    // Aloca memoria no dispositivo e host
    CHECK_CUDA_ERROR(cudaMallocHost(&h_data, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));

    // inicia os dados
    for(int i = 0; i < N; i++){
        h_data[i] = static_cast<float>(i);
    }

    // PRIORIDADE NOS STREAMS
    // Muito util se precisamos de uma parte dos dados primeiro
    // para depois ser usada por outro stream

    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));

    // cria um evento
    CHECK_CUDA_ERROR(cudaEventCreate(&event));

    // copia de memoria assincrona e executa o stream1
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));
    kernel1<<<(N + 255) / 256, 256, 0, stream1>>>(d_data, N);

    // Grava o evento no stream1
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream1));

    // Faz o stream2 esperar pelo evento
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event, 0));

    // Executa kernel no stream2
    kernel2<<<(N + 255) / 256, 256, 0, stream2>>>(d_data, N);

    // calback stream2
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0));

    // copia de memoria de volta pro host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2));

    // sincroniza
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));


    for(int i = 0; i < N; i++){
        float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
        if(fabs(h_data[i] - expected) > 1e-5){
            fprintf(stderr,"Resultado falhou no elemento %d", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Passou no teste!\n");

    // limpa memoria

    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaEventDestroy(event));

    return 0;
}
