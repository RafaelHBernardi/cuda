#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <iostream>

#define BLOCK_SIZE 16

/// Só lembrando o nvtx é um jeito de marcar intervalos para facilitar
// analise de desempenho no painel da nvidia
// nvcc -o 00 00\ nvtx_matmul.cu -lnvToolsExt

__global__ void matrixMulKernel(float* A, float* B, float* C, int N){
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float value = 0;

    if (row < N && col < N) {
        for (int k = 0; k < N; ++k) {
            value += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = value;
    }
}

void matrixMul(float* A, float *B, float *C, int N){
    // Marca o inicio de um intervalo para profiling com NVTX
    nvtxRangePush("Matrix Multiplication");

    float *d_A, *d_B, *d_C;
    int size = N * N * sizeof(float);

    // Começa a marcar pra alocação de memória
    nvtxRangePush("Memory Allocation");
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);
    nvtxRangePop(); // Alocação de memória acabou
    
    nvtxRangePush("Copia a memória para o dispositivo (GPU)");
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    nvtxRangePop(); // Cópia de memória acabou

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    // Ainda não entendi perfeitamente essa conta de número de bloco
    // mas fds
    // é meio que pensar q é pra englobar toda a matriz
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Dnv começa a marcar a execução do kernel
    nvtxRangePush("Kernel Execution");
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B,d_C, N);
    cudaDeviceSynchronize(); // Espera o kernel terminar
    nvtxRangePop(); // Execução do kernel acabou

    nvtxRangePush("Copia a memória de volta para o host (CPU)");
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
    nvtxRangePop(); // Cópia de memória acabou

    nvtxRangePush("Libera a memória do dispositivo (GPU)");
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    nvtxRangePop(); // Liberação de memória acabou

    nvtxRangePop(); // Intervalo de multiplicação de matrizes acabou
}

int main(){
    const int N = 1024; // Tamanho da matriz (N x N)
    float *A = new float[N * N];
    float *B = new float[N * N];
    float *C = new float[N * N];
    //  Matrizes de tamanho N x N, então tem N*N elementos ( muita coisa)

    // Inicia a matriz
    matrixMul(A, B, C, N);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;

}
