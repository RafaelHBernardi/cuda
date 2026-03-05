#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 1024
#define K 512
#define N 2048
#define BLOCK_SIZE 32

// Multiplicação de matrizes na CPU
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int l = 0; l < k; l++) {
                C[i * n + j] += A[i * k + l] * B[l * n + j];
            }
        }
    }
} 

// CUDA kernel para multiplicar matriz
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // esse threadIdx.y soma o deslocamento dentro do bloco
    // que é o produto ali
    if (row < m && col < n) {
        float value = 0;
        for (int l = 0; l < k; l++) {
            value += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = value;
    }
}

// Inicia a matriz com valores aleatórios
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}   

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu;
    float *d_a, *d_b, *d_c;
    int size_a = M * K * sizeof(float);
    int size_b = K * N * sizeof(float);
    int size_c = M * N * sizeof(float);

    // Aloca memória na CPU
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c_cpu = (float *)malloc(size_c);
    h_c_gpu = (float *)malloc(size_c);

    // inicia a matriz
    srand(time(NULL));
    init_matrix(h_a, M, K);
    init_matrix(h_b, K, N);

    // aloca memória na GPU
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    // copia os dados para a GPU
    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    // Define as dimensões dos blocos e grades
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE -1) / BLOCK_SIZE );

    // Executa warmup
    printf("Executando warmup...\n");
    for(int i = 0; i < 3; i++) {
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU
    printf("Executando benchmark na CPU...\n");
    double cpu_total = 0;
    for(int i = 0; i < 20; i++) {
        double start_cpu = get_time();
        matmul_cpu(h_a, h_b, h_c_cpu, M, K, N);
        double end_cpu = get_time();
        cpu_total += end_cpu - start_cpu;
    }
    double cpu_avg_time = cpu_total / 20.0;

    // Benchmark GPU
    printf("Executando benchmark na GPU...\n");
    double gpu_total = 0;
    for(int i = 0; i < 20; i++) {
        double start_gpu = get_time();
        matmul_gpu<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, K, N);
        cudaDeviceSynchronize();
        double end_gpu = get_time();
        gpu_total += end_gpu - start_gpu;
    }
    double gpu_avg_time = gpu_total / 20.0; 

    // Printa os resultados

    printf("Tempo médio CPU: %f segundos\n", cpu_avg_time);
    printf("Tempo médio GPU: %f segundos\n", gpu_avg_time);
    printf("Aceleração: %fx\n", cpu_avg_time / gpu_avg_time);

    // Libera memória
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}


