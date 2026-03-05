#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <cuda_runtime.h>   

#define N 10000000 // Tamanho do vetor = 10 milhões
#define BLOCK_SIZE_1D 1024
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8

void adicao_vetor_cpu(float *a, float *b, float *c, int n){
    for(int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
    // Faz a soma em sequencia, por isso é lento
}


// CUDA kernel para adição de vetor 1D
__global__ void adicao_vetor_gpu_1d(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        c[i] = a[i] + b[i];
        // executa pra cada elemento do vetor de uma vez
    }
}

__global__ void adicao_vetor_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    
    // essa conta já fode a gpu, é mais complexa,
    // o ideal seria manter a unidimensionalidade 
    // idx = i + j*nx + k*nx*ny está transformando em 1D
    
    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;
        if (idx < nx * ny * nz) {
            c[idx] = a[idx] + b[idx];
        }
    }
}

void inicia_vetor(float *v, int n){
    for(int i = 0; i < n; i++){
        v[i] = (float)rand() / RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(){
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N * sizeof(float);

    // Aloca memória no host (CPU)
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1d = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);

    // Inicializa os vetores de entrada
    srand(time(NULL));
    inicia_vetor(h_a, N);
    inicia_vetor(h_b, N);

    // Aloca memória no device (GPU)
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    // Copia os vetores de entrada para a GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int numBlocks1D = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    int nx = 100, ny = 100, nz = 1000; // 100 x 100 x 1000 = 10 milhões
    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x, 
        (ny + block_size_3d.y - 1) / block_size_3d.y, 
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    printf("Executando warm-up...\n");
    for(int i = 0; i < 3; i++){
        adicao_vetor_cpu(h_a, h_b, h_c_cpu, N);
        adicao_vetor_gpu_1d<<<numBlocks1D, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        adicao_vetor_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking CPU\n ");
    double cpu_time = 0.0;
    for(int i = 0; i < 5; i++){
        double start_time = get_time();
        adicao_vetor_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_time / 5.0;

    printf("Benchmarking GPU 1D\n ");
    double gpu_time_1d = 0.0;
    for(int i = 0; i < 100; i++){
        cudaMemset(d_c_1d, 0, size); // Limpa o vetor de saída 

        double start_time = get_time();
        adicao_vetor_gpu_1d<<<numBlocks1D, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_time_1d += end_time - start_time;
    }
    double gpu_avg_time_1d = gpu_time_1d / 100.0;

    // verificando os resultados do 1d
    cudaMemcpy(h_c_gpu_1d, d_c_1d, size, cudaMemcpyDeviceToHost);
    bool correct_id = true;
    for(int i = 0; i < N; i++){
        if(fabs(h_c_cpu[i] - h_c_gpu_1d[i]) > 1e-4){
            correct_id = false;
           std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_1d[i] << std::endl;
            break;
        }
    }


    printf("Benchmarking GPU 3D\n ");
    double gpu_time_3d = 0.0;
    for(int i = 0; i < 100; i++){
        cudaMemset(d_c_3d, 0, size); 
        double start_time = get_time();
        adicao_vetor_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_time_3d += end_time - start_time;
    }
    double gpu_avg_time_3d = gpu_time_3d / 100.0;

    // Verifica resultado
    cudaMemcpy(h_c_gpu_3d, d_c_3d, size, cudaMemcpyDeviceToHost);
    bool correct_3d = true;
    for(int i = 0; i < N; i++){
        if(fabs(h_c_cpu[i] - h_c_gpu_3d[i]) > 1e-4){
            correct_3d = false;
            std::cout << i << " cpu: " << h_c_cpu[i] << " != " << h_c_gpu_3d[i] << std::endl;
            break;
        }
    }
    
    printf("Resultados corretos: GPU 1D = %s, GPU 3D = %s\n", correct_id ? "SIM" : "NAO", correct_3d ? "SIM" : "NAO");

    printf("CPU average time: %f seconds\n", cpu_avg_time);
    printf("GPU 1D average time: %f seconds\n", gpu_avg_time_1d);
    printf("GPU 3D average time: %f seconds\n", gpu_avg_time_3d);
    printf("Speedup GPU 1D vs CPU: %.2fx\n", cpu_avg_time / gpu_avg_time_1d);
    printf("Speedup GPU 3D vs CPU: %.2fx\n", cpu_avg_time / gpu_avg_time_3d);

    // Libera memória
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}