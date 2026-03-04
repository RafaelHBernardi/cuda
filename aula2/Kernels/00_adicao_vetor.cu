#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 10000000 // Tamanho do vetor = 10 milhões
#define BLOCK_SIZE 256 // Número de threads por bloco

// a,b,c vetores
void adicao_vetor(float *a, float *b, float *c, int n){ 
    for(int i = 0; i < n; i++){
        c[i] = a[i] + b[i];
    }
}

__global__ void adicao_vetor_gpu(float *a, float *b, float *c, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Mas porque esse i é esse valor? 
    // blockIdx.x é o índice do bloco, blockDim.x é o número de threads por bloco, 
    // threadIdx.x é o índice da thread dentro do bloco
    // pense que na outra função o for roda n vezes, aqui
    // cada thread roda 1 if
    if(i < n){
        c[i] = a[i] + b[i];
    }
}

void init_vector(float *v, int n){
    for(int i = 0; i < n; i++){
        v[i] = (float)rand() / RAND_MAX; // valor entre 0 e 1
    }
}

// função so pra pegar o tempo de exec
double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;   
}

int main(){
    
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu; // CPU
    float *d_a, *d_b, *d_c; // GPU
    size_t size = N * sizeof(float);

    // aloca memória na CPU
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu = (float*)malloc(size);
    init_vector(h_a, N);
    init_vector(h_b, N);

    srand(time(NULL)); 
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    //  Apartir dq so copiei e colei o codigo mas basicamente:
    //  Aloca memória na cpu -> copia do host pra gpu -> executa kernel -> copia devolta pra cpu -> libera memória
    //  A adição que ocorre na CPU vai ser lenta e na GPU muito rápida, por conta do if(i < n) que garante que cada thread só vai 
    //  acessar um elemento do vetor, e a GPU tem milhares de threads rodando em paralelo, enquanto a CPU tem poucos núcleos.
    //  na cpu vai ficar rodando o for sequencialmente 

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // N = 1024, BLOCK_SIZE = 256, num_blocks = 4
    // (N + BLOCK_SIZE - 1) / BLOCK_SIZE = ( (1025 + 256 - 1) / 256 ) = 1280 / 256 = 4 rounded 

    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        adicao_vetor(h_a, h_b, h_c_cpu, N);
        adicao_vetor_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    // Benchmark CPU implementation
    printf("Benchmarking CPU implementation...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        adicao_vetor(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 20.0;

    // Benchmark GPU implementation
    printf("Benchmarking GPU implementation...\n");
    double gpu_total_time = 0.0;
    for (int i = 0; i < 20; i++) {
        double start_time = get_time();
        adicao_vetor_gpu<<<num_blocks, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_total_time += end_time - start_time;
    }
    double gpu_avg_time = gpu_total_time / 20.0;

    // Print results
    printf("CPU average time: %f milliseconds\n", cpu_avg_time*1000);
    printf("GPU average time: %f milliseconds\n", gpu_avg_time*1000);
    printf("Speedup: %fx\n", cpu_avg_time / gpu_avg_time);

    // Verify results (optional)
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    printf("Results are %s\n", correct ? "correct" : "incorrect");

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
