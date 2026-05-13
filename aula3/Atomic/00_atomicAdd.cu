#include <cuda_runtime.h>
#include <stdio.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

// Kernel sem atomics ( incorreto )
__global__ void incrementadorNaoAtomico(int* cont){
    // Not locked
    int old = *cont;
    int new_value = old+1;
    // Não unlucked
    *cont = new_value;
}

// com atomicos
__global__ void incrementCounterAtomic(int* cont){
    int a = atomicAdd(cont, 1);
}

int main(){
    int h_counterNonAtomic = 0;
    int h_counterAtomic = 0;
    int *d_counterNonAtomic, *d_counterAtomic;

    // aloca memoria no dispositivo ( GPU )
    cudaMalloc((void**)&d_counterNonAtomic, sizeof(int));
    cudaMalloc((void**)&d_counterAtomic, sizeof(int));

    // Copy the values for Device ( GPU)
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(int), cudaMemcpyHostToDevice);

    // Roda Kernels
    incrementadorNaoAtomico<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);

    // Copia devolta
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Non-atomic counter value: %d\n", h_counterNonAtomic);
    printf("Atomic counter value: %d\n", h_counterAtomic);

    // Libera memoria do dispositivo ( gpu )
    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    return 0;
}