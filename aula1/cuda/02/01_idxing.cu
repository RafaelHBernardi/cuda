#include <stdio.h>

// Kernel
__global__ void whoami(void){
    
    // Grid -> Bloco -> Thread
    // 3 D -> 3D -> 3D
    // Vetor de 3 componentes
    int block_id =
    blockIdx.x +
    blockIdx.y * gridDim.x +
    blockIdx.z * gridDim.x * gridDim.y;

    int block_offset =
        block_id *
        blockDim.x * blockDim.y * blockDim.z;

    int thread_offset =
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y;
    
    int id = block_offset + thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset); 
}

int main(){
    const int b_x = 2, b_y = 3, b_z = 4;
    const int t_x = 4, t_y = 4, t_z = 4; // maximo de tamanho de warp
    // vamos ter 2 warps de 32 threads por bloco
    // pois, o grid terá 2 blocos no eixo x, 3 no y e 4 no z
    // cada bloco tem 4 threads em cada dimensão, 4x4x4 = 64 threads
    // cada warp é um grupo de 32 threads q executam em paralelo
    // ent tem 2 warps  

    int blocks_per_grid = b_x * b_y * b_z;
    int threads_per_block = t_x * t_y * t_z;

    dim3 blocksPerGrid(b_x, b_y, b_z); // 3d cube
    dim3 threadsPerBlock(t_x, t_y, t_z); // 3d cube

    // é do proprio CUDA   
    /// Esses <<< >>> é o que define a configuração
    // de execução do kernel, ou seja, quantos blocos e threads por bloco

    whoami<<<blocksPerGrid, threadsPerBlock>>>();
    cudaDeviceSynchronize();
}