#include <stdio.h>

// Kernel
__global__ void whoami(void){
    
    // Grid -> Bloco -> Thread
    // 3 D -> 3D -> 3D
    // Vetor de 3 componentes
    int block_id =
    blockIdx.x *
    blockIdx.y * gridDim.x *
    blockIdx.z * gridDim.x * gridDim.y;

    int block_offset =
        block_id *
        blockDim.x * blockDim.y * blockDim.z;

    int thread_offset =
        threadIdx.x *
        threadIdx.y * blockDim.x *
        threadIdx.z * blockDim.x * blockDim.y;
    
    int id = block_offset + thread_offset;

    printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset); 
}