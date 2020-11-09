#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif

#include <stdio.h>


__global__ void check_idx()
{
    printf("threadIdx: (%d, %d, %d)\t"
           "blockIdx:  (%d, %d, %d)\t\t// my coordinates\n"
           "blockDim:  (%d, %d, %d)\t"
           "gridDim:   (%d, %d, %d)\t\t// total threads/blocks\n\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z);
}


int main()
{
    int elements = 6;
    dim3 block(3);
    dim3 grid((elements + block.x-1) / block.x);

    printf("FROM HOST SIDE:\n"
           "grid:  (%d, %d, %d)\t\t// number of blocks in each dimension\n"
           "block: (%d, %d, %d)\t\t// number of threads in each block\n\n",
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);

    printf("FROM DEVICE SIDE:\n");
    check_idx<<<grid, block>>>();
    // alternatively:
    // check_idx<<<2, 3>>>();

    cudaDeviceReset();

    return 0;
}
