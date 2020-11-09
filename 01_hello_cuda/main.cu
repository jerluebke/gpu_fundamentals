#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif

#include <stdio.h>


// kernel definition
__global__ void hello_from_gpu()
{
    // thread coordinates: blockIdx and threadIdx
    int x = threadIdx.x;
    printf("Hello there, CUDA from thread %d!\n", x);
}

int main()
{
    // kernel invocation
    hello_from_gpu<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}
