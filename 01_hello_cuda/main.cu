#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif

// #include <cuda_runtime.h>
#include <stdio.h>


__global__ void hello_from_gpu()
{
    int x = threadIdx.x;
    printf("Hello there, CUDA from thread %d!\n", x);
}

int main()
{
    hello_from_gpu<<<1,10>>>();
    cudaDeviceReset();
    return 0;
}
