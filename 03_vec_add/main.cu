#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif

#include <stdio.h>
#include <sys/time.h>


double cpu_seconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1e-6);
}

void check_result(float* host_ref, float* gpu_ref, const int n)
{
    double eps = 1e-8;
    for ( int i = 0; i < n; ++i ) {
        if ( abs(host_ref[i] - gpu_ref[i]) > eps ) {
            printf("arrays do not match!\n"
                   "[%d] host: %5.2f\tgpu: %5.2f\n",
                   i, host_ref[i], gpu_ref[i]);
            return;
        } 
    }

    printf("arrays match.\n");
}

void init_data(float* ip, int size)
{
    srand(time(NULL));

    for ( int i = 0; i < size; ++i )
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}


// compute vector sum hC = hA + hB sequentially
void vec_add_sequential(float* hA, float* hB, float* hC, int n)
{
    for ( int i = 0; i < n; ++i )
        hC[i] = hA[i] + hB[i];
}


__global__ void vec_add_kernel(float* A, float* B, float* C, const int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( idx < N )
        C[idx] = A[idx] + B[idx];
}

// parallel version:
// allocate device memory dA, dB, dC
// copy copy dA, dB to device memory
//
// launch kernel: device performs vector addition
//
// copy dC from device memory
// free device vectors
//
// TODO: error checking
void vec_add_parallel(float* A, float* B, float* C, int n)
{
    int size = n  * sizeof(float);
    float *dA, *dB, *dC;

    cudaMalloc((void**)&dA, n*sizeof(float));
    cudaMalloc((void**)&dB, n*sizeof(float));
    cudaMalloc((void**)&dC, n*sizeof(float));

    // cudaMemcpy(dst, src, count, kind)
    // kind: cudaMemcpyHostToHost
    //       cudaMemcpyHostToDevice
    //       cudaMemcpyDeviceToHost
    //       cudaMemcpyDeviceToDevice
    //
    // NOTE: cannot copy memory between multiple GPUs!
    cudaMemcpy(dA, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, size, cudaMemcpyHostToDevice);

    double start = cpu_seconds();
    vec_add_kernel<<<n/256+1,256>>>(dA, dB, dC, n);
    cudaDeviceSynchronize();
    printf("GPU time: %f\n\n", cpu_seconds() - start);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dC);
    cudaFree(dB);
    cudaFree(dA);
}


int main()
{
    int n = 262144;
    float hA[n], hB[n], hC[n], dC[n];
    init_data(hA, n);
    init_data(hB, n);

    double start = cpu_seconds();
    vec_add_sequential(hA, hB, hC, n);
    printf("CPU time: %f\n", cpu_seconds() - start);

    vec_add_parallel(hA, hB, dC, n);

    check_result(hC, dC, n);

    return 0;
}

/* vim: set tw=79 ts=4 sw=4 et ic ai : */
