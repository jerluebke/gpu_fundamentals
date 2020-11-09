#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif


// compute vector sum hC = hA + hB sequentially
void vec_add_sequential(float* hA, float* hB, float* hC, int n)
{
    for ( int i = 0; i < n; ++i )
        hC[i] = hA[i] + hB[i];
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
__global__ void vec_add_kernel(float*, float*, float*, const int);
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

    vec_add_kernel<<<ceil(n/256.0),256>>>(dA, dB, dC, n);

    cudaMemcpy(C, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dC);
    cudaFree(dB);
    cudaFree(dA);
}

__global__ void vec_add_kernel(float* A, float* B, float* C, const int N)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if ( idx < N )
        C[idx] = A[idx] + B[idx];
}


int main()
{
    int n;

    // sequential version:
    // allocate hA, hB, hC
    float *hA, *hB, *hC;
    vec_add_sequential(hA, hB, hC, n);


    return 0;
}
