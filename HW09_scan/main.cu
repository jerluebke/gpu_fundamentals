/**
 * \file main.cu
 * \brief HW09, Brent-Kung Scan-Algorithm
 * \author Jeremiah Lübke
 * \date 28.01.2021
 * \copyright MIT License
 */
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_timer.h>

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <time.h>



/*****************************************************************************
 * UTILITY                                                                   *
 *****************************************************************************/

/**
 * \brief time measurement record.
 * \var t_max, t_min, t_mean, t_std     timing statistics.
 */
typedef struct timing_result_s {
    float t_max, t_min, t_mean, t_std;
} timing_result_t;


/**
 * \brief perform time measurement of a piece of code.
 *
 * Runs \c Code \c Runs times and write max, min, mean and std time to the
 * \c Result struct.
 */
#define TIMEIT(Runs, Result, Code) \
{ \
    float t_max = 0.0, t_min = FLT_MAX, t_mean = 0.0, t_var = 0.0; \
    for ( unsigned timeit_idx = 0; timeit_idx < Runs; ++timeit_idx ) { \
        static StopWatchInterface* timer = NULL; \
        sdkCreateTimer(&timer); \
        sdkStartTimer(&timer); \
        \
        Code \
        \
        sdkStopTimer(&timer); \
      	float t_diff = sdkGetTimerValue(&timer); \
        float delta = t_diff - t_mean; \
        t_mean += delta / (timeit_idx + 1); \
        t_var += delta * (t_diff - t_mean); \
        t_max = t_diff > t_max ? t_diff : t_max; \
        t_min = t_diff < t_min ? t_diff : t_min; \
    } \
    \
    if ( (Result) != NULL ) { \
        (Result)->t_max = t_max; \
        (Result)->t_min = t_min; \
        (Result)->t_mean = t_mean; \
        (Result)->t_std = sqrt(t_var); \
    } \
}


bool check_results(int* host_ref, int* gpu_ref, const int n)
{
    for ( int i = 0; i < n; ++i ) {
        if ( host_ref[i] != gpu_ref[i] ) {
            printf("WARNING: arrays do not match! "
                   "[%d] host: %d\tgpu: %d\n",
                   i, host_ref[i], gpu_ref[i]);
            return false;
        }
    }
    printf("arrays match.\n");
    return true;
}


void init_data(int* ip, size_t size, int *fill_value = NULL)
{
    if ( fill_value != NULL )
        for ( size_t i = 0; i < size; ++i )
            ip[i] = (rand() & 0xFF) / 10;
    else
        for ( size_t i = 0; i < size; ++i )
            ip[i] = *fill_value;
}


/*****************************************************************************/


#define BLOCK_SIZE 64
#define SECTION_SIZE 64


void scan_cpu(int *out, int *in, int length)
{
    out[0] = in[0];
    for ( int i = 1; i < length; ++i )
        out[i] = in[i] + out[i-1];
}


__global__ void scan_gpu_block_level(int *out, int *in, int length)
{
    __shared__ int smem[SECTION_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int thid = threadIdx.x;
    if ( i < length )
        smem[thid] = in[i];

    for ( unsigned stride = 1; stride < blockDim.x; stride *= 2 ) {
        __syncthreads();
        unsigned idx = (thid + 1) * 2 * stride - 1;
        if ( idx < blockDim.x )
            smem[idx] += smem[idx - stride];
    }

    for ( unsigned stride = SECTION_SIZE / 4; stride > 0; stride /= 2 ) {
        __syncthreads();
        unsigned idx = (thid + 1) * 2 * stride - 1;
        if ( idx + stride < BLOCK_SIZE )
            smem[idx + stride] += smem[idx];
    }

    __syncthreads();

    if ( i < length )
        out[i] = smem[thid];
}


__global__ void scan_gpu_grid_level(int *out, int *block_tmp, int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int end = blockIdx.x * blockDim.x + SECTION_SIZE - 1;
    int thid = threadIdx.x;

    if ( thid == 0 ) {
        for ( unsigned block = blockIdx.x+1; block < gridDim.x; ++block ) {
            atomicAdd(&block_tmp[block], out[end]);
        }
    }

    __syncthreads();

    if ( i <  length )
        out[i] += block_tmp[blockIdx.x];
}


/*****************************************************************************/


void test_scan(int data_size)
{
    int size = data_size * sizeof(int);
    int grid_size = data_size / BLOCK_SIZE + 1;

    int *h_in, *h_out, *h_ref, *h_tmp, *d_in, *d_out, *d_tmp;
    h_in = (int*)malloc(size);
    h_out = (int*)malloc(size);
    h_ref = (int*)malloc(size);
    h_tmp = (int*)malloc(grid_size*sizeof(int));
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMalloc((void**)&d_tmp, grid_size*sizeof(int));

    init_data(h_in, data_size);
    init_data(h_tmp,  grid_size, 0);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tmp, h_tmp, grid_size*sizeof(int), cudaMemcpyHostToDevice);

    scan_cpu(h_out, h_in, data_size);

    scan_gpu_block_level<<<grid_size, BLOCK_SIZE>>>(d_out, d_in, data_size);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    scan_gpu_grid_level<<<grid_size, BLOCK_SIZE>>>(d_out, d_tmp, data_size);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    cudaMemcpy(h_ref, d_out, size, cudaMemcpyDeviceToHost);
    check_results(h_out, h_ref, data_size);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp);
    free(h_in);
    free(h_out);
    free(h_ref);
    free(h_tmp);
}


/*****************************************************************************/
/*****************************************************************************/


const char* HEADER = "+==========+\n"
                     "| SCANNING |\n"
                     "+==========+\n";
// const char* RESULT = "%s (%d): %5.3f GB/s, %5.3f GB/s, %5.3f\n";
const char* END = "+===============+\n";


int main()
{
    srand(time(NULL));
    const int max_data_size = 1000000;
    const int data_size_step = 100000;
    int data_size = 100000;

    puts(HEADER);
    for ( ; data_size <= max_data_size; data_size += data_size_step )
        test_scan(data_size);
    puts(END);
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
