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


bool check_results(int* host_ref, int* gpu_ref, const int n,
        bool print_status = true)
{
    for ( int i = 0; i < n; ++i ) {
        if ( host_ref[i] != gpu_ref[i] ) {
            printf("WARNING: arrays do not match! "
                   "[%d] host: %d\tgpu: %d\n",
                   i, host_ref[i], gpu_ref[i]);
            return false;
        }
    }
    if ( print_status )
        printf("arrays match.\n");
    return true;
}


void init_data(int* ip, size_t size)
{
    for ( size_t i = 0; i < size; ++i )
        ip[i] = (int)(rand() & 0xFF) / 10;
}


/*****************************************************************************/


#define BLOCK_SIZE 1024
#define CPU_TIMING_RUNS 100
#define GPU_TIMING_RUNS 1000


void scan_cpu(int *out, int *in, int length)
{
    out[0] = in[0];
    for ( int i = 1; i < length; ++i )
        out[i] = in[i] + out[i-1];
}


__global__ void scan_gpu_block_level(int *out, int *in, int *block_tmp,
                                     int length)
{
    __shared__ int smem[BLOCK_SIZE];

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

    for ( unsigned stride = BLOCK_SIZE / 4; stride > 0; stride /= 2 ) {
        __syncthreads();
        unsigned idx = (thid + 1) * 2 * stride - 1;
        if ( idx + stride < BLOCK_SIZE )
            smem[idx + stride] += smem[idx];
    }

    __syncthreads();

    if ( i < length )
        out[i] = smem[thid];

    if ( thid == 0 )
        block_tmp[blockIdx.x] = smem[BLOCK_SIZE-1];
}


__global__ void scan_gpu_grid_level(int *out, int *block_tmp, int length)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < length && blockIdx.x > 0 )
        out[i] += block_tmp[blockIdx.x-1];
}


/*****************************************************************************/


void test_scan(int data_size, timing_result_t *cpu_time,
               timing_result_t *gpu_time)
{
    int size = data_size * sizeof(int);
    int grid_size = ceil(data_size / (float)BLOCK_SIZE);

    int *h_in, *h_out, *h_ref, *d_in, *d_out, *d_tmp_1, *d_tmp_2;
    h_in = (int*)malloc(size);
    h_out = (int*)malloc(size);
    h_ref = (int*)malloc(size);
    cudaMalloc((void**)&d_in, size);
    cudaMalloc((void**)&d_out, size);
    cudaMalloc((void**)&d_tmp_1, (size+1) / BLOCK_SIZE);
    cudaMalloc((void**)&d_tmp_2, 4);

    init_data(h_in, data_size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    TIMEIT(CPU_TIMING_RUNS, cpu_time, scan_cpu(h_out, h_in, data_size);)

    #define COMMA ,
    #define CUDA_ERROR_CHECKS \
        checkCudaErrors(cudaGetLastError()); \
        checkCudaErrors(cudaDeviceSynchronize());
    TIMEIT(GPU_TIMING_RUNS, gpu_time,
        scan_gpu_block_level<<<grid_size COMMA BLOCK_SIZE>>>(
                d_out, d_in, d_tmp_1, data_size);
        CUDA_ERROR_CHECKS
        scan_gpu_block_level<<<1 COMMA BLOCK_SIZE>>>(
                d_tmp_1, d_tmp_1, d_tmp_2, data_size / BLOCK_SIZE);
        CUDA_ERROR_CHECKS
        scan_gpu_grid_level<<<grid_size COMMA BLOCK_SIZE>>>(
                d_out, d_tmp_1, data_size);
        CUDA_ERROR_CHECKS
    )
    #undef COMMA
    #undef CUDA_ERROR_CHECKS

    cudaMemcpy(h_ref, d_out, size, cudaMemcpyDeviceToHost);
    check_results(h_out, h_ref, data_size, false);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_tmp_1);
    cudaFree(d_tmp_2);
    free(h_in);
    free(h_out);
    free(h_ref);
}


/*****************************************************************************/
/*****************************************************************************/


const char* HEADER = "+==========+\n"
                     "| SCANNING |\n"
                     "+==========+\n";
const char* RESULT_HEADER = "+---------+-------+-------+---------+\n" \
                            "| length  | cpu   | gpu   | speedup |\n" \
                            "+---------+-------+-------+---------+";
const char* RESULT = "| %7d | %5.3f | %5.3f | %5.3f   |\n";
const char* END = "+---------+-------+-------+---------+";


int main()
{
    srand(time(NULL));
    const int max_data_size = 1000000;
    const int data_size_step = 100000;
    int data_size = 100000;
    timing_result_t cpu_time, gpu_time;

    puts(HEADER);
    puts(RESULT_HEADER);
    for ( ; data_size <= max_data_size; data_size += data_size_step ) {
        test_scan(data_size, &cpu_time, &gpu_time);
        printf(RESULT, data_size, cpu_time.t_mean, gpu_time.t_mean,
               cpu_time.t_mean / gpu_time.t_mean);
    }
    puts(END);
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
