/**
 * \file main.cu
 * \brief HW05, matrix transposition
 * \author Jeremiah Lübke
 * \date 09.12.2020
 * \copyright MIT License
 */
#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
__device__ void __syncthreads();
#endif

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
    if ( Result != NULL ) { \
        Result->t_max = t_max; \
        Result->t_min = t_min; \
        Result->t_mean = t_mean; \
        Result->t_std = sqrt(t_var); \
    } \
}


/*****************************************************************************/


bool check_results(float* host_ref, float* gpu_ref, const size_t n,
        bool print_status = true)
{
    double eps = 1e-6;
    for ( size_t i = 0; i < n; ++i ) {
        if ( abs(host_ref[i] - gpu_ref[i]) > eps ) {
            printf("WARNING: arrays do not match! "
                   "[%zu] host: %5.2f\tgpu: %5.2f\n",
                   i, host_ref[i], gpu_ref[i]);
            return false;
        }
    }

    if ( print_status )
        printf("arrays match.\n");
    return true;
}

void init_data(float* ip, size_t size)
{
    for ( size_t i = 0; i < size; ++i )
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}


/*****************************************************************************/
/*****************************************************************************/


const int MATRIX_WIDTH = 5000;
const int MATRIX_HEIGHT = 1000;
const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;
const int REPETITIONS = 1000;


void transpose_cpu(float *odata, float *idata, int width, int height)
{
    for ( int r = 0; r < REPETITIONS; ++r ) {
        for ( int i = 0; i < height ; ++i )
            for ( int j = 0; j < width ; ++j )
                odata[j*height+i] = idata[i*width+j];
    }
}


__global__ void copy_kernel(float *odata, float *idata, int width, int height)
{
    int x_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    int idx = x_idx + width * y_idx;
    for ( int r = 0; r < REPETITIONS; ++r ) {
        for ( int i = 0; i < TILE_DIM; i += BLOCK_ROWS ) {
            if ( idx+i*width < width * height )
                odata[idx+i*width] = idata[idx+i*width];
        }
    }
}


__global__ void transpose_naive_kernel(
        float *odata, float *idata, int width, int height)
{
    int x_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    int idx_in = x_idx + width * y_idx;
    int idx_out = y_idx + height * x_idx;
    for ( int r = 0; r < REPETITIONS; ++r ) {
        for ( int i = 0; i < TILE_DIM; i += BLOCK_ROWS ) {
            if ( y_idx+i < height && x_idx < width )
                odata[idx_out+i] = idata[idx_in+i*width];
        }
    }
}


__global__ void transpose_coalesced(
        float *odata, float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    int idx_in = x_idx + width * y_idx;
    int x_idx_t = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_idx_t = blockIdx.x * TILE_DIM + threadIdx.y;
    int idx_out = x_idx_t + height * y_idx_t;
    for ( int r = 0; r < REPETITIONS; ++r ) {
        for ( int i = 0; i < TILE_DIM; i += BLOCK_ROWS )
            if ( y_idx+i < height && x_idx < width )
                tile[threadIdx.y+i][threadIdx.x] = idata[idx_in+i*width];
        __syncthreads();
        for ( int i = 0; i < TILE_DIM; i += BLOCK_ROWS )
            if ( y_idx_t+i < width && x_idx_t <  height )
                odata[idx_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
}


__global__ void transpose_coalesced_no_bank_conflict(
        float *odata, float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM+1];
    int x_idx = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_idx = blockIdx.y * TILE_DIM + threadIdx.y;
    int idx_in = x_idx + width * y_idx;
    int x_idx_t = blockIdx.y * TILE_DIM + threadIdx.x;
    int y_idx_t = blockIdx.x * TILE_DIM + threadIdx.y;
    int idx_out = x_idx_t + height * y_idx_t;
    for ( int r = 0; r < REPETITIONS; ++r ) {
       for ( int i = 0; i < TILE_DIM; i += BLOCK_ROWS )
            if ( y_idx+i < height && x_idx < width )
                tile[threadIdx.y+i][threadIdx.x] = idata[idx_in+i*width];
        __syncthreads();
        for ( int i = 0; i < TILE_DIM; i += BLOCK_ROWS )
            if ( y_idx_t+i < width && x_idx_t <  height )
                odata[idx_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
}


/*****************************************************************************/


float bandwidth_from_time(int size, float time)
{
    return (2 * size * sizeof(float) * REPETITIONS) \
           / ((float)(1<<30)) / (1e-3*time);
}


void do_benchmark(timing_result_t timing_results[5])
{
    int size = MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(float);
    float *hIn, *hOut, *refOut, *dIn, *dOut;

    hIn = (float*)malloc(size);
    hOut = (float*)malloc(size);
    refOut = (float*)malloc(size);
    cudaMalloc((void**)&dIn, size);
    cudaMalloc((void**)&dOut, size);
    init_data(hIn, MATRIX_WIDTH*MATRIX_HEIGHT);
    cudaMemcpy(dIn, hIn, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid(MATRIX_WIDTH / TILE_DIM + 1, MATRIX_HEIGHT / TILE_DIM + 1);

    /* copy_kernel<<<grid, block>>>(dOut, dIn, MATRIX_WIDTH, MATRIX_HEIGHT);
    cudaDeviceSynchronize();
    cudaMemcpy(refOut, dOut, size, cudaMemcpyDeviceToHost);
    check_results(hIn, refOut, MATRIX_WIDTH*MATRIX_HEIGHT); */

#define COMMA ,
    TIMEIT(1, (&timing_results[0]),
        transpose_cpu(hOut, hIn, MATRIX_WIDTH, MATRIX_HEIGHT);)

    TIMEIT(1, (&timing_results[1]),
        copy_kernel<<<grid COMMA block>>>(
                dOut, dIn, MATRIX_WIDTH, MATRIX_HEIGHT);
        cudaDeviceSynchronize();)
    cudaMemcpy(refOut, dOut, size, cudaMemcpyDeviceToHost);
    check_results(hIn, refOut, MATRIX_WIDTH*MATRIX_HEIGHT);

    TIMEIT(1, (&timing_results[2]),
        transpose_naive_kernel<<<grid COMMA block>>>(
                dOut, dIn, MATRIX_WIDTH, MATRIX_HEIGHT);
        cudaDeviceSynchronize();)
    cudaMemcpy(refOut, dOut, size, cudaMemcpyDeviceToHost);
    check_results(hOut, refOut, MATRIX_WIDTH*MATRIX_HEIGHT);

    TIMEIT(1, (&timing_results[3]),
        transpose_coalesced<<<grid COMMA block>>>(
                dOut, dIn, MATRIX_WIDTH, MATRIX_HEIGHT);
        cudaDeviceSynchronize();)
    cudaMemcpy(refOut, dOut, size, cudaMemcpyDeviceToHost);
    check_results(hOut, refOut, MATRIX_WIDTH*MATRIX_HEIGHT);

    TIMEIT(1, (&timing_results[4]),
        transpose_coalesced_no_bank_conflict<<<grid COMMA block>>>(
                dOut, dIn, MATRIX_WIDTH, MATRIX_HEIGHT);
        cudaDeviceSynchronize();)
    cudaMemcpy(refOut, dOut, size, cudaMemcpyDeviceToHost);
    check_results(hOut, refOut, MATRIX_WIDTH*MATRIX_HEIGHT);
#undef COMMA

    free(hIn);
    free(hOut);
    free(refOut);
    cudaFree(dIn);
    cudaFree(dOut);
}


/*****************************************************************************/
/*****************************************************************************/


const char* HEADER = "+===========+\n"
                     "| TRANSPOSE |\n"
                     "+===========+\n";
const char* ALGORITHM[5] = {
    "cpu_transpose:             %6.2f GB/s, %6.2f %%\n",
    "gpu_copy:                  %6.2f GB/s, %6.2f %%\n",
    "gpu_transpose_naive:       %6.2f GB/s, %6.2f %%\n",
    "gpu_transpose_coalesced:   %6.2f GB/s, %6.2f %%\n",
    "gpu_transpose_coal_padded: %6.2f GB/s, %6.2f %%\n"
};


int main()
{
    srand(time(NULL));
    timing_result_t timing_results[5];
    do_benchmark(timing_results);
    float peak = bandwidth_from_time(MATRIX_WIDTH*MATRIX_HEIGHT,
                                     timing_results[4].t_mean);

    puts("\n"); puts(HEADER);
    for ( int i = 0; i < 5; ++i ) {
        float bandwidth_i = bandwidth_from_time(MATRIX_WIDTH*MATRIX_HEIGHT,
                                                timing_results[i].t_mean);
        printf(ALGORITHM[i], bandwidth_i, bandwidth_i / peak * 100.0);
    }
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
