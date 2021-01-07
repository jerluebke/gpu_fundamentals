/**
 * \file main.cu
 * \brief HW06, matrix multiplication and transposition with texture memory
 * \author Jeremiah Lübke
 * \date 07.01.2021
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


void matmul_cpu(float *hA, float *hB, float *hC,
        const int M, const int N, const int P)
{
    for ( int i = 0; i < M; ++i )
        for ( int j = 0; j < P; ++j ) {
            float c = 0.0;
            for ( int k = 0; k < N; ++k )
                c += hA[i * N + k] * hB[k * P + j];
            hC[i * P + j] = c;
        }
}


void transpose_cpu(float *hO, float *hI,
        const int width, const int height)
{
    for ( int i = 0; i < height; ++i )
        for ( int j = 0; j < width; ++j )
            hO[j * height + i] = hI[i * width + j];
}


__global__ void matmul_kernel(float *dA, float *dB, float *dC,
        const int M, const int N, const int P, const int tile_dim)
{
    extern __shared__ float smem[];
    float *dAs = smem;
    float *dBs = &smem[tile_dim * tile_dim];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * tile_dim + tx;
    int col = bx * tile_dim + tx;

    float sum = 0.0;
    if ( ty < tile_dim && tx < tile_dim ) {
        for ( int m = 0; m < N / tile_dim + 1; ++m ) {
            if ( row < M && (m * tile_dim + tx) < N )
                dAs[ty * tile_dim + tx] = dA[row * N + (m * tile_dim + tx)];
            else
                dAs[ty * tile_dim + tx] = 0.0;

            if ( (m * tile_dim + ty) < N && col < P )
                dBs[ty * tile_dim + tx] = dB[(m * tile_dim + ty) * P + col];
            else
                dBs[ty * tile_dim + tx] = 0.0;

            __syncthreads();

            for ( int k = 0; k < tile_dim; ++k )
                sum += dAs[ty * tile_dim + k] * dBs[k * tile_dim + tx];
            __syncthreads();
        }
    }

    if ( row < M && col < P )
        dC[row * P + col] = sum;
}


__global__ void transpose_kernel(float *dO, float *dI,
        const int width, const int height,
        const int tile_dim, const int block_rows)
{
    extern __shared__ float tile[];
    int x_idx = blockIdx.x * tile_dim + threadIdx.x;
    int y_idx = blockIdx.y * tile_dim + threadIdx.y;
    int idx_in = x_idx + width * y_idx;
    int x_idx_t = blockIdx.y * tile_dim + threadIdx.x;
    int y_idx_t = blockIdx.x * tile_dim + threadIdx.y;
    int idx_out = x_idx_t + height * y_idx_t;
    for ( int i = 0; i < tile_dim; i += block_rows )
        if ( y_idx + i < height && x_idx < width )
        // if ( (threadIdx.y + i) < tile_dim && threadIdx.x < tile_dim
        //         && idx_in < width )
            tile[(threadIdx.y + i) * tile_dim + threadIdx.x] = dI[idx_in + i * width];
    __syncthreads();
    for ( int i = 0; i < tile_dim; i += block_rows )
        if ( y_idx_t + i < width && x_idx_t <  height )
            dO[idx_out + i * height] = tile[threadIdx.x * tile_dim + threadIdx.y + i];
}


/*****************************************************************************/
/*****************************************************************************/


// float bandwidth_from_time(int size, float time)
// {
//     return (2 * size * sizeof(float) * REPETITIONS) \
//            / ((float)(1<<30)) / (1e-3*time);
// }


const int MATRIX_WIDTH = 1000;
const int MATRIX_HEIGHT = 1000;


void compare_transpose(const int tile_dim)
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

    int block_rows = tile_dim / 4;
    // int block_rows = 8;
    dim3 block(tile_dim, block_rows);
    dim3 grid(MATRIX_WIDTH / tile_dim + 1, MATRIX_HEIGHT / tile_dim + 1);

    transpose_cpu(hOut, hIn, MATRIX_WIDTH, MATRIX_HEIGHT);
    transpose_kernel<<<grid, block, tile_dim*tile_dim>>>(dOut, dIn,
            MATRIX_WIDTH, MATRIX_HEIGHT, tile_dim, block_rows);
    cudaDeviceSynchronize();
    cudaMemcpy(refOut, dOut, size, cudaMemcpyDeviceToHost);
    check_results(hOut, refOut, MATRIX_WIDTH*MATRIX_HEIGHT);

    free(hIn);
    free(hOut);
    free(refOut);
    cudaFree(dIn);
    cudaFree(dOut);
}


void compare_matmul(const int tile_dim)
{
    int size = MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(float);
    float *hA, *hB, *hC, *refC, *dA, *dB, *dC;

    hA = (float*)malloc(size);
    hB = (float*)malloc(size);
    hC = (float*)malloc(size);
    refC = (float*)malloc(size);
    cudaMalloc((void**)&dA, size);
    cudaMalloc((void**)&dB, size);
    cudaMalloc((void**)&dC, size);
    init_data(hA, MATRIX_WIDTH * MATRIX_HEIGHT);
    init_data(hB, MATRIX_WIDTH * MATRIX_HEIGHT);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 block(tile_dim, tile_dim);
    dim3 grid(MATRIX_WIDTH / tile_dim + 1, MATRIX_WIDTH / tile_dim + 1);

    matmul_cpu(hA, hB, hC, MATRIX_WIDTH, MATRIX_WIDTH, MATRIX_WIDTH);
    matmul_kernel<<<block, grid, tile_dim*tile_dim>>>(dA, dB, dC,
            MATRIX_WIDTH, MATRIX_WIDTH, MATRIX_WIDTH, tile_dim);
    cudaDeviceSynchronize();
    cudaMemcpy(refC, dC, size, cudaMemcpyDeviceToHost);
    check_results(hC, refC, MATRIX_WIDTH * MATRIX_HEIGHT);

    free(hA);
    free(hB);
    free(hC);
    free(refC);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
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
    const int num_block_sizes = 3;
    int block_sizes[num_block_sizes] = { 4, 8, 16 };

    for ( int i = 0; i < num_block_sizes; ++i )
        compare_transpose(block_sizes[i]);

    // for ( int i = 0; i < num_block_sizes; ++i )
    //     compare_matmul(block_sizes[i]);


    /*
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
    */
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
