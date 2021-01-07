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


const int MATRIX_WIDTH = 1000;
const int MATRIX_HEIGHT = 1000;
const int TIMING_RUNS = 100;


/*****************************************************************************/
/*****************************************************************************/


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


/*****************************************************************************/


float bandwidth_from_time(int size, float time)
{
    return (2 * size * sizeof(float)) \
           / ((float)(1<<30)) / (1e-3*time);
}


bool check_results(float* host_ref, float* gpu_ref, const size_t n,
        bool print_status = true)
{
    // double eps = 1e-6;
    double eps = 0.5;
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


__global__ void matmul_shared_kernel(float *dA, float *dB, float *dC,
        const int M, const int N, const int P, const int tile_dim)
{
    extern __shared__ float smem[];
    float *dAs = smem;
    float *dBs = &smem[tile_dim * tile_dim];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * tile_dim + ty;
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


__global__ void matmul_texture_kernel(
        float *dC, cudaTextureObject_t texA, cudaTextureObject_t texB,
        const int M, const int N, const int P, const int tile_dim)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < M && j < P ) {
        float sum = 0.0;
        for ( int k = 0; k < N; ++k )
            sum += tex2D<float>(texA, (float)k, (float)i) \
                 * tex2D<float>(texB, (float)j, (float)k);
        dC[i * P + j] = sum;
    }
}




void transpose_cpu(float *hO, float *hI,
        const int width, const int height)
{
    for ( int i = 0; i < height; ++i )
        for ( int j = 0; j < width; ++j )
            hO[j * height + i] = hI[i * width + j];
}


__global__ void transpose_shared_kernel(float *dO, float *dI,
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
            tile[(threadIdx.y + i) * tile_dim + threadIdx.x] = dI[idx_in + i * width];
    __syncthreads();
    for ( int i = 0; i < tile_dim; i += block_rows )
        if ( y_idx_t + i < width && x_idx_t <  height )
            dO[idx_out + i * height] = tile[threadIdx.x * tile_dim + threadIdx.y + i];
}


__global__ void transpose_texture_kernel(
        float *dO, cudaTextureObject_t texIn,
        const int width, const int height,
        const int tile_dim, const int block_rows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int out = y + height * x;
    for ( int i = 0; i < tile_dim; i += block_rows ) {
        if ( y + i < height && x < width )
            dO[out+i] = tex2D<float>(texIn, x, y+i);
    }
}



/*****************************************************************************/
/*****************************************************************************/


void do_transpose_shared(const int tile_dim, bool compare = true)
{
    int size = MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(float);
    int tile_size = tile_dim * tile_dim * sizeof(float);
    float *hIn, *hOut, *refOut, *dIn, *dOut;

    hIn = (float*)malloc(size);
    cudaMalloc((void**)&dIn, size);
    cudaMalloc((void**)&dOut, size);
    init_data(hIn, MATRIX_WIDTH*MATRIX_HEIGHT);
    cudaMemcpy(dIn, hIn, size, cudaMemcpyHostToDevice);

    int block_rows = tile_dim;
    dim3 block(tile_dim, block_rows);
    dim3 grid(MATRIX_WIDTH / tile_dim + 1, MATRIX_HEIGHT / tile_dim + 1);
    transpose_shared_kernel<<<grid, block, tile_size>>>(dOut, dIn,
            MATRIX_WIDTH, MATRIX_HEIGHT, tile_dim, block_rows);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if ( compare ) {
        hOut = (float*)malloc(size);
        refOut = (float*)malloc(size);
        transpose_cpu(hOut, hIn, MATRIX_WIDTH, MATRIX_HEIGHT);
        cudaMemcpy(refOut, dOut, size, cudaMemcpyDeviceToHost);
        check_results(hOut, refOut, MATRIX_WIDTH*MATRIX_HEIGHT);
        free(hOut);
        free(refOut);
    }

    cudaFree(dIn);
    cudaFree(dOut);
    free(hIn);
}


void do_matmul_shared(const int tile_dim, bool compare = true) 
{
    int size = MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(float);
    int tile_size = tile_dim * tile_dim * sizeof(float);
    float *hA, *hB, *hC, *refC, *dA, *dB, *dC;

    hA = (float*)malloc(size);
    hB = (float*)malloc(size);
    cudaMalloc((void**)&dA, size);
    cudaMalloc((void**)&dB, size);
    cudaMalloc((void**)&dC, size);
    init_data(hA, MATRIX_WIDTH * MATRIX_HEIGHT);
    init_data(hB, MATRIX_WIDTH * MATRIX_HEIGHT);
    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 block(tile_dim, tile_dim);
    dim3 grid(MATRIX_WIDTH / tile_dim + 1, MATRIX_HEIGHT / tile_dim + 1);
    matmul_shared_kernel<<<grid, block, 2*tile_size>>>(dA, dB, dC,
            MATRIX_WIDTH, MATRIX_WIDTH, MATRIX_WIDTH, tile_dim);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if ( compare ) {
        hC = (float*)malloc(size);
        refC = (float*)malloc(size);
        matmul_cpu(hA, hB, hC, MATRIX_WIDTH, MATRIX_WIDTH, MATRIX_WIDTH);
        cudaMemcpy(refC, dC, size, cudaMemcpyDeviceToHost);
        check_results(hC, refC, MATRIX_WIDTH * MATRIX_HEIGHT);
        free(hC);
        free(refC);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
}



cudaTextureObject_t create_matrix_texture(
        cudaArray *cuarr, int width, int height)
{
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuarr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeBorder;
    texDesc.addressMode[1] = cudaAddressModeBorder;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

    return texObj;
}


void do_transpose_texture(const int block_dim, bool compare = true)
{
    int size = MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(float);
    float *hIn, *hOut, *refOut, *dOut;
    cudaArray *cuarrIn;
    hIn = (float*)malloc(size);
    cudaMalloc((void**)&dOut, size);

    cudaChannelFormatDesc channelDesc = \
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&cuarrIn, &channelDesc, MATRIX_WIDTH, MATRIX_HEIGHT);

    init_data(hIn, MATRIX_WIDTH * MATRIX_HEIGHT);
    cudaMemcpyToArray(cuarrIn, 0, 0, hIn, size, cudaMemcpyHostToDevice);
    cudaTextureObject_t texIn = create_matrix_texture(
            cuarrIn, MATRIX_WIDTH, MATRIX_HEIGHT);

    dim3 block(block_dim, block_dim);
    dim3 grid(MATRIX_WIDTH / block_dim + 1, MATRIX_HEIGHT / block_dim + 1);
    transpose_texture_kernel<<<grid, block>>>(dOut, texIn, MATRIX_WIDTH,
            MATRIX_HEIGHT, block_dim, block_dim);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if ( compare ) {
        hOut = (float*)malloc(size);
        refOut = (float*)malloc(size);
        transpose_cpu(hOut, hIn, MATRIX_WIDTH, MATRIX_HEIGHT);
        cudaMemcpy(refOut, dOut, size, cudaMemcpyDeviceToHost);
        check_results(hOut, refOut, MATRIX_WIDTH * MATRIX_HEIGHT);
        free(hOut);
        free(refOut);
    }

    cudaDestroyTextureObject(texIn);
    cudaFreeArray(cuarrIn);
    cudaFree(dOut);
    free(hIn);
}


void do_matmul_texture(const int block_dim, bool compare = true)
{
    int size = MATRIX_WIDTH * MATRIX_HEIGHT * sizeof(float);
    float *hA, *hB, *hC, *refC, *dC;
    cudaArray *cuarrA, *cuarrB;
    hA = (float*)malloc(size);
    hB = (float*)malloc(size);
    cudaMalloc((void**)&dC, size);

    cudaChannelFormatDesc channelDesc = \
            cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaMallocArray(&cuarrA, &channelDesc, MATRIX_WIDTH, MATRIX_HEIGHT);
    cudaMallocArray(&cuarrB, &channelDesc, MATRIX_WIDTH, MATRIX_HEIGHT);

    init_data(hA, MATRIX_WIDTH * MATRIX_HEIGHT);
    init_data(hB, MATRIX_WIDTH * MATRIX_HEIGHT);
    cudaMemcpyToArray(cuarrA, 0, 0, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpyToArray(cuarrB, 0, 0, hB, size, cudaMemcpyHostToDevice);

    cudaTextureObject_t texA = create_matrix_texture(
            cuarrA, MATRIX_WIDTH, MATRIX_HEIGHT);
    cudaTextureObject_t texB = create_matrix_texture(
            cuarrB, MATRIX_WIDTH, MATRIX_HEIGHT);


    dim3 block(block_dim, block_dim);
    dim3 grid(MATRIX_WIDTH / block_dim + 1, MATRIX_HEIGHT / block_dim + 1);
    matmul_texture_kernel<<<grid, block>>>(dC, texA, texB,
            MATRIX_WIDTH, MATRIX_WIDTH, MATRIX_WIDTH, block_dim);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    if ( compare ) {
        hC = (float*)malloc(size);
        refC = (float*)malloc(size);
        matmul_cpu(hA, hB, hC, MATRIX_WIDTH, MATRIX_WIDTH, MATRIX_WIDTH);
        cudaMemcpy(refC, dC, size, cudaMemcpyDeviceToHost);
        check_results(hC, refC, MATRIX_WIDTH * MATRIX_HEIGHT);
        free(hC);
        free(refC);
    }

    cudaDestroyTextureObject(texA);
    cudaDestroyTextureObject(texB);
    cudaFreeArray(cuarrA);
    cudaFreeArray(cuarrB);
    cudaFree(dC);
    free(hA);
    free(hB);
}



/*****************************************************************************/
/*****************************************************************************/


const char* HEADER = "+===================+\n"
                     "| TEXTURE vs SHARED |\n"
                     "+===================+\n";
const char* RESULT = "%s (%d): %5.3f GB/s, %5.3f GB/s, %5.3f\n";
const char* END = "+===================+\n";


int main()
{
    srand(time(NULL));
    const int num_block_sizes = 3;
    int block_sizes[num_block_sizes] = { 4, 8, 16 };
    timing_result_t shared_time, texture_time;


    puts(HEADER);
    puts("Verifying correctness for block sizes 4, 8, 16...");
    for ( int i = 0; i < num_block_sizes; ++i ) {
        printf("transpose shared: ");
        do_transpose_shared(block_sizes[i]);
    }
    printf("\n");

    for ( int i = 0; i < num_block_sizes; ++i ) {
        printf("transpose texture: ");
        do_transpose_texture(block_sizes[i]);
    }
    printf("\n");

    for ( int i = 0; i < num_block_sizes; ++i ) {
        printf("matmul shared: ");
        do_matmul_shared(block_sizes[i]);
    }
    printf("\n");

    for ( int i = 0; i < num_block_sizes; ++i ) {
        printf("matmul texture: ");
        do_matmul_texture(block_sizes[i]);
    }
    printf("\n");


    int data_size = MATRIX_WIDTH * MATRIX_HEIGHT;
    float shared_bandwidth, texture_bandwidth;

    puts("(results: shared bandwidth, texture bandwidth, gain factor)");

    for ( int i = 0; i < num_block_sizes; ++i ) {
        TIMEIT(TIMING_RUNS, &shared_time,
            do_transpose_shared(block_sizes[i], false);)
        TIMEIT(TIMING_RUNS, &texture_time,
            do_transpose_texture(block_sizes[i], false);)

        shared_bandwidth = bandwidth_from_time(data_size, shared_time.t_mean);
        texture_bandwidth = bandwidth_from_time(data_size, texture_time.t_mean);
        printf(RESULT, "transpose", block_sizes[i],
                shared_bandwidth, texture_bandwidth,
                shared_bandwidth / texture_bandwidth);
    }

    for ( int i = 0; i < num_block_sizes; ++i ) {
        TIMEIT(TIMING_RUNS, &shared_time,
            do_matmul_shared(block_sizes[i], false);)
        TIMEIT(TIMING_RUNS, &texture_time,
            do_matmul_texture(block_sizes[i], false);)

        shared_bandwidth = bandwidth_from_time(data_size, shared_time.t_mean);
        texture_bandwidth = bandwidth_from_time(data_size, texture_time.t_mean);
        printf(RESULT, "matmul", block_sizes[i],
                shared_bandwidth, texture_bandwidth,
                shared_bandwidth / texture_bandwidth);
    }

    puts(END);
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
