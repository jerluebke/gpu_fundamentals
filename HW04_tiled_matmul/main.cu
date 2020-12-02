/**
 * \file main.cu
 * \brief HW04, tiled matrix multiplication
 * \author Jeremiah Lübke
 * \date 01.12.2020
 * \copyright MIT License
 */
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_timer.h>

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <time.h>


/*****************************************************************************/

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

void check_results(float* host_ref, float* gpu_ref, const size_t n)
{
    double eps = 1e-1;
    for ( size_t i = 0; i < n; ++i ) {
        if ( abs(host_ref[i] - gpu_ref[i]) > eps ) {
            printf("arrays do not match!\n"
                   "[%zu] host: %5.2f\tgpu: %5.2f\n",
                   i, host_ref[i], gpu_ref[i]);
            return;
        }
    }

    printf("arrays match.\n");
}

void init_data(float* ip, size_t size)
{
    for ( size_t i = 0; i < size; ++i )
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}


/*****************************************************************************/


const size_t TILE_WIDTH = 16;


/**
 * \brief matrix multiplication on CPU, for comparison
 * \param[in] Mh, Nh matrices to multiply.
 * \param[out] Ph resulting matrix.
 * \param[in] width size of square matrices.
 */
void matmul_cpu(float* Mh, float* Nh, float* Ph, size_t width)
{
    for ( size_t i = 0; i < width; ++i ) {
        for ( size_t j = 0; j < width; ++j ) {
            float p_value = 0.0;
            for ( size_t k = 0; k < width; ++k ) {
                p_value += Mh[i*width+k] * Nh[k*width+j];
            }
            Ph[i*width+j] = p_value;
        }
    }
}


/**
 * \brief matrix multiplication, on GPU, reading data from global device
 * memory.
 * \param[in] Md, Nd matrices to multiply.
 * \param[out] Pd resulting matrix.
 * \param[in] width size of square matrices.
 */
__global__ void matmul_global_kernel(float* Md, float* Nd, float* Pd, size_t width)
{
    // store intermediate result here
    float p_value = 0.0;

    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < width && j < width ) {
        for ( size_t k = 0; k < width; ++k ) {
            float m_elem = Md[i*width+k];
            float n_elem = Nd[k*width+j];
            p_value += m_elem * n_elem;
        }

        Pd[i*width+j] = p_value;
    }
}


/**
 * \brief matrix multiplication, on GPU, loading data tile-wise into shared
 * memory.
 * \param[in] Md, Nd matrices to multiply.
 * \param[out] Pd resulting matrix.
 * \param[in] width size of square matrices.
 */
__global__ void matmul_shared_kernel(float* Md, float* Nd, float* Pd, size_t width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    size_t bx = blockIdx.x, by = blockIdx.y;
    size_t tx = threadIdx.x, ty = threadIdx.y;
    size_t row = by * TILE_WIDTH + ty;
    size_t col = bx * TILE_WIDTH + tx;

    float p_value = 0.0;

    // number_of_tiles = (width + TILE_WIDTH - 1) / TILE_WIDTH
    for ( size_t m = 0; m < width / TILE_WIDTH + 1; ++m ) {
        // collaborative loading of tiles into shared memory
        if ( row < width && (m * TILE_WIDTH + tx) < width )
            Mds[ty][tx] = Md[row*width + (m * TILE_WIDTH + tx)];
        else
            Mds[ty][tx] = 0.0;
        if ( col < width && (m * TILE_WIDTH + ty) < width )
            Nds[ty][tx] = Nd[col + (m * TILE_WIDTH + ty)*width];
        else
            Mds[ty][tx] = 0.0;

        __syncthreads();

        for ( size_t k = 0; k < TILE_WIDTH; ++k )
            p_value += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }

    if ( row < width && col << width )
        Pd[row*width+col] = p_value;
}


void matmul_gpu_global(float* M, float* N, float* P, size_t width)
{
    size_t size = width*width*sizeof(float);
    float *Md, *Nd, *Pd;
    cudaMalloc((void**)&Md, size);
    cudaMalloc((void**)&Nd, size);
    cudaMalloc((void**)&Pd, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

    // cudaDeviceProp dev_prop;
    // cudaGetDeviceProperties(&dev_prop, 0);
    // int blocks_per_sm = dev_prop.maxThreadsPerMultiProcessor \
    //                     / (TILE_WIDTH * TILE_WIDTH);
    // assert(blocks_per_sm <= dev_prop.maxBlocksPerMultiProcessor);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(width/TILE_WIDTH+1, width/TILE_WIDTH+1);

    matmul_global_kernel<<<grid,block>>>(Md, Nd, Pd, width);
    cudaDeviceSynchronize();

    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    cudaFree(Pd);
    cudaFree(Nd);
    cudaFree(Md);
}


void matmul_gpu_shared(float* M, float* N, float* P, size_t width)
{
    size_t size = width*width*sizeof(float);
    float *Md, *Nd, *Pd;
    cudaMalloc((void**)&Md, size);
    cudaMalloc((void**)&Nd, size);
    cudaMalloc((void**)&Pd, size);
    cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(width/TILE_WIDTH+1, width/TILE_WIDTH+1);

    matmul_shared_kernel<<<grid,block>>>(Md, Nd, Pd, width);
    cudaDeviceSynchronize();

    cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);
    cudaFree(Pd);
    cudaFree(Nd);
    cudaFree(Md);
}



/*****************************************************************************/


int main()
{
    srand(time(NULL));

    size_t width = 1<<10;
    float *Mh, *Nh, *Ph, *Pg, *Ps;
    Mh = (float*)malloc(width*width*sizeof(float));
    Nh = (float*)malloc(width*width*sizeof(float));
    Ph = (float*)malloc(width*width*sizeof(float));
    Pg = (float*)malloc(width*width*sizeof(float));
    Ps = (float*)malloc(width*width*sizeof(float));
    if ( Mh == NULL || Nh == NULL || Ph == NULL || Pg == NULL || Ps == NULL ) {
        fprintf(stderr, "malloc failed.\n");
        return EXIT_FAILURE;
    }

    init_data(Mh, width*width);
    init_data(Nh, width*width);

    matmul_cpu(Mh, Nh, Ph, width);
    matmul_gpu_global(Mh, Nh, Pg, width);
    matmul_gpu_shared(Mh, Nh, Ps, width);

    check_results(Ph, Pg, width*width);
    check_results(Ph, Ps, width*width);

    free(Mh);
    free(Nh);
    free(Ph);
    free(Pg);
    free(Ps);

    return EXIT_SUCCESS;
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
