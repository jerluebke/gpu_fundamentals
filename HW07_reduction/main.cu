/**
 * \file main.cu
 * \brief HW07, first sum-reduction algorithm
 * \author Jeremiah Lübke
 * \date 14.01.2021
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


void reduce_sum_cpu(float *odata, float *idata, size_t size)
{
    *odata = 0.0;
    for ( size_t i = 0; i < size; ++i )
        *odata += idata[i];
}


__global__ void reduce_sum_global_kernel(float *odata, float *idata, size_t size)
{
    for ( size_t i = blockIdx.x * gridDim.x + threadIdx.x;
            i < size;
            i += blockDim.x * gridDim.x ) {
        atomicAdd(odata, idata[i]);
    }
}


#define ILP 8
__global__ void reduce_sum_shared_kernel(float *odata, float *idata, size_t size)
{
    __shared__ float bsum[1];
    size_t bi = ILP * blockIdx.x * gridDim.x;
    size_t ti = ILP * threadIdx.x;

    // sub-reduction on thread-level with ILP
    float tsum = 0.0;
    for ( size_t i = 0; i < ILP; ++i ) {
        if ( ti + bi + i > size )
            break;
        tsum += idata[ti + bi + i];
    }

    __syncthreads();

    // sub-reduction on block-level
    atomicAdd(bsum, tsum);

    __syncthreads();

    // final reduction on grid-level in global memory
    if ( ti == 0 )
        atomicAdd(odata, bsum[0]);
}


/*****************************************************************************/


#define SIZE (1 << 20)
void test_reduction(int block_size)
{
    const int number_of_floats = SIZE / sizeof(float);

    float zero = 0.0;
    float *hData, *dData, *hSum, *refSum, *dSum;
    hData = (float*)malloc(SIZE);
    hSum = (float*)malloc(sizeof(float));
    refSum = (float*)malloc(sizeof(float));
    cudaMalloc((void**)&dData, SIZE);
    cudaMalloc((void**)&dSum, sizeof(float));
    init_data(hData, number_of_floats);
    *hSum = zero;
    cudaMemcpy(dData, hData, SIZE, cudaMemcpyHostToDevice);

    int grid_size = number_of_floats / block_size + 1;
    reduce_sum_cpu(hSum, hData, number_of_floats);

    cudaMemcpy(dSum, &zero, sizeof(float), cudaMemcpyHostToDevice);
    reduce_sum_global_kernel<<<grid_size, block_size>>>(dSum, dData, number_of_floats);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(refSum, dSum, sizeof(float), cudaMemcpyDeviceToHost);
    if ( *hSum == *refSum )
        printf("results match.\n");
    else
        printf("results do not match: %f != %f\n", *hSum, *refSum);

    cudaMemcpy(dSum, &zero, sizeof(float), cudaMemcpyHostToDevice);
    reduce_sum_shared_kernel<<<grid_size, block_size>>>(dSum, dData, number_of_floats);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    cudaMemcpy(refSum, dSum, sizeof(float), cudaMemcpyDeviceToHost);
    if ( *hSum == *refSum )
        printf("results match.\n");
    else
        printf("results do not match: %f != %f\n", *hSum, *refSum);
}


/*****************************************************************************/
/*****************************************************************************/


const char* HEADER = "+===============+\n"
                     "| SUM REDUCTION |\n"
                     "+===============+\n";
// const char* RESULT = "%s (%d): %5.3f GB/s, %5.3f GB/s, %5.3f\n";
const char* END = "+===============+\n";


int main()
{
    srand(time(NULL));
    const int num_block_sizes = 4;
    int block_sizes[num_block_sizes] = { 64, 128, 256, 512 };
    // timing_result_t shared_time, texture_time;

    puts(HEADER);
    for ( int i = 0; i < num_block_sizes; ++i )
        test_reduction(block_sizes[i]);
    puts(END);
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
