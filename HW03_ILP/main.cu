/**
 * \file main.cu
 * \brief HW03, Fundamentals of GPU Programming: Investigating ILP (Instruction
 *  Level Parallelism)
 * \author Jeremiah Lübke
 * \date 23.11.2020
 * \copyright MIT License
 */
#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif

#include <helper_cuda.h>
#include <helper_timer.h>

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
        /* float t_start = sdkGetTimerValue(&timer); */ \
        sdkStartTimer(&timer); \
        \
        Code \
        \
        sdkStopTimer(&timer); \
      	float t_diff = sdkGetTimerValue(&timer) /*- t_start */; \
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


const size_t NUM_ITERATIONS = 10000;
__device__ float GLOBAL_ARRAY[32];


__global__ void ilp32_kernel(float b, float c)
{
    float a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0,
        a9 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0, a16 = 0,
        a17 = 0, a18 = 0, a19 = 0, a20 = 0, a21 = 0, a22 = 0, a23 = 0, a24 = 0,
        a25 = 0, a26 = 0, a27 = 0, a28 = 0, a29 = 0, a30 = 0, a31 = 0, a32 = 0;
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c; a2 = a2*b+c; a3 = a3*b+c; a4 = a4*b+c;
        a5 = a5*b+c; a6 = a6*b+c; a7 = a7*b+c; a8 = a8*b+c;
        a9 = a9*b+c; a10 = a10*b+c; a11 = a11*b+c; a12 = a12*b+c;
        a13 = a13*b+c; a14 = a14*b+c; a15 = a15*b+c; a16 = a16*b+c;
        a17 = a17*b+c; a18 = a18*b+c; a19 = a19*b+c; a20 = a20*b+c;
        a21 = a21*b+c; a22 = a22*b+c; a23 = a23*b+c; a24 = a24*b+c;
        a25 = a25*b+c; a26 = a26*b+c; a27 = a27*b+c; a28 = a28*b+c;
        a29 = a29*b+c; a30 = a30*b+c; a31 = a31*b+c; a32 = a32*b+c;
    }
    GLOBAL_ARRAY[0] = a1; GLOBAL_ARRAY[1] = a2; GLOBAL_ARRAY[2] = a3; GLOBAL_ARRAY[3] = a4;
    GLOBAL_ARRAY[4] = a5; GLOBAL_ARRAY[5] = a6; GLOBAL_ARRAY[6] = a7; GLOBAL_ARRAY[7] = a8;
    GLOBAL_ARRAY[8] = a9; GLOBAL_ARRAY[9] = a10; GLOBAL_ARRAY[10] = a11; GLOBAL_ARRAY[11] = a12;
    GLOBAL_ARRAY[12] = a13; GLOBAL_ARRAY[13] = a14; GLOBAL_ARRAY[14] = a15; GLOBAL_ARRAY[15] = a16;
    GLOBAL_ARRAY[16] = a17; GLOBAL_ARRAY[17] = a18; GLOBAL_ARRAY[18] = a19; GLOBAL_ARRAY[19] = a20;
    GLOBAL_ARRAY[20] = a21; GLOBAL_ARRAY[21] = a22; GLOBAL_ARRAY[22] = a23; GLOBAL_ARRAY[23] = a24;
    GLOBAL_ARRAY[24] = a25; GLOBAL_ARRAY[25] = a26; GLOBAL_ARRAY[26] = a27; GLOBAL_ARRAY[27] = a28;
    GLOBAL_ARRAY[28] = a29; GLOBAL_ARRAY[29] = a30; GLOBAL_ARRAY[30] = a31; GLOBAL_ARRAY[31] = a32;
}

__global__ void ilp16_kernel(float b, float c)
{
    float a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0,
        a9 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0, a16 = 0;
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c; a2 = a2*b+c; a3 = a3*b+c; a4 = a4*b+c;
        a5 = a5*b+c; a6 = a6*b+c; a7 = a7*b+c; a8 = a8*b+c;
        a9 = a9*b+c; a10 = a10*b+c; a11 = a11*b+c; a12 = a12*b+c;
        a13 = a13*b+c; a14 = a14*b+c; a15 = a15*b+c; a16 = a16*b+c;
    }
    GLOBAL_ARRAY[0] = a1; GLOBAL_ARRAY[1] = a2; GLOBAL_ARRAY[2] = a3; GLOBAL_ARRAY[3] = a4;
    GLOBAL_ARRAY[4] = a5; GLOBAL_ARRAY[5] = a6; GLOBAL_ARRAY[6] = a7; GLOBAL_ARRAY[7] = a8;
    GLOBAL_ARRAY[8] = a9; GLOBAL_ARRAY[9] = a10; GLOBAL_ARRAY[10] = a11; GLOBAL_ARRAY[11] = a12;
    GLOBAL_ARRAY[12] = a13; GLOBAL_ARRAY[13] = a14; GLOBAL_ARRAY[14] = a15; GLOBAL_ARRAY[15] = a16;
}

__global__ void ilp8_kernel(float b, float c)
{
    float a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0;
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c; a2 = a2*b+c; a3 = a3*b+c; a4 = a4*b+c;
        a5 = a5*b+c; a6 = a6*b+c; a7 = a7*b+c; a8 = a8*b+c;
    }
    GLOBAL_ARRAY[0] = a1; GLOBAL_ARRAY[1] = a2; GLOBAL_ARRAY[2] = a3; GLOBAL_ARRAY[3] = a4;
    GLOBAL_ARRAY[4] = a5; GLOBAL_ARRAY[5] = a6; GLOBAL_ARRAY[6] = a7; GLOBAL_ARRAY[7] = a8;
}

__global__ void ilp4_kernel(float b, float c)
{
    float a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c; a2 = a2*b+c; a3 = a3*b+c; a4 = a4*b+c;
    }
    GLOBAL_ARRAY[0] = a1; GLOBAL_ARRAY[1] = a2; GLOBAL_ARRAY[2] = a3; GLOBAL_ARRAY[3] = a4;
}

__global__ void ilp1_kernel(float b, float c)
{
    float a1 = 0;
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c;
    }
    GLOBAL_ARRAY[0] = a1;
}


void ilp_test(int num_blocks, int block_size,
        const unsigned timing_runs, timing_result_t timing_results[])
{
    float b = (float)(rand() & 0xFF);
    float c = (float)(rand() & 0xFF);

    #define COMMA ,
    TIMEIT(timing_runs, (&timing_results[0]),
        ilp32_kernel<<<num_blocks COMMA block_size>>>(b COMMA c);
        cudaDeviceSynchronize();
    )
    TIMEIT(timing_runs, (&timing_results[1]),
        ilp16_kernel<<<num_blocks COMMA block_size>>>(b COMMA c);
        cudaDeviceSynchronize();
    )
    TIMEIT(timing_runs, (&timing_results[2]),
        ilp8_kernel<<<num_blocks COMMA block_size>>>(b COMMA c);
        cudaDeviceSynchronize();
    )
    TIMEIT(timing_runs, (&timing_results[3]),
        ilp4_kernel<<<num_blocks COMMA block_size>>>(b COMMA c);
            cudaDeviceSynchronize();
    )
    TIMEIT(timing_runs, (&timing_results[4]),
        ilp1_kernel<<<num_blocks COMMA block_size>>>(b COMMA c);
        cudaDeviceSynchronize();
    )
    #undef COMMA
}


/*****************************************************************************/


// const char* TIMING_RESULT = "\t%6.3f\t%6.3f\t%6.3f\t%6.3f";
const char* TIMING_RESULT = "\t%7.6f";


int main()
{
    srand(time(NULL));

    const int timing_runs = 1000;
    const int config_number = 6;
    const int ilp_config_number = 5;
    const int blocks_per_sm[config_number] = { 32, 16, 8, 4, 2, 1 };
    const int ilp_config[ilp_config_number] = { 32, 16, 8, 4, 1 };
    timing_result_t timing_results[ilp_config_number];

    cudaDeviceProp dev_prop;
    cudaGetDeviceProperties(&dev_prop, 0);

    for ( int i = 0; i < config_number; ++i ) {
        if ( blocks_per_sm[i] > dev_prop.maxBlocksPerMultiProcessor ) {
            fprintf(stderr, "%d exceeds maximal number of blocks per sm (%d)!\n",
                    blocks_per_sm[i], dev_prop.maxBlocksPerMultiProcessor);
            continue;
        }

        int block_size = dev_prop.maxThreadsPerMultiProcessor / blocks_per_sm[i];
        ilp_test(dev_prop.multiProcessorCount * blocks_per_sm[i], block_size,
                timing_runs, timing_results);

        fprintf(stderr, "block size: %d\ttotal threads: %d\n", block_size,
                dev_prop.multiProcessorCount*blocks_per_sm[i]*block_size);
        printf("%d", block_size);
        for ( int j = 0; j < ilp_config_number; ++j )
            printf(TIMING_RESULT, timing_results[j].t_mean / ilp_config[j]);
            // printf(TIMING_RESULT, timing_results[j].t_max, timing_results[j].t_min,
            //                       timing_results[j].t_mean, timing_results[j].t_std);
        printf("\n");
    }

    return 0;
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
