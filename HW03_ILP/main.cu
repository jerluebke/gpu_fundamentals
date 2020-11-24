/**
 * \file main.cu
 * \brief HW03, Fundamentals of GPU Programming: Investigating ILP (Instruction
 *  Level Parallelism)
 * \author Jeremiah Lübke
 * \date 23.11.2020
 * \copyright MIT License
 */
#include <helper_cuda.h>
#include <helper_timer.h>

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


const size_t TIMING_RUNS = 1000;
const size_t NUM_ITERATIONS = 1 << 18;
__device__ float GLOBAL_ARRAY[16*1024];


__global__ void ilp16_kernel(float b, float c)
{
    size_t offset = 16 * threadIdx.x;
    float a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0,
        a9 = 0, a10 = 0, a11 = 0, a12 = 0, a13 = 0, a14 = 0, a15 = 0, a16 = 0;
    #pragma unroll 16
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c; a2 = a2*b+c; a3 = a3*b+c; a4 = a4*b+c;
        a5 = a5*b+c; a6 = a6*b+c; a7 = a7*b+c; a8 = a8*b+c;
        a9 = a9*b+c; a10 = a10*b+c; a11 = a11*b+c; a12 = a12*b+c;
        a13 = a13*b+c; a14 = a14*b+c; a15 = a15*b+c; a16 = a16*b+c;
    }
    GLOBAL_ARRAY[offset+0] = a1; GLOBAL_ARRAY[offset+1] = a2;
    GLOBAL_ARRAY[offset+2] = a3; GLOBAL_ARRAY[offset+3] = a4;
    GLOBAL_ARRAY[offset+4] = a5; GLOBAL_ARRAY[offset+5] = a6;
    GLOBAL_ARRAY[offset+6] = a7; GLOBAL_ARRAY[offset+7] = a8;
    GLOBAL_ARRAY[offset+8] = a9; GLOBAL_ARRAY[offset+9] = a10;
    GLOBAL_ARRAY[offset+10] = a11; GLOBAL_ARRAY[offset+11] = a12;
    GLOBAL_ARRAY[offset+12] = a13; GLOBAL_ARRAY[offset+13] = a14;
    GLOBAL_ARRAY[offset+14] = a15; GLOBAL_ARRAY[offset+15] = a16;
}

__global__ void ilp8_kernel(float b, float c)
{
    size_t offset = 8 * threadIdx.x;
    float a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0, a8 = 0;
    #pragma unroll 16
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c; a2 = a2*b+c; a3 = a3*b+c; a4 = a4*b+c;
        a5 = a5*b+c; a6 = a6*b+c; a7 = a7*b+c; a8 = a8*b+c;
    }
    GLOBAL_ARRAY[offset+0] = a1; GLOBAL_ARRAY[offset+1] = a2;
    GLOBAL_ARRAY[offset+2] = a3; GLOBAL_ARRAY[offset+3] = a4;
    GLOBAL_ARRAY[offset+4] = a5; GLOBAL_ARRAY[offset+5] = a6;
    GLOBAL_ARRAY[offset+6] = a7; GLOBAL_ARRAY[offset+7] = a8;
}

__global__ void ilp4_kernel(float b, float c)
{
    size_t offset = 4 * threadIdx.x;
    float a1 = 0, a2 = 0, a3 = 0, a4 = 0;
    #pragma unroll 16
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c; a2 = a2*b+c; a3 = a3*b+c; a4 = a4*b+c;
    }
    GLOBAL_ARRAY[offset+0] = a1; GLOBAL_ARRAY[offset+1] = a2;
    GLOBAL_ARRAY[offset+2] = a3; GLOBAL_ARRAY[offset+3] = a4;
}

__global__ void ilp1_kernel(float b, float c)
{
    float a1 = 0;
    #pragma unroll 16
    for ( size_t i = 0; i < NUM_ITERATIONS; ++i ) {
        a1 = a1*b+c;
    }
    GLOBAL_ARRAY[threadIdx.x] = a1;
}


void ilp_test(int block_size, timing_result_t timing_results[])
{
    float b = (float)(rand() & 0xFF);
    float c = (float)(rand() & 0xFF);

    #define COMMA ,
    TIMEIT(TIMING_RUNS, (&timing_results[0]),
        ilp16_kernel<<<1 COMMA block_size>>>(b COMMA c);
        cudaDeviceSynchronize();
    )
    TIMEIT(TIMING_RUNS, (&timing_results[1]),
        ilp8_kernel<<<1 COMMA block_size>>>(b COMMA c);
        cudaDeviceSynchronize();
    )
    TIMEIT(TIMING_RUNS, (&timing_results[2]),
        ilp4_kernel<<<1 COMMA block_size>>>(b COMMA c);
            cudaDeviceSynchronize();
    )
    TIMEIT(TIMING_RUNS, (&timing_results[3]),
        ilp1_kernel<<<1 COMMA block_size>>>(b COMMA c);
        cudaDeviceSynchronize();
    )
    #undef COMMA
}


/*****************************************************************************/


const char* HEADER = "+=====+\n"
                     "| ILP |\n"
                     "+=====+\n"
                     "timing runs: %d\n"
                     "\n"
                     "Mean time per %d instructions (in ms):\n"
                     "+---------++-------+-------+-------+-------++-----+-----+-----+\n"
                     "| threads || ILP16 | ILP8  | ILP4  | ILP1  || r16 | r8  | r4  |\n"
                     "+---------++-------+-------+-------+-------++-----+-----+-----+\n";
const char* RESULT_ROW = "| %7d || %5.3f | %5.3f | %5.3f | %5.3f || %3.1f | %3.1f | %3.1f |\n";
const char* RESULT_END = "+---------++-------+-------+-------+-------++-----+-----+-----+";


int main()
{
    srand(time(NULL));

    const int ilp_config_number = 4;
    timing_result_t timing_results[ilp_config_number];

    printf(HEADER, TIMING_RUNS, NUM_ITERATIONS);
    for ( int i = 1; i <= 32; ++i ) {
        ilp_test(i*32, timing_results);
        float t0 = timing_results[0].t_mean / 16;
        float t1 = timing_results[1].t_mean / 8;
        float t2 = timing_results[2].t_mean / 4;
        float t3 = timing_results[3].t_mean;
        printf(RESULT_ROW, i*32, t0, t1, t2, t3, t0 / t3, t1 / t3, t2 / t3);
    }
    puts(RESULT_END);

    return 0;
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
