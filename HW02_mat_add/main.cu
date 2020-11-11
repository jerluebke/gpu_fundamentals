/**
 * \file main.cu
 * \brief HW02, Fundamentals of GPU Programming: Comparing execution times for
 *  matrix addition on CPU and GPU.
 * \author Jeremiah Lübke
 * \date 11.11.2020
 */
#include <helper_cuda.h>
#include <helper_timer.h>

#include <assert.h>
#include <stdio.h>
#include <time.h>
#include <float.h>


/*****************************************************************************/


const char* HEADER = "+=========+\n"
                     "| mat_add |\n"
                     "+=========+\n";
const char* RESULT_1_START = "+--------------+\n"
                             "| Task 2 and 3 |\n"
                             "+--------------+\n"
                             "gpu block size: %d x %d\n"
                             "timing runs: %d\n"
                             "(all times in ms)\n";
const char* RESULT_1 = "matrix size: %d x %d\n"
                       "+------+--------+--------+--------+--------+\n"
                       "| mode | t_min  | t_max  | t_mean | t_std  |\n"
                       "+------+--------+--------+--------+--------|\n"
                       "| cpu  | %6.3f | %6.3f | %6.3f | %6.3f |\n"
                       "| gpu  | %6.3f | %6.3f | %6.3f | %6.3f |\n"
                       "+------+--------+--------+--------+--------+\n";
const char* NEW_ENTRY = "\n"
                        "+------------------------------------------+"
                        "\n";

const char* NEXT_TASK = "\n\n+===================================================+\n\n";

const char* RESULT_2_START = "+--------+\n"
                             "| Task 4 |\n"
                             "+--------+\n"
                             "matrix size: %d x %d\n"
                             "timing runs: %d\n"
                             "\n"
                             "Results (ms):\n"
                             "+-------+--------+--------+--------+--------+-------+\n"
                             "| block | t_min  | t_max  | t_mean | t_std  | match |\n"
                             "+-------+--------+--------+--------+--------+-------+\n";
const char* RESULT_2_ROW = "| %dx%d | %6.3f | %6.3f | %6.3f | %6.3f | %-5s |\n";
const char* RESULT_2_END = "+-------+--------+--------+--------+--------+-------+\n";



/*****************************************************************************/


enum Mode { CPU, GPU };

/**
 * \brief time measurement record.
 * \var mode                CPU or GPU.
 * \var runs                number of repetitions.
 * \var block_x, block_y    GPU thread block size (irrelevant for mode == CPU).
 * \var M, N                matrix block size.
 * \var t_max, t_min, t_mean, t_std     timing statistics.
 */
typedef struct timing_result_s {
    Mode mode;
    unsigned runs;
    unsigned block_x, block_y;
    size_t M, N;
    float t_max, t_min, t_mean, t_std;
} timing_result_t;


/**
 * \brief perform time measurement of a piece of code.
 *
 * Runs \c Code \c Runs times and write max, min, mean and std time to the
 * \c Result struct.
 */
#define TIMEIT(Runs, Result, Code) \
    float t_max = 0.0, t_min = FLT_MAX, t_mean = 0.0, t_var = 0.0; \
    for ( unsigned timeit_idx = 0; timeit_idx < Runs; ++timeit_idx ) { \
        static StopWatchInterface* timer = NULL; \
        sdkCreateTimer(&timer); \
        float t_start = sdkGetTimerValue(&timer); \
        sdkStartTimer(&timer); \
        \
        Code \
        \
        sdkStopTimer(&timer); \
      	float t_diff = sdkGetTimerValue(&timer) - t_start; \
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
    }



/**
 * \brief compare arrays with size \c n elementwise by checking whether the
 * absolute value of the difference is smaller than some epsilon > 0.
 * \return 0 on success, 1 on failure.
 */
int check_results(float* host_ref, float* gpu_ref, const int n,
        bool print_status = true)
{
    double eps = 1e-8;
    for ( int i = 0; i < n; ++i ) {
        if ( abs(host_ref[i] - gpu_ref[i]) > eps ) {
            printf("arrays do not match!\n"
                   "[%d] host: %5.2f\tgpu: %5.2f\n",
                   i, host_ref[i], gpu_ref[i]);
            return 1;
        }
    }

    if ( print_status )
        printf("arrays match.\n");
    return 0;
}


/**
 * \brief initialize an array with random floats in the range [0, 255]
 */
void init_data(float* ip, const int size)
{
    for ( int i = 0; i < size; ++i )
        ip[i] = (float)(rand() & 0xFF);
}


/*****************************************************************************/


/**
 * \brief basic matrix type, which holds its data in linear memory and knows
 * its size.
 */
typedef struct matrix_s {
    size_t M, N;
    float* data;
} matrix_t;


/**
 * \brief constructor of matrix type, allocate data with given size.
 * \param[out] m new matrix object
 * \param[in] M, N size of matrix
 * \param[in] data initial data of matrix (optional)
 * \return 0 on success, 1 on failure of malloc
 */
int new_matrix(matrix_t* m, size_t M, size_t N, const float* data = NULL)
{
    m->M = M;
    m->N = N;
    m->data = NULL;
    if ( (m->data = (float*)malloc(M*N*sizeof(float))) == NULL ) {
        fprintf(stderr, "malloc failed!\n");
        return 1;
    }
    if ( data != NULL ) {
        memcpy(m->data, data, M*N*sizeof(float));
    }
    return 0;
}
int new_matrix_d(matrix_t* m, size_t M, size_t N, const float* data = NULL)
{
    m->M = M;
    m->N = N;
    m->data = NULL;
    if ( cudaMalloc((void**)&m->data, M*N*sizeof(float)) != cudaSuccess ) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return 1;
    }
    if ( data != NULL
            && cudaMemcpy(m->data, data, M*N*sizeof(float),
                          cudaMemcpyHostToDevice) != cudaSuccess ) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        cudaFree(m->data);
        return 1;
    }
    return 0;
}


/**
 * \brief destructor of matrix type, frees allocated memory.
 */
void free_matrix(matrix_t m)
{
    free(m.data);
}
void free_matrix_d(matrix_t m)
{
    cudaFree(m.data);
}


/**
 * \brief read matrix element.
 * \param mat matrix to access.
 * \param i, j position of element.
 * \return element at (i, j).
 */
inline float matrix_get(const matrix_t mat, size_t i, size_t j)
{
    return mat.data[i*mat.N+j];
}
__device__ inline
float matrix_get_d(const matrix_t mat, size_t i, size_t j)
{
    return mat.data[i*mat.N+j];
}


/**
 * \brief write matrix element.
 * \param mat matrix to access.
 * \param i, j position of element.
 * \param val new value of element at (i, j).
 */
inline void matrix_set(matrix_t mat, size_t i, size_t j, float val)
{
    mat.data[i*mat.N+j] = val;
}
__device__ inline
void matrix_set_d(matrix_t mat, size_t i, size_t j, float val)
{
    mat.data[i*mat.N+j] = val;
}


/**
 * \brief pretty print matrix
 */
void matrix_print(const matrix_t mat, const char* name)
{
    int offset = strlen(name) + 4;
    printf("%s = ", name);
    for ( size_t i = 0; i < mat.M; ++i ) {
        printf("%*c", i == 0 ? 0 : offset,
                      i == 0 ? '/' : i < mat.M-1 ? '|' : '\\');
        for ( size_t j = 0; j < mat.N; ++j )
            printf(" %5.2f ", matrix_get(mat, i, j));
        printf("%c\n", i == 0 ? '\\' : i < mat.M-1 ? '|' : '/');
    }
    printf("\n");
}


/*****************************************************************************/


/**
 * \brief element-wise matrix addition, sequentially on CPU.
 * \param[in] hA, hB matrices to add
 * \param[out] hC resulting matrix
 * \param[in] timing_runs number of repetitions for time measurement
 * \param[out] timing_result result struct for time measurement
 */
void mat_add_cpu(const matrix_t hA, const matrix_t hB, matrix_t hC,
        const unsigned timing_runs = 1, timing_result_t* timing_result = NULL)
{
    size_t size_A = hA.M * hA.N;
    size_t size_B = hB.M * hB.N;
    size_t size_C = hC.M * hC.N;
    assert(size_A == size_B && size_B == size_C);

    if ( timing_result != NULL ) {
        timing_result->mode = CPU;
        timing_result->runs = timing_runs;
        timing_result->M = hA.M;
        timing_result->N = hA.N;
    }

    TIMEIT(timing_runs, timing_result,
        for ( size_t i = 0; i < size_A; ++i )
            hC.data[i] = hA.data[i] + hB.data[i];
    )
}


__global__ void mat_add_kernel(const matrix_t dA, const matrix_t dB, matrix_t dC)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row < dA.M && col < dA.N ) {
        float sum = matrix_get_d(dA, row, col) + matrix_get_d(dB, row, col);
        matrix_set_d(dC, row, col, sum);
    }
}


/**
 * \brief element-wise matrix addition, in parallel on GPU.
 * \param[in] hA, hB matrices to add.
 * \param[out] hC resulting matrix.
 * \param[in] block_width, block_height number of threads per block in x and y
 *      direction.
 * \param[in] timing_runs number of repetitions for time measurement
 * \param[out] timing_result result struct for time measurement
 */
void mat_add_gpu(const matrix_t hA, const matrix_t hB, matrix_t hC,
        const unsigned block_width, const unsigned block_height,
        const unsigned timing_runs = 1, timing_result_t* timing_result = NULL)
{
    assert(hA.M == hB.M && hB.M == hC.M);
    assert(hA.N == hB.N && hB.N == hC.N);

    matrix_t dA, dB, dC;
    new_matrix_d(&dA, hA.M, hA.N, hA.data);
    new_matrix_d(&dB, hB.M, hB.N, hB.data);
    new_matrix_d(&dC, hC.M, hC.N, hC.data);

    dim3 grid(dA.N / block_height + 1, dA.M / block_width + 1);
    dim3 block(block_height, block_width);

    if ( timing_result != NULL ) {
        timing_result->mode = GPU;
        timing_result->runs = timing_runs;
        timing_result->M = hA.M;
        timing_result->N = hA.N;
        timing_result->block_x = block.x;
        timing_result->block_y = block.y;
    }

    #define COMMA ,
    TIMEIT(timing_runs, timing_result,
        mat_add_kernel<<<grid COMMA block>>>(dA COMMA dB COMMA dC);
        cudaDeviceSynchronize();
    )
    #undef COMMA

    if ( cudaMemcpy(hC.data, dC.data, dC.M*dC.N*sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess ) {
        fprintf(stderr, "cudaMemcpy failed!\n");
    }

    free_matrix_d(dA);
    free_matrix_d(dB);
    free_matrix_d(dC);
}


/*****************************************************************************/


int main()
{
    srand(time(NULL));


    /**********
     * setup  *
     **********/
    const int matrix_size_num = 5;
    const int block_size_num = 4;
    const int matrix_sizes[matrix_size_num][2] = {
        { 10, 10},
        { 100, 100},
        { 1000, 1000},
        { 500, 2000},
        { 100, 10000},
    };
    const int block_sizes[block_size_num][2] = {
        { 16, 16 },
        { 16, 32 },
        { 32, 16 },
        { 32, 32 },
    };

    const int timing_runs = 10000;
    timing_result_t cpu_time, gpu_time;
    matrix_t hA, hB, hC, dC;


    /************************************************
     * comparing CPU and GPU execution times for    *
     *  different matrix sizes (tasks 2, 3)         *
     ************************************************/
    puts(HEADER);

    int block_x = block_sizes[0][0];
    int block_y = block_sizes[0][1];
    printf(RESULT_1_START, block_x, block_y, timing_runs);
    puts(NEW_ENTRY);

    for ( int i = 0; i < matrix_size_num; ++i ) {
        int M = matrix_sizes[i][0];
        int N = matrix_sizes[i][1];
        int size = M * N;
        new_matrix(&hA, M, N); new_matrix(&hB, M, N);
        new_matrix(&hC, M, N); new_matrix(&dC, M, N);
        init_data(hA.data, size); init_data(hB.data, size);

        mat_add_cpu(hA, hB, hC, timing_runs, &cpu_time);
        mat_add_gpu(hA, hB, dC, block_x, block_y, timing_runs, &gpu_time);

        printf(RESULT_1, M, N,
               cpu_time.t_min, cpu_time.t_max,
               cpu_time.t_mean, cpu_time.t_std,
               gpu_time.t_min, gpu_time.t_max,
               gpu_time.t_mean, gpu_time.t_std);
        check_results(hC.data, dC.data, size);

        if ( i < matrix_size_num-1 )
            puts(NEW_ENTRY);

        free_matrix(hA); free_matrix(hB);
        free_matrix(hC); free_matrix(dC);
    }


    /********************************************
     * comparing GPU execution times for        *
     *  different thread block sizes (task 4)   *
     ********************************************/
    puts(NEXT_TASK);

    int M = matrix_sizes[matrix_size_num-1][0];
    int N = matrix_sizes[matrix_size_num-1][1];
    int size = M * N;
    new_matrix(&hA, M, N); new_matrix(&hB, M, N);
    new_matrix(&hC, M, N); new_matrix(&dC, M, N);
    init_data(hA.data, size); init_data(hB.data, size);
    mat_add_cpu(hA, hB, hC);

    printf(RESULT_2_START, M, N, timing_runs);
    for ( int i = 0; i < block_size_num; ++i ) {
        int block_x = block_sizes[i][0];
        int block_y = block_sizes[i][1];
        mat_add_gpu(hA, hB, dC, block_x, block_y, timing_runs, &gpu_time);
        int match = check_results(hC.data, dC.data, size, false);
        printf(RESULT_2_ROW, block_x, block_y,
                             gpu_time.t_min, gpu_time.t_max,
                             gpu_time.t_mean, gpu_time.t_std,
                             match == 0 ? "Yes" : "No");
    }
    puts(RESULT_2_END);

    free_matrix(hA); free_matrix(hB);
    free_matrix(hC); free_matrix(dC);


    return 0;
}


/* vim: set tw=79 ts=4 sw=4 et ic ai : */
