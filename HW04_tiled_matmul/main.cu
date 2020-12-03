/**
 * \file main.cu
 * \brief HW04, tiled matrix multiplication
 * \author Jeremiah Lübke
 * \date 03.12.2020
 * \copyright MIT License
 */
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_timer.h>

#include <assert.h>
#include <float.h>
#include <stdio.h>
#include <time.h>


#define SUBMISSION 1


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
        bool print_status = false)
{
    double eps = 1e-1;
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



/***************************************************************************** 
 * MATRIX                                                                    *
 *****************************************************************************/

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
__host__ __device__
inline float matrix_get(const matrix_t mat, size_t i, size_t j)
{
    if ( i < mat.M && j < mat.N )
        return mat.data[i*mat.N+j];
    else
        return 0.0;
}


/**
 * \brief write matrix element.
 * \param mat matrix to access.
 * \param i, j position of element.
 * \param val new value of element at (i, j).
 */
__host__ __device__
inline void matrix_set(matrix_t mat, size_t i, size_t j, float val)
{
    if ( i < mat.M && j < mat.N )
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



/***************************************************************************** 
 * TILED MATRIX MULTIPLICATION                                               *
 *****************************************************************************/

/**
 * \brief check whether the given block configuration allows full occupancy of
 * the device.
 */
void check_device_properties(const size_t tile_shape[2])
{
    /*
        max_thr_per_sm / blk_size <= max_blk_per_sm
        max_shd_mem / (2*blk_size*flt_size) >= max_blk_per_sm
    */

    int max_threads_per_sm;
    int max_blocks_per_sm;
    int max_shared_memory_per_sm;
    cudaDeviceGetAttribute(&max_threads_per_sm,
            cudaDevAttrMaxThreadsPerMultiProcessor, 0);
#if SUBMISSION      // cudaDevAttrMaxBlocksPerMultiprocessor is not available
                    // on the shared GPU cluster.
    max_blocks_per_sm = 16;
#else
    cudaDeviceGetAttribute(&max_blocks_per_sm,
            cudaDevAttrMaxBlocksPerMultiprocessor, 0);
#endif
    cudaDeviceGetAttribute(&max_shared_memory_per_sm,
            cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);

    int block_size = tile_shape[0] * tile_shape[1];
    int number_of_blocks_by_threads = max_threads_per_sm / block_size;
    int number_of_blocks_by_shared_mem = max_shared_memory_per_sm \
        / (2*block_size*sizeof(float));

    if ( number_of_blocks_by_threads > max_blocks_per_sm )
        fprintf(stderr, "WARNING: blocks too small. block configuration "
                "%zux%zu allows %d blocks, but only %d are available on the "
                "multiprocessors!\n",
                tile_shape[0], tile_shape[1], number_of_blocks_by_threads,
                max_blocks_per_sm);
    if ( number_of_blocks_by_shared_mem < max_blocks_per_sm )
        fprintf(stderr, "WARNING: blocks require too much shared memory. "
                "block configuration %zux%zu allows only %d " "blocks, but %d "
                "are available on the multiprocessors!\n",
                tile_shape[0], tile_shape[1], number_of_blocks_by_shared_mem,
                max_blocks_per_sm);
}


/**
 * \brief Allocates four test matrices on host and device side, initialize the
 * first two with random elements.
 *
 * \return 0 on success, something else if allocating or copying of memory
 * fails.
 *
 * \see perform_matmul_test for the purpose of each matrix.
 */
int allocate_and_init_test_matrices(matrix_t test_matrices_h[4],
        matrix_t test_matrices_d[4], const size_t matrix_shape[3])
{
    int status = 0;

    status |= new_matrix(&test_matrices_h[0], matrix_shape[0], matrix_shape[1]);
    status |= new_matrix(&test_matrices_h[1], matrix_shape[1], matrix_shape[2]);
    status |= new_matrix(&test_matrices_h[2], matrix_shape[0], matrix_shape[2]);
    status |= new_matrix(&test_matrices_h[3], matrix_shape[0], matrix_shape[2]);

    init_data(test_matrices_h[0].data, test_matrices_h[0].M * test_matrices_h[0].N);
    init_data(test_matrices_h[1].data, test_matrices_h[1].M * test_matrices_h[1].N);

    status |= new_matrix_d(&test_matrices_d[0], matrix_shape[0], matrix_shape[1],
            test_matrices_h[0].data);
    status |= new_matrix_d(&test_matrices_d[1], matrix_shape[1], matrix_shape[2],
            test_matrices_h[1].data);
    status |= new_matrix_d(&test_matrices_d[2], matrix_shape[0], matrix_shape[2]);
    status |= new_matrix_d(&test_matrices_d[3], matrix_shape[0], matrix_shape[2]);

    return status;
}


void free_test_matrices(matrix_t test_matrices_h[4],
        matrix_t test_matrices_d[4])
{
    for ( int i = 0; i < 4; ++i ) {
        free_matrix(test_matrices_h[i]);
        free_matrix_d(test_matrices_d[i]);
    }
}


/*****************************************************************************/


void mat_mul_cpu(matrix_t hA, matrix_t hB, matrix_t hC)
{
    for ( size_t i = 0; i < hC.M; ++i )
        for ( size_t j = 0; j < hC.N; ++j ) {
            float c = 0.0;
            for ( size_t k = 0; k < hA.N; ++k )
                c += matrix_get(hA, i, k) * matrix_get(hB, k, j);
            matrix_set(hC, i, j, c);
        }
}


__global__ void mat_mul_global_kernel(matrix_t dA, matrix_t dB, matrix_t dC)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if ( i < dC.M && j < dC.N ) {
        float sum = 0.0;
        for ( size_t k = 0; k < dA.N; ++k )
            sum += matrix_get(dA, i, k) * matrix_get(dB, k, j);
        matrix_set(dC, i, j, sum);
    }
}


__global__ void mat_mul_shared_kernel(matrix_t dA, matrix_t dB, matrix_t dC,
        const size_t tile_width, const size_t tile_height)
{
    extern __shared__ float smem[];
    matrix_t dAs = { tile_height, tile_width, smem };
    matrix_t dBs = { tile_width, tile_height,
                     &smem[tile_width*tile_height] };

    size_t bx = blockIdx.x, by = blockIdx.y;
    size_t tx = threadIdx.x, ty = threadIdx.y;
    size_t row = by * tile_height + ty;
    size_t col = bx * tile_width + tx;

    float sum = 0.0;

    for ( size_t m = 0; m < dA.N / tile_width + 1; ++m ) {
        // collaborative loading of tiles into shared memory
        // matrix_* functions take care of bounds checking
        matrix_set(dAs, ty, tx, matrix_get(dA, row, m*tile_width+tx));
        matrix_set(dBs, ty, tx, matrix_get(dB, m*tile_width+ty, col));
        __syncthreads();

        for ( size_t k = 0; k < tile_width; ++k )
            sum += matrix_get(dAs, ty, k) * matrix_get(dBs, k, tx);
        __syncthreads();
    }

    matrix_set(dC, row, col, sum);
}


/**
 * \brief Run three matrix multiplication algorithms (CPU, GPU with global
 * memory, GPU with shared memory) on given test matrices. Records timing and
 * checks for correctness (with CPU version as reference).
 *
 * \param test_matrices_h   array of four matrices for the CPU calculation.
 * \param test_matrices_d   array of four matrices for the GPU calculations.
 * \param tile_shape        two-element array: <tt>[tile_width,
 *      tile_height]</tt>.
 * \param timing_results    array of three timing result structs for the three
 *      algorithms.
 * \param timing_runs       number of timing runs.
 * \return  -1 if cudaMemcpy fails, else 0
 *
 * \note The input matrices are expected as four-element arrays, where the
 * first two are denoted by A and B, readily allocated (on CPU and GPU resp.)
 * and initialized. The third and fourth element of the host matrices are
 * respectively the result matrix <tt>C=A*B</tt> of the CPU calculation and the
 * target matrix, into which the GPU results are to be copyied for comparison.
 * The third and fourth element of the device matrices serve as result matrices
 * for the two GPU algorithms.
 */
int perform_matmul_test(
        matrix_t test_matrices_h[4],
        matrix_t test_matrices_d[4],
        const size_t tile_shape[2],
        timing_result_t timing_result[3],
        const size_t timing_runs[2],
        bool do_cpu = true)
{
    matrix_t hA = test_matrices_h[0];
    matrix_t hB = test_matrices_h[1];
    matrix_t hC = test_matrices_h[2];
    matrix_t Cref = test_matrices_h[3];
    matrix_t dA = test_matrices_d[0];
    matrix_t dB = test_matrices_d[1];
    matrix_t dCg = test_matrices_d[2];
    matrix_t dCs = test_matrices_d[3];

    size_t tile_size = tile_shape[0] * tile_shape[1] * sizeof(float);
    dim3 block(tile_shape[0], tile_shape[1]);
    dim3 grid(hC.N / tile_shape[0] + 1, hC.M / tile_shape[1] + 1);

    /* mat_mul_cpu(hA, hB, hC);
    mat_mul_shared_kernel<<<grid, block, 2*tile_size>>>(dA, dB, dCs,
            tile_shape[0], tile_shape[1]);
    cudaDeviceSynchronize(); */


#define COMMA ,
    if ( do_cpu )
        TIMEIT(timing_runs[0], (&timing_result[0]), mat_mul_cpu(hA COMMA hB COMMA hC);)

    TIMEIT(timing_runs[1], (&timing_result[1]),
            mat_mul_global_kernel<<<grid COMMA block>>>(dA COMMA dB COMMA dCg);
            cudaDeviceSynchronize();)
    TIMEIT(timing_runs[1], (&timing_result[2]),
            mat_mul_shared_kernel<<<grid COMMA block COMMA 2*tile_size>>>(
                dA COMMA dB COMMA dCs COMMA tile_shape[0] COMMA tile_shape[1]);
            cudaDeviceSynchronize();)
#undef COMMA


    if ( do_cpu ) {
        if ( cudaMemcpy(Cref.data, dCg.data, Cref.M*Cref.N*sizeof(float),
                    cudaMemcpyDeviceToHost) != cudaSuccess ) {
            fprintf(stderr, "cudaMemcpy failed!\n");
            return -1;
        }
        check_results(hC.data, Cref.data, hC.M*hC.N);

        if ( cudaMemcpy(Cref.data, dCs.data, Cref.M*Cref.N*sizeof(float),
                    cudaMemcpyDeviceToHost) != cudaSuccess ) {
            fprintf(stderr, "cudaMemcpy failed!\n");
            return -1;
        }
        check_results(hC.data, Cref.data, hC.M*hC.N);
    }

    return 0;
}



/***************************************************************************** 
 * MAIN                                                                      *
 *****************************************************************************/


const char* HEADER = "+=========+\n"
                     "| mat_mul |\n"
                     "+=========+\n"
                     "C = A * B\n";
const char* START_COMP = "Performing computations (cpu timing runs: %d, gpu timing runs: %d)...\n";
const char* CONFIG = "A shape: (%d, %d), B shape: (%d, %d), tile shape: (%d, %d), do cpu: %s\n";
const char* END_COMP = "Done.\n\n";
const char* RESULT_HEADER = "Timing Results:\n"
                            "---------------\n"
                            "(times in ms, ratios gpu/cpu)\n";
const char* RESULT_START = "A shape: %d x %d, B shape: %d x %d\n"
                           "+-------++----------+--------+--------++--------+--------+-------+\n"
                           "| tile  || CPU      | global | shared || g/cpu  | s/cpu  | s/g   | \n"
                           "+-------++----------+--------+--------++--------+--------+-------+\n";
const char* RESULT_ROW = "| %2dx%2d || %8.2f | %6.2f | %6.2f || %6.3f | %6.3f | %5.3f |\n";
const char* RESULT_END = "+-------++----------+--------+--------++--------+--------+-------+\n";



int main()
{
    srand(time(NULL));

    const int tile_shape_count = 4;
    const int matrix_shape_count = 5;
    const size_t tile_shapes[tile_shape_count][2] = {
        {  8,  8 },
        {  8, 16 },
        { 16, 16 },
        { 16, 32 },
    };
    // matrix_shape_ij, N_jk
    const size_t matrix_shapes[matrix_shape_count][3] = {
        {  300,  100,  500 },
        {  600,  200, 1000 },
        { 1200,  400, 2000 },
        { 2400,  800, 4000 },
        { 3000, 1000, 5000 },
    };

    const size_t timing_runs[2] = { 7, 1000 };
    timing_result_t timing_results[3*matrix_shape_count*tile_shape_count];
    matrix_t test_matrices_h[4];
    matrix_t test_matrices_d[4];


    puts(HEADER);
    for ( int j = 0; j < tile_shape_count; ++j ) {
        check_device_properties(tile_shapes[j]);
    }

    printf(START_COMP, timing_runs[0], timing_runs[1]);
    for ( int i = 0; i < matrix_shape_count; ++i ) {
        allocate_and_init_test_matrices(
                test_matrices_h, test_matrices_d, matrix_shapes[i]);

        for ( int j = 0; j < tile_shape_count; ++j ) {
            bool do_cpu = j == 0 && i < 3;
            printf(CONFIG, test_matrices_h[0].M, test_matrices_h[0].N,
                           test_matrices_h[1].M, test_matrices_h[1].N,
                           tile_shapes[j][0], tile_shapes[j][1],
                           do_cpu ? "true" : "false");

            perform_matmul_test(test_matrices_h, test_matrices_d, tile_shapes[j],
                    &timing_results[3*(i*matrix_shape_count+j)], timing_runs,
                    do_cpu);
        }
        free_test_matrices(test_matrices_h, test_matrices_d);
    }
    puts(END_COMP);


    puts(RESULT_HEADER);
    for ( int i = 0; i < matrix_shape_count; ++i ) {
        printf(RESULT_START, matrix_shapes[i][0], matrix_shapes[i][1],
                             matrix_shapes[i][1], matrix_shapes[i][2]);

        for ( int j = 0; j < tile_shape_count; ++j ) {
            bool do_cpu = j == 0 && i < 3;
            float global_time = timing_results[3*(i*matrix_shape_count+j)+1].t_mean;
            float shared_time = timing_results[3*(i*matrix_shape_count+j)+2].t_mean;
            float sg_ratio = shared_time / global_time;
            float cpu_time, gc_ratio, sc_ratio;
            if ( do_cpu ) {
                cpu_time = timing_results[3*(i*matrix_shape_count)].t_mean;
                gc_ratio = global_time / cpu_time;
                sc_ratio = shared_time / cpu_time;
            }
            else if ( i < 3 ) {
                cpu_time = timing_results[3*(i*matrix_shape_count)].t_mean;
                gc_ratio = global_time / cpu_time;
                sc_ratio = shared_time / cpu_time;
            }
            else {
                cpu_time = -1;
                gc_ratio = -1;
                sc_ratio = -1;
            }
            printf(RESULT_ROW, tile_shapes[j][0], tile_shapes[j][1],
                               cpu_time, global_time, shared_time,
                               gc_ratio, sc_ratio, sg_ratio);
        }
        puts(RESULT_END);
    }


    return EXIT_SUCCESS;
}


/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
