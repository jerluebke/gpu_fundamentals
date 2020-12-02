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

bool check_results(float* host_ref, float* gpu_ref, const size_t n,
        bool print_status = true)
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

__host__ __device__
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
            Nds[ty][tx] = 0.0;

        __syncthreads();

        for ( size_t k = 0; k < TILE_WIDTH; ++k )
            p_value += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
    }

    if ( row < width && col < width )
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



/*****************************************************************************/



void check_device_properties(const size_t tile_shape[2]);

int allocate_and_init_test_matrices(
        matrix_t test_matrices_h[4],
        matrix_t test_matrices_d[4],
        const size_t matrix_shape[3]);

void free_test_matrices(matrix_t test_matrices_h[4],
        matrix_t test_matrices_d[4]);

int perform_matmul_test(
        matrix_t test_matrices_h[4],
        matrix_t test_matrices_d[4],
        const size_t tile_shape[2],
        timing_result_t timing_result[3],
        const size_t timing_runs);


int main()
{
    srand(time(NULL));

    const int tile_shape_count = 5;
    const int matrix_shape_count = 6;
    const size_t tile_shapes[tile_shape_count][2] = {
        {  8,  8 },
        {  8, 16 },
        { 16, 16 },
        // { 16, 32 },
        // { 32, 32 },
    };
    // matrix_shape_ij, N_jk
    const size_t matrix_shapes[matrix_shape_count][3] = {
        {  300,  100,  500 },
        {  600,  200, 1000 },
        { 1200,  400, 2000 },
        { 2400,  800, 4000 },
        { 3000, 1000, 5000 },
        { 4800, 1600, 8000 },
    };

    const size_t timing_runs = 1000;
    timing_result_t timing_results[3*matrix_shape_count*tile_shape_count];
    matrix_t test_matrices_h[4];
    matrix_t test_matrices_d[4];

    for ( int j = 0; j < tile_shape_count; ++j ) {
        check_device_properties(tile_shapes[j]);
    }

    for ( int i = 0; i < matrix_shape_count; ++i ) {
        allocate_and_init_test_matrices(
                test_matrices_h, test_matrices_d, matrix_shapes[i]);
        for ( int j = 0; j < tile_shape_count; ++j ) {
            perform_matmul_test(test_matrices_h, test_matrices_d, tile_shapes[j],
                    &timing_results[3*(i*matrix_shape_count+j)], timing_runs);
        }
        free_test_matrices(test_matrices_h, test_matrices_d);
    }


#if 0
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
#endif

    return EXIT_SUCCESS;
}



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
    cudaDeviceGetAttribute(&max_blocks_per_sm,
            cudaDevAttrMaxBlocksPerMultiprocessor, 0);
    cudaDeviceGetAttribute(&max_shared_memory_per_sm,
            cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);

    int block_size = tile_shape[0] * tile_shape[1];
    int number_of_blocks_by_threads = max_threads_per_sm / block_size;
    int number_of_blocks_by_shared_mem = max_shared_memory_per_sm \
        / (2*block_size*sizeof(float));

    if ( number_of_blocks_by_threads > max_blocks_per_sm )
        fprintf(stderr, "WARNING: block configuration %zux%zu allows %d blocks, "
                "but only %d are available on the multiprocessors!\n",
                tile_shape[0], tile_shape[1], number_of_blocks_by_threads,
                max_blocks_per_sm);
    if ( number_of_blocks_by_shared_mem < max_blocks_per_sm )
        fprintf(stderr, "WARNING: block configuration %zux%zu allows only %d "
                "blocks, but %d are available on the multiprocessors!\n",
                tile_shape[0], tile_shape[1], number_of_blocks_by_shared_mem,
                max_blocks_per_sm);
}


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
        const size_t tile_shape[2])
{
    extern __shared__ float smem[];
    matrix_t dAs = { tile_shape[0], tile_shape[1], smem };
    matrix_t dBs = { tile_shape[1], tile_shape[0],
                     &smem[tile_shape[0]*tile_shape[1]] };

    size_t bx = blockIdx.x, by = blockIdx.y;
    size_t tx = threadIdx.x, ty = threadIdx.y;
    size_t row = by * tile_shape[0] + ty;
    size_t col = bx * tile_shape[1] + tx;

    float sum = 0.0;

    for ( size_t m = 0; m < dA.N / dAs.N + 1; ++m ) {
        // collaborative loading of tiles into shared memory
        // matrix_* functions take care of bounds checking
        matrix_set(dAs, ty, tx, matrix_get(dA, row, m*dAs.N+tx));
        matrix_set(dBs, tx, ty, matrix_get(dB, m*dBs.M+ty, col));
        __syncthreads();

        for ( size_t k = 0; k < dAs.N; ++k )
            sum += matrix_get(dAs, ty, k) * matrix_get(dBs, k, tx);
        __syncthreads();
    }

    matrix_set(dC, row, col, sum);
}



int perform_matmul_test(
        matrix_t test_matrices_h[4],
        matrix_t test_matrices_d[4],
        const size_t tile_shape[2],
        timing_result_t timing_result[3],
        const size_t timing_runs)
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
    dim3 grid(hC.M / tile_shape[0] + 1, hC.N / tile_shape[1] + 1);

#define COMMA ,
    TIMEIT(timing_runs, (&timing_result[0]), mat_mul_cpu(hA COMMA hB COMMA hC);)
    TIMEIT(timing_runs, (&timing_result[1]),
            mat_mul_global_kernel<<<grid COMMA block>>>(dA COMMA dB COMMA dCg);
            cudaDeviceSynchronize();)
    TIMEIT(timing_runs, (&timing_result[2]),
            mat_mul_shared_kernel<<<grid COMMA block COMMA 2*tile_size>>>(
                dA COMMA dB COMMA dCs COMMA tile_shape);
            cudaDeviceSynchronize();)
#undef COMMA

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

    return 0;
}



/* vim: set ff=unix tw=79 sw=4 ts=4 et ic ai : */
