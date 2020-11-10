#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif

#include <assert.h>
#include <stdio.h>
#include <time.h>



/**
 * \brief compare arrays with size \c n elementwise by checking whether the
 * absolute value of the difference is smaller than some epsilon > 0.
 */
void check_results(float* host_ref, float* gpu_ref, const int n)
{
    double eps = 1e-8;
    for ( int i = 0; i < n; ++i ) {
        if ( abs(host_ref[i] - gpu_ref[i]) > eps ) {
            printf("arrays do not match!\n"
                   "[%d] host: %5.2f\tgpu: %5.2f\n",
                   i, host_ref[i], gpu_ref[i]);
            return;
        }
    }

    printf("arrays match.\n");
}


/**
 * \brief initialize an array with random floats in the range [0, 255]
 */
void init_data(float* ip, const int size)
{
    srand(time(NULL));
    for ( int i = 0; i < size; ++i )
        ip[i] = (float)(rand() & 0xFF);
}


/*****************************************************************************/


/**
 * \brief basic matrix type, which holds its data in linear memory and knows
 * its size.
 */
struct matrix_s
{
    size_t M, N;
    float* data;
};
typedef struct matrix_s matrix_t;


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
 */
void mat_add_cpu(const matrix_t hA, const matrix_t hB, matrix_t hC)
{
    size_t size_A = hA.M * hA.N;
    size_t size_B = hB.M * hB.N;
    size_t size_C = hC.M * hC.N;
    assert(size_A == size_B && size_B == size_C);

    for ( size_t i = 0; i < size_A; ++i )
        hC.data[i] = hA.data[i] + hB.data[i];
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
 */
void mat_add_gpu(const matrix_t hA, const matrix_t hB, matrix_t hC,
        const int block_width, const int block_height)
{
    assert(hA.M == hB.M && hB.M == hC.M);
    assert(hA.N == hB.N && hB.N == hC.N);

    matrix_t dA, dB, dC;
    new_matrix_d(&dA, hA.M, hA.N, hA.data);
    new_matrix_d(&dB, hB.M, hB.N, hB.data);
    new_matrix_d(&dC, hC.M, hC.N, hC.data);

    dim3 grid(dA.M / block_width + 1, dA.N / block_height + 1);
    dim3 block(block_width, block_height);
    mat_add_kernel<<<grid, block>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    if ( cudaMemcpy(hC.data, dC.data, dC.M*dC.N*sizeof(float),
                cudaMemcpyDeviceToHost) != cudaSuccess ) {
        fprintf(stderr, "cudaMemcpy failed!\n");
    }

    free_matrix_d(dA);
    free_matrix_d(dB);
    free_matrix_d(dC);
}


/*****************************************************************************/


/*

   cuda timer:
    #include <helper_cuda.h>
    #include <helper_timer.h>

    float t_start, t_end;
    static StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    t_start = sdkGetTimerValue(&timer);

    // do something...
    cudaDeviceSynchronize();

    sdkStopTimer(&timer);
    t_end = sdkGetTimerValue(&timer);

    // execution time in ms: t_end - t_start



    matrix sizes:
        10x10
        100x100
        1000x1000
        500x2000
        100x10000

    block sizes (for 100x10000):
        16x16
        16x32
        32x16
*/

int main()
{
}


/* vim: set tw=79 ts=4 sw=4 et ic ai : */
