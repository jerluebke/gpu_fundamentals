#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif

#include <assert.h>
#include <stdio.h>

#define BLOCK_WIDTH 16


void check_results(float* host_ref, float* gpu_ref, const int n)
{
    double eps = 1e-1;
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

void init_data(float* ip, int size)
{
    srand(time(NULL));
    for ( int i = 0; i < size; ++i )
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
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
 * \brief constructor of matrix type, allocate object and data with given size,
 * returns NULL on failure.
 */
matrix_t* new_matrix(size_t M, size_t N, float* data = NULL)
{
    matrix_t* m;
    if ( (m = (matrix_t*)malloc(sizeof(matrix_t))) == NULL ) {
        return NULL;
    }
    if ( (m->data = (float*)malloc(M*N*sizeof(float))) == NULL ) {
        free(m);
        return NULL;
    }
    m->M = M;
    m->N = N;

    if ( data != NULL )
        memcpy(m->data, data, M*N*sizeof(float));

    return m;
}


/**
 * \brief destructor of matrix type, frees all allocated memory and sets
 * pointer to NULL.
 */
void free_matrix(matrix_t* m)
{
    free(m->data);
    free(m);
    m = NULL;
}


/**
 * \brief read matrix element.
 * \param mat matrix to access.
 * \param i, j position of element.
 * \return element at (i, j).
 */
inline float matrix_get(matrix_t* mat, size_t i, size_t j)
{
    return mat->data[i*mat->N+j];
}
__device__ float matrix_get_d(matrix_t* mat, size_t i, size_t j)
{
    return mat->data[i*mat->N+j];
}

/**
 * \brief write matrix element.
 * \param mat matrix to access.
 * \param i, j position of element.
 * \param val new value of element at (i, j).
 */
inline void matrix_set(matrix_t* mat, size_t i, size_t j, float val)
{
    mat->data[i*mat->N+j] = val;
}
__device__ inline void matrix_set_d(matrix_t* mat, size_t i, size_t j, float val)
{
    mat->data[i*mat->N+j] = val;
}


/**
 * \brief pretty print matrix
 */
void matrix_print(matrix_t* mat, const char* name)
{
    int offset = strlen(name) + 4;
    printf("%s = ", name);
    for ( size_t i = 0; i < mat->M; ++i ) {
        printf("%*c", i == 0 ? 0 : offset,
                      i == 0 ? '/' : i < mat->M-1 ? '|' : '\\');
        for ( size_t j = 0; j < mat->N; ++j )
            printf(" %5.2f ", matrix_get(mat, i, j));
        printf("%c\n", i == 0 ? '\\' : i < mat->M-1 ? '|' : '/');
    }
    printf("\n");
}


/*****************************************************************************/


/**
 * \brief matrix multiplication, sequentially on CPU.
 *
 * \f[
 *      hA \in M(m, n), hB \in M(n, p), hC \in M(m, p): \quad
 *      hC_{ij} = \sum_{k=0}^{n} hA_{ik} hB_{kj}
 * \f]
 *
 * \param[in] hA, hB matrices to multiply.
 * \param[out] hC resulting matrix.
 */
void matmul_cpu(matrix_t* hA, matrix_t* hB, matrix_t* hC)
{
    assert(hA->N == hB->M);

    for ( size_t i = 0; i < hA->M; ++i ) {
        for ( size_t j = 0; j < hB->N; ++j ) {
            float sum = 0.0;
            for ( size_t k = 0; k < hA->N; ++k ) {
                sum += matrix_get(hA, i, k) * matrix_get(hB, k, j);
            }
            matrix_set(hC, i, j, sum);
        }
    }
}


// __global__ void matmul_gpu_kernel(matrix_t* dA, matrix_t* dB, matrix_t* dC)
__global__ void matmul_gpu_kernel(matrix_t dA, matrix_t dB, matrix_t dC)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if ( row < dA.M && col < dB.N ) {
        float sum = 0.0;
        for ( size_t k = 0; k < dA.N; ++k )
            sum += matrix_get_d(&dA, row, k) * matrix_get_d(&dB, k, col);
        matrix_set_d(&dC, row, col, sum);
    }
}


void matmul_gpu(matrix_t* A, matrix_t* B, matrix_t* C)
{
#define TRANSFER_SIZE(Name) \
    d##Name.M = Name->M; d##Name.N = Name->N; \
    size_t Name##size = Name->M * Name->N * sizeof(float);

    matrix_t dA, dB, dC;
    TRANSFER_SIZE(A)
    TRANSFER_SIZE(B)
    TRANSFER_SIZE(C)
    cudaMalloc((void**)&(dA.data), Asize);
    cudaMalloc((void**)&(dB.data), Bsize);
    cudaMalloc((void**)&(dC.data), Csize);
    cudaMemcpy(dA.data, A->data, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB.data, B->data, Bsize, cudaMemcpyHostToDevice);

    dim3 grid(A->M/BLOCK_WIDTH+1, B->N/BLOCK_WIDTH+1);
    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

    matmul_gpu_kernel<<<grid,block>>>(dA, dB, dC);
    cudaDeviceSynchronize();

    cudaMemcpy(C->data, dC.data, Csize, cudaMemcpyDeviceToHost);
    cudaFree(dC.data);
    cudaFree(dB.data);
    cudaFree(dA.data);

#undef TRANSFER_SIZE
}


#if 0
void OLD_matmul_gpu(matrix_t* A, matrix_t* B, matrix_t* C)
{
    matrix_t *dA, *dB, *dC;
    float *Adata, *Bdata, *Cdata;
    size_t Asize = A->M * A->N * sizeof(float);
    size_t Bsize = B->M * B->N * sizeof(float);
    size_t Csize = C->M * C->N * sizeof(float);
    cudaMalloc((void**)&dA, sizeof(matrix_t));
    cudaMalloc((void**)&dB, sizeof(matrix_t));
    cudaMalloc((void**)&dC, sizeof(matrix_t));
    cudaMalloc((void**)&Adata, Asize);
    cudaMalloc((void**)&Bdata, Bsize);
    cudaMalloc((void**)&Cdata, Csize);
    cudaMemcpy(&dA->data, &Adata, sizeof(dA->data), cudaMemcpyHostToDevice);
    cudaMemcpy(&dB->data, &Bdata, sizeof(dB->data), cudaMemcpyHostToDevice);
    cudaMemcpy(&dC->data, &Cdata, sizeof(dC->data), cudaMemcpyHostToDevice);
    dA->M = A->M; dA->N = A->N;
    dB->M = B->M; dB->N = B->N;
    dC->M = C->M; dC->N = C->N;
    cudaMemcpy(dA->data, A->data, Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(dB->data, B->data, Bsize, cudaMemcpyHostToDevice);


    dim3 grid(A->M/BLOCK_WIDTH+1, B->N/BLOCK_WIDTH+1);
    dim3 block(BLOCK_WIDTH, BLOCK_WIDTH);

    // matmul_gpu_kernel<<<grid,block>>>(dA, dB, dC);
    cudaDeviceSynchronize();


    cudaMemcpy(C->data, dC->data, Csize, cudaMemcpyDeviceToHost);
    cudaFree(dC);
    cudaFree(dB);
    cudaFree(dA);
}
#endif


/*****************************************************************************/


int main()
{
    size_t Ma = 256;
    size_t Na = 256;
    size_t Mb = 256;
    size_t Nb = 256;

    matrix_t *hA, *hB, *hC, *dC;
    hA = new_matrix(Ma, Na);
    hB = new_matrix(Mb, Nb);
    hC = new_matrix(Ma, Nb);
    dC = new_matrix(Ma, Nb);
    init_data(hA->data, Ma*Na);
    init_data(hB->data, Mb*Nb);

    matmul_cpu(hA, hB, hC);
    matmul_gpu(hA, hB, dC);

    check_results(hC->data, dC->data, Ma*Nb);


#if 0
    matrix_t *sx, *sz, *isy;
    sx = new_matrix(2, 2);
    sz = new_matrix(2, 2);
    isy = new_matrix(2, 2);

    matrix_set(sx, 0, 1, 1.0);
    matrix_set(sx, 1, 0, 1.0);
    matrix_set(sz, 0, 0, 1.0);
    matrix_set(sz, 1, 1, -1.0);

    matmul_cpu(sx, sz, isy);
    matrix_print(sx, "sigma_x");
    matrix_print(sz, "sigma_z");
    matrix_print(isy, "i*sigma_y");

    free_matrix(sx);
    free_matrix(sz);
    free_matrix(isy);


    matrix_t *A, *B, *C;
    float Adata[6] = { 3, 2, 1, 1, 0, 2 };
    float Bdata[6] = { 1, 2, 0, 1, 4, 0 };
    A = new_matrix(2, 3, Adata);
    B = new_matrix(3, 2, Bdata);
    C = new_matrix(2, 2);

    matmul_cpu(A, B, C);
    matrix_print(A, "A");
    matrix_print(B, "B");
    matrix_print(C, "C");

    free_matrix(A);
    free_matrix(B);
    free_matrix(C);
#endif
}


/* vim: set tw=79 ts=4 sw=4 et ic ai : */
