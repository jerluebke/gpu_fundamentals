#ifdef __clang__
cudaError_t cudaConfigureCall(dim3, dim3, size_t=0, cudaStream_t=0);
#endif

#include <assert.h>
#include <stdio.h>


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


/*****************************************************************************/


int main()
{
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
#endif


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
}


/* vim: set tw=79 ts=4 sw=4 et ic ai : */
