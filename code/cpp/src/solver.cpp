#include <Solver/solver.h>

#include <functional>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <type_traits>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <ATen/Dispatch.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDADataType.h>
#include <ATen/cuda/CUDASparseDescriptors.h>
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/types.h>

namespace solver
{

template<typename T>
struct dependent_false : std::false_type
{
};

template<typename T>
inline constexpr bool dependent_false_v = dependent_false<T>::value;

template<typename scalar_t>
cublasStatus_t
cublasGaxpy(cublasHandle_t handle, int n, const scalar_t* alpha, const scalar_t* x, int incx, scalar_t* y, int incy)
{
    if constexpr(std::is_same_v<scalar_t, float>)
    {
        return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
    }
    else if constexpr(std::is_same_v<scalar_t, double>)
    {
        return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
    }
    else
    {
        static_assert(dependent_false_v<scalar_t>, "Unknown type");
    }
}

template<typename scalar_t>
cublasStatus_t
cublasGdot(cublasHandle_t handle, int n, const scalar_t* x, int incx, scalar_t* y, int incy, scalar_t* result)
{
    if constexpr(std::is_same_v<scalar_t, float>)
    {
        return cublasSdot(handle, n, x, incx, y, incy, result);
    }
    else if constexpr(std::is_same_v<scalar_t, double>)
    {
        return cublasDdot(handle, n, x, incx, y, incy, result);
    }
    else
    {
        static_assert(dependent_false_v<scalar_t>, "Unknown type");
    }
}

template<typename scalar_t>
cublasStatus_t cublasGscal(cublasHandle_t handle, int n, const scalar_t* alpha, scalar_t* x, int incx)
{
    if constexpr(std::is_same_v<scalar_t, float>)
    {
        return cublasSscal(handle, n, alpha, x, incx);
    }
    else if constexpr(std::is_same_v<scalar_t, double>)
    {
        return cublasDscal(handle, n, alpha, x, incx);
    }
    else
    {
        static_assert(dependent_false_v<scalar_t>, "Unknown type");
    }
}

// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/conjugateGradient/main.cpp
std::tuple<torch::Tensor, int> cg_cuda(const torch::Tensor& A,
                                       const torch::Tensor& b,
                                       const std::optional<torch::Tensor>& x0,
                                       const std::optional<int>& maxiter,
                                       const std::optional<torch::Tensor>& M,
                                       const std::optional<std::function<void(const torch::Tensor&)>>& callback,
                                       const float rtol,
                                       const float atol)
{
    TORCH_CHECK_EQ(A.is_sparse() || A.is_sparse_csr(), true);    // Sparsity check is a bit fragile
    TORCH_CHECK_EQ(A.device(), b.device());
    TORCH_CHECK_EQ(A.dtype(), b.dtype());
    TORCH_CHECK_EQ(A.device().is_cuda(), true);
    TORCH_CHECK_EQ(A.size(0), A.size(1));
    TORCH_CHECK_EQ(A.size(0), b.size(0));
    if(x0)
    {
        TORCH_CHECK_EQ(A.device(), x0->device());
        TORCH_CHECK_EQ(A.dtype(), x0->dtype());
        TORCH_CHECK_EQ(A.size(0), x0->size(0));
    }
    // Preconditioning not implemented
    if(M)
    {
        throw std::invalid_argument("Preconditioning is not implemented.");
    }

    torch::Tensor x;
    int k          = 0;
    bool converged = false;

    AT_DISPATCH_FLOATING_TYPES(
        A.scalar_type(),
        "cg",
        [&]()
        {
            auto A_csr        = A.to_sparse_csr();
            auto b_contiguous = b.contiguous();

            const auto dtype_scalar = torch::TensorOptions {}.dtype(A.dtype()).device(A.device());
            const auto N            = static_cast<int>(A.size(0));

            x          = x0 ? x0->contiguous().clone() : torch::zeros({N}, dtype_scalar);
            auto raw_x = x.data_ptr<scalar_t>();
            auto raw_b = b_contiguous.data_ptr<scalar_t>();

            auto r         = torch::empty({N}, dtype_scalar);
            auto old_r     = torch::empty({N}, dtype_scalar);
            auto p         = torch::empty({N}, dtype_scalar);
            auto Ax        = torch::empty({N}, dtype_scalar);
            auto raw_r     = r.data_ptr<scalar_t>();
            auto raw_old_r = old_r.data_ptr<scalar_t>();
            auto raw_p     = p.data_ptr<scalar_t>();
            auto raw_Ax    = Ax.data_ptr<scalar_t>();

            /* Create CUBLAS context */
            auto cublas_handle = torch::cuda::getCurrentCUDABlasHandle();

            /* Create CUSPARSE context */
            auto cusparse_handle = torch::cuda::getCurrentCUDASparseHandle();

            /* Wrap raw data into cuSPARSE generic API objects */
            auto mat_A = torch::cuda::sparse::CuSparseSpMatCsrDescriptor {A_csr};

            auto vec_x  = torch::cuda::sparse::CuSparseDnVecDescriptor {x};
            auto vec_p  = torch::cuda::sparse::CuSparseDnVecDescriptor {p};
            auto vec_Ax = torch::cuda::sparse::CuSparseDnVecDescriptor {Ax};

            /* Initialize problem data */
            AT_CUDA_CHECK(cudaMemcpy(raw_r, raw_b, N * sizeof(scalar_t), cudaMemcpyDeviceToDevice));

            auto one       = scalar_t {1.0};
            auto minus_one = scalar_t {-1.0};
            auto zero      = scalar_t {0.0};

            /* Allocate workspace for cuSPARSE */
            auto buffer_size = size_t {0};
            TORCH_CUDASPARSE_CHECK(cusparseSpMV_bufferSize(cusparse_handle,
                                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                           &one,
                                                           mat_A.descriptor(),
                                                           vec_x.descriptor(),
                                                           &zero,
                                                           vec_Ax.descriptor(),
                                                           torch::cuda::getCudaDataType<scalar_t>(),
                                                           CUSPARSE_SPMV_ALG_DEFAULT,
                                                           &buffer_size));
            auto buffer     = torch::empty({static_cast<int32_t>(std::ceil(static_cast<double>(buffer_size) /
                                                                       static_cast<double>(sizeof(scalar_t))))},
                                       dtype_scalar);
            auto raw_buffer = buffer.data_ptr();

            /* Begin CG */
            scalar_t alpha, beta, minus_alpha, old_r_squared, r_squared, r_dot_old_r, pAp;

            TORCH_CUDASPARSE_CHECK(cusparseSpMV(cusparse_handle,
                                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                &one,
                                                mat_A.descriptor(),
                                                vec_x.descriptor(),
                                                &zero,
                                                vec_Ax.descriptor(),
                                                torch::cuda::getCudaDataType<scalar_t>(),
                                                CUSPARSE_SPMV_ALG_DEFAULT,
                                                raw_buffer));

            TORCH_CUDABLAS_CHECK(cublasGaxpy<scalar_t>(cublas_handle, N, &minus_one, raw_Ax, 1, raw_r, 1));
            TORCH_CUDABLAS_CHECK(cublasGdot<scalar_t>(cublas_handle, N, raw_r, 1, raw_r, 1, &r_squared));
            old_r_squared = r_squared;

            const auto tol   = std::max<scalar_t>(scalar_t {rtol} * std::sqrt(r_squared), scalar_t {atol});
            const auto max_k = maxiter ? *maxiter : 10 * N;
            for(; k < max_k && r_squared >= tol * tol; ++k)
            {
                if(k > 0)
                {
                    // Polak-Ribiere
                    TORCH_CUDABLAS_CHECK(cublasGdot<scalar_t>(cublas_handle, N, raw_r, 1, raw_old_r, 1, &r_dot_old_r));
                    beta = std::max(scalar_t {0}, (r_squared - r_dot_old_r) / old_r_squared);

                    TORCH_CUDABLAS_CHECK(cublasGscal<scalar_t>(cublas_handle, N, &beta, raw_p, 1));
                    TORCH_CUDABLAS_CHECK(cublasGaxpy<scalar_t>(cublas_handle, N, &one, raw_r, 1, raw_p, 1));
                }
                else
                {
                    AT_CUDA_CHECK(cudaMemcpy(raw_p, raw_r, N * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
                }

                // Update x
                TORCH_CUDASPARSE_CHECK(cusparseSpMV(cusparse_handle,
                                                    CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                    &one,
                                                    mat_A.descriptor(),
                                                    vec_p.descriptor(),
                                                    &zero,
                                                    vec_Ax.descriptor(),
                                                    torch::cuda::getCudaDataType<scalar_t>(),
                                                    CUSPARSE_SPMV_ALG_DEFAULT,
                                                    raw_buffer));
                TORCH_CUDABLAS_CHECK(cublasGdot<scalar_t>(cublas_handle, N, raw_p, 1, raw_Ax, 1, &pAp));
                alpha = r_squared / pAp;

                TORCH_CUDABLAS_CHECK(cublasGaxpy<scalar_t>(cublas_handle, N, &alpha, raw_p, 1, raw_x, 1));

                // Update r
                AT_CUDA_CHECK(cudaMemcpy(raw_old_r, raw_r, N * sizeof(scalar_t), cudaMemcpyDeviceToDevice));

                const auto compute_exact_residual = true;    // Always use slower accurate version
                if(compute_exact_residual)
                {
                    // This is unoptimized but works
                    TORCH_CUDASPARSE_CHECK(cusparseSpMV(cusparse_handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &one,
                                                        mat_A.descriptor(),
                                                        vec_x.descriptor(),
                                                        &zero,
                                                        vec_Ax.descriptor(),
                                                        torch::cuda::getCudaDataType<scalar_t>(),
                                                        CUSPARSE_SPMV_ALG_DEFAULT,
                                                        raw_buffer));
                    AT_CUDA_CHECK(cudaMemcpy(raw_r, raw_b, N * sizeof(scalar_t), cudaMemcpyDeviceToDevice));
                    TORCH_CUDABLAS_CHECK(cublasGaxpy<scalar_t>(cublas_handle, N, &minus_one, raw_Ax, 1, raw_r, 1));
                }
                else
                {
                    minus_alpha = -alpha;
                    TORCH_CUDABLAS_CHECK(cublasGaxpy<scalar_t>(cublas_handle, N, &minus_alpha, raw_Ax, 1, raw_r, 1));
                }

                old_r_squared = r_squared;
                TORCH_CUDABLAS_CHECK(cublasGdot<scalar_t>(cublas_handle, N, raw_r, 1, raw_r, 1, &r_squared));

                if(callback)
                {
                    (*callback)(x);
                }

                // printf("CG: iteration = %3d, residual = %e\n", k, std::sqrt(r_squared));

                // if(r_squared > old_r_squared)
                // {
                //     printf("CG: error increased, stopped iterating\n");
                //     // break;
                // }
            }

            converged = r_squared < tol * tol;
        });

    return {x, converged ? 0 : k};
}
}    // namespace solver
