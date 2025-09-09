import torch
import typing

import cpp_solver


class COOSolve(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A                  : torch.Tensor,
        b                  : torch.Tensor,
        initial_guess      : torch.Tensor,
        precond            : typing.Callable[[torch.Tensor], torch.Tensor],
        max_iterations     : torch.Tensor = torch.tensor([1000], device = "cuda"),
        relative_tolerance : torch.Tensor = torch.tensor([1e-3], device = "cuda"),
        absolute_tolerance : torch.Tensor = torch.tensor([0.0], device = "cuda"),
    ):
        data_type = b.dtype
        A = A.to(torch.float64)
        b = b.to(torch.float64)
        initial_guess = initial_guess.to(torch.float64)

        result, _ = cpp_solver.Solver.cg_cuda(A, b, initial_guess, max_iterations[0], precond, None, relative_tolerance[0], absolute_tolerance[0])
        result = result.to(data_type)

        ctx.save_for_backward(A, result, max_iterations, relative_tolerance, absolute_tolerance)

        return result

    @staticmethod
    def backward(ctx, gradient):
        A, result, max_iterations, relative_tolerance, absolute_tolerance = ctx.saved_tensors
        data_type = result.dtype
        row_indices = A.indices()[0]
        col_indices = A.indices()[1]

        A_T = torch.sparse_coo_tensor(torch.flip(A.indices(), (0,)), A.values(), dtype = A.dtype, device=A.device)
        A_T = A_T.to_sparse_csr()
        b = gradient.to(torch.float64).contiguous() # gradient is not contiguous by default => error when using data_ptr
        initial_guess = torch.zeros_like(b)

        # compute gradient wrt. b
        # solve A^T x' = dL/dx  =>  x' = dL/db
        gradient_b, _ = cpp_solver.Solver.cg_cuda(A_T, b, initial_guess, max_iterations[0], None, None, relative_tolerance[0], absolute_tolerance[[0]])

        # compute sparse gradient wrt non-zero entries in A
        gradient_b_selected = gradient_b[row_indices]
        result_selected = result[col_indices]
        gradient_A_values = - gradient_b_selected * result_selected
        gradient_A = torch.sparse_coo_tensor(A.indices(), gradient_A_values, A.shape)

        # compute dense gradient
        # gradient_A = - gradient_b.unsqueeze(1) * result.unsqueeze(0)

        gradient_A = gradient_A.to(data_type)
        gradient_b = gradient_b.to(data_type)

        return gradient_A, gradient_b, None, None, None, None, None