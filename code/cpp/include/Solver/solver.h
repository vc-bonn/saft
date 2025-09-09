#pragma once
#include <torch/types.h>
#include <functional>
#include <tuple>
#include <optional>


namespace solver
{
std::tuple<torch::Tensor, int>
cg_cuda(const torch::Tensor& A,
        const torch::Tensor& b,
        const std::optional<torch::Tensor>& x0                                   = std::nullopt,
        const std::optional<int>& maxiter                                        = std::nullopt,
        const std::optional<torch::Tensor>& M                                    = std::nullopt,
        const std::optional<std::function<void(const torch::Tensor&)>>& callback = std::nullopt,
        const float rtol                                                         = 1e-5f,
        const float atol                                                         = 0.0f);
}    // namespace solver