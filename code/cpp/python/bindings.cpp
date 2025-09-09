#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <pybind11/pybind11.h>
#include <torch/python.h>
#include <Solver/solver.h>


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    auto m_solver = m.def_submodule("Solver");
    m_solver.def("cg_cuda", &solver::cg_cuda);
}