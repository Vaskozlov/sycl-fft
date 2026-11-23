#pragma once

#include <cstddef>

namespace syclfft::detail
{
    void require_cuda_ptx_compiler();

    int compile_cuda_ptx(
        void *diagnostics, const char *source, unsigned major, unsigned minor, int disable_fma,
        char **ptx, std::size_t *ptx_size) noexcept;
} // namespace syclfft::detail
