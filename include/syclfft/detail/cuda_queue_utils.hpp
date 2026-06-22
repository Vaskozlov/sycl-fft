#pragma once

#include <sycl/sycl.hpp>

namespace syclfft::detail
{
    inline bool is_cuda_queue(const sycl::queue &queue)
    {
#if defined(SYCLFFT_USE_ADAPTIVECPP)
        return queue.get_device().get_backend() == sycl::backend::cuda;
#elif defined(SYCLFFT_USE_DPCPP)
        return queue.get_backend() == sycl::backend::ext_oneapi_cuda;
#else
        (void)queue;
        return false;
#endif
    }
} // namespace syclfft::detail
