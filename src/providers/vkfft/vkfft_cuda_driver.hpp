#pragma once

#include <cuda.h>
#include <dlfcn.h>
#include <memory>

#define SYCLFFT_VKFFT_CUDA_FUNCTIONS(F)                                                            \
    F(cuCtxGetCurrent)                                                                             \
    F(cuCtxGetDevice)                                                                              \
    F(cuCtxPopCurrent)                                                                             \
    F(cuCtxPushCurrent)                                                                            \
    F(cuCtxSynchronize)                                                                            \
    F(cuDeviceGetAttribute)                                                                        \
    F(cuEventCreate)                                                                               \
    F(cuEventDestroy)                                                                              \
    F(cuEventRecord)                                                                               \
    F(cuEventSynchronize)                                                                          \
    F(cuFuncSetAttribute)                                                                          \
    F(cuGetErrorString)                                                                            \
    F(cuLaunchKernel)                                                                              \
    F(cuMemAlloc)                                                                                  \
    F(cuMemFree)                                                                                   \
    F(cuMemcpyDtoH)                                                                                \
    F(cuMemcpyHtoD)                                                                                \
    F(cuModuleGetFunction)                                                                         \
    F(cuModuleLoadDataEx)                                                                          \
    F(cuModuleUnload)                                                                              \
    F(cuStreamGetCtx)                                                                              \
    F(cuStreamSynchronize)

namespace syclfft::detail::vkfft_cuda
{
    struct driver_api
    {
#define SYCLFFT_VKFFT_CUDA_DECLARE(name) decltype(&::name) fn_##name{};
        SYCLFFT_VKFFT_CUDA_FUNCTIONS(SYCLFFT_VKFFT_CUDA_DECLARE)
#undef SYCLFFT_VKFFT_CUDA_DECLARE
    };

    inline driver_api driver;

    inline const char *load_driver(driver_api &api, bool retain)
    {
        auto close_driver = [](void *handle) { dlclose(handle); };
        std::unique_ptr<void, decltype(close_driver)> library{
            dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL), close_driver};
        if (!library) {
            return "Unable to load libcuda.so.1";
        }
#define SYCLFFT_VKFFT_CUDA_STRING_IMPL(name) #name
#define SYCLFFT_VKFFT_CUDA_STRING(name) SYCLFFT_VKFFT_CUDA_STRING_IMPL(name)
#define SYCLFFT_VKFFT_CUDA_LOAD(name)                                                              \
    api.fn_##name = reinterpret_cast<decltype(api.fn_##name)>(                                     \
        dlsym(library.get(), SYCLFFT_VKFFT_CUDA_STRING(name)));                                    \
    if (!api.fn_##name) {                                                                          \
        return "CUDA driver symbol unavailable: " SYCLFFT_VKFFT_CUDA_STRING(name);                 \
    }
        SYCLFFT_VKFFT_CUDA_FUNCTIONS(SYCLFFT_VKFFT_CUDA_LOAD)
#undef SYCLFFT_VKFFT_CUDA_LOAD
#undef SYCLFFT_VKFFT_CUDA_STRING
#undef SYCLFFT_VKFFT_CUDA_STRING_IMPL
        if (retain) {
            // A plan may survive into static destruction of the host application.
            (void)library.release();
        }
        return nullptr;
    }

    inline const char *probe_driver()
    {
        driver_api probe;
        return load_driver(probe, false);
    }

    inline const char *driver_error()
    {
        static const auto * const error = load_driver(driver, true);
        return error;
    }
} // namespace syclfft::detail::vkfft_cuda

#undef SYCLFFT_VKFFT_CUDA_FUNCTIONS

// Redirect calls only in the translation unit that includes the VkFFT CUDA backend.
#undef cuCtxPopCurrent
#undef cuCtxPushCurrent
#undef cuEventDestroy
#undef cuEventRecord
#undef cuLaunchKernel
#undef cuMemAlloc
#undef cuMemFree
#undef cuMemcpyDtoH
#undef cuMemcpyHtoD
#undef cuStreamSynchronize
#define cuCtxGetCurrent ::syclfft::detail::vkfft_cuda::driver.fn_cuCtxGetCurrent
#define cuCtxGetDevice ::syclfft::detail::vkfft_cuda::driver.fn_cuCtxGetDevice
#define cuCtxPopCurrent ::syclfft::detail::vkfft_cuda::driver.fn_cuCtxPopCurrent
#define cuCtxPushCurrent ::syclfft::detail::vkfft_cuda::driver.fn_cuCtxPushCurrent
#define cuCtxSynchronize ::syclfft::detail::vkfft_cuda::driver.fn_cuCtxSynchronize
#define cuDeviceGetAttribute ::syclfft::detail::vkfft_cuda::driver.fn_cuDeviceGetAttribute
#define cuEventCreate ::syclfft::detail::vkfft_cuda::driver.fn_cuEventCreate
#define cuEventDestroy ::syclfft::detail::vkfft_cuda::driver.fn_cuEventDestroy
#define cuEventRecord ::syclfft::detail::vkfft_cuda::driver.fn_cuEventRecord
#define cuEventSynchronize ::syclfft::detail::vkfft_cuda::driver.fn_cuEventSynchronize
#define cuFuncSetAttribute ::syclfft::detail::vkfft_cuda::driver.fn_cuFuncSetAttribute
#define cuGetErrorString ::syclfft::detail::vkfft_cuda::driver.fn_cuGetErrorString
#define cuLaunchKernel ::syclfft::detail::vkfft_cuda::driver.fn_cuLaunchKernel
#define cuMemAlloc ::syclfft::detail::vkfft_cuda::driver.fn_cuMemAlloc
#define cuMemFree ::syclfft::detail::vkfft_cuda::driver.fn_cuMemFree
#define cuMemcpyDtoH ::syclfft::detail::vkfft_cuda::driver.fn_cuMemcpyDtoH
#define cuMemcpyHtoD ::syclfft::detail::vkfft_cuda::driver.fn_cuMemcpyHtoD
#define cuModuleGetFunction ::syclfft::detail::vkfft_cuda::driver.fn_cuModuleGetFunction
#define cuModuleLoadDataEx ::syclfft::detail::vkfft_cuda::driver.fn_cuModuleLoadDataEx
#define cuModuleUnload ::syclfft::detail::vkfft_cuda::driver.fn_cuModuleUnload
#define cuStreamGetCtx ::syclfft::detail::vkfft_cuda::driver.fn_cuStreamGetCtx
#define cuStreamSynchronize ::syclfft::detail::vkfft_cuda::driver.fn_cuStreamSynchronize
