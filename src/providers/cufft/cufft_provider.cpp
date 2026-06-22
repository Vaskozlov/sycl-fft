#include <algorithm>
#include <array>
#include <cstring>
#include <cuda_runtime_api.h>
#include <cufft.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <syclfft/detail/cufft_provider_abi.hpp>
#include <vector>

namespace
{

    void set_error(char *output, std::size_t capacity, const std::string &message)
    {
        if (!output || capacity == 0) {
            return;
        }
        const auto count = std::min(capacity - 1, message.size());
        std::memcpy(output, message.data(), count);
        output[count] = '\0';
    }

    const char *cufft_error(cufftResult result)
    {
        switch (result) {
        case CUFFT_SUCCESS:
            return "success";
        case CUFFT_INVALID_PLAN:
            return "invalid plan";
        case CUFFT_ALLOC_FAILED:
            return "allocation failed";
        case CUFFT_INVALID_TYPE:
            return "invalid type";
        case CUFFT_INVALID_VALUE:
            return "invalid value";
        case CUFFT_INTERNAL_ERROR:
            return "internal error";
        case CUFFT_EXEC_FAILED:
            return "execution failed";
        case CUFFT_SETUP_FAILED:
            return "setup failed";
        case CUFFT_INVALID_SIZE:
            return "invalid size";
        default:
            return "unknown cuFFT error";
        }
    }

    void check(cufftResult result, const char *operation)
    {
        if (result != CUFFT_SUCCESS) {
            throw std::runtime_error(std::string(operation) + ": " + cufft_error(result));
        }
    }

    bool is_cuda(const sycl::queue &queue)
    {
#if defined(SYCLFFT_USE_ADAPTIVECPP)
        return queue.get_device().get_backend() == sycl::backend::cuda;
#else
        return queue.get_backend() == sycl::backend::ext_oneapi_cuda;
#endif
    }

    int native_device_id(const sycl::queue &queue)
    {
#if defined(SYCLFFT_USE_ADAPTIVECPP)
        return sycl::get_native<sycl::backend::cuda>(queue.get_device());
#else
        return static_cast<int>(
            sycl::get_native<sycl::backend::ext_oneapi_cuda>(queue.get_device()));
#endif
    }

    class cufft_plan_impl
    {
    public:
        cufft_plan_impl(sycl::queue queue, const syclfft::detail::cufft_plan_config_v1 &config)
          : queue_(std::move(queue))
          , direction_(config.transform_direction)
          , double_precision_(config.double_precision)
        {
            if (!is_cuda(queue_)) {
                throw std::runtime_error("queue is not CUDA-backed");
            }
            if (config.rank == 0 || config.rank > 3 || config.batch_count == 0) {
                throw std::runtime_error("invalid cuFFT rank or batch count");
            }
            std::array<int, 3> lengths{};
            for (std::uint32_t index = 0; index < config.rank; ++index) {
                if (config.lengths[index] == 0
                    || config.lengths[index]
                           > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("cuFFT dimensions must fit in int");
                }
                lengths[index] = static_cast<int>(config.lengths[index]);
            }
            if (config.batch_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("cuFFT batch count must fit in int");
            }
            const auto cuda_result = cudaSetDevice(native_device_id(queue_));
            if (cuda_result != cudaSuccess) {
                throw std::runtime_error(
                    std::string("cudaSetDevice: ") + cudaGetErrorString(cuda_result));
            }
            check(
                cufftPlanMany(
                    &handle_,
                    static_cast<int>(config.rank),
                    lengths.data(),
                    nullptr,
                    1,
                    0,
                    nullptr,
                    1,
                    0,
                    double_precision_ ? CUFFT_Z2Z : CUFFT_C2C,
                    static_cast<int>(config.batch_count)),
                "cufftPlanMany");
        }

        ~cufft_plan_impl()
        {
            if (handle_) {
                cufftDestroy(handle_);
            }
        }

        sycl::event
            execute(const void *input, void *output, std::span<const sycl::event> dependencies)
        {
            const int fft_direction =
                direction_ == syclfft::direction::forward ? CUFFT_FORWARD : CUFFT_INVERSE;
#if defined(SYCLFFT_USE_ADAPTIVECPP)
            std::vector<sycl::event> deps(dependencies.begin(), dependencies.end());
            return queue_.AdaptiveCpp_enqueue_custom_operation(
                [=, this](sycl::interop_handle &handle) {
                auto stream = handle.get_native_queue<sycl::backend::cuda>();
                check(cufftSetStream(handle_, stream), "cufftSetStream");
                if (double_precision_) {
                    check(
                        cufftExecZ2Z(
                            handle_,
                            reinterpret_cast<cufftDoubleComplex *>(const_cast<void *>(input)),
                            reinterpret_cast<cufftDoubleComplex *>(output),
                            fft_direction),
                        "cufftExecZ2Z");
                } else {
                    check(
                        cufftExecC2C(
                            handle_,
                            reinterpret_cast<cufftComplex *>(const_cast<void *>(input)),
                            reinterpret_cast<cufftComplex *>(output),
                            fft_direction),
                        "cufftExecC2C");
                }
            }, deps);
#else
            std::vector<sycl::event> deps(dependencies.begin(), dependencies.end());
            return queue_.submit([&](sycl::handler &command_group) {
                command_group.depends_on(deps);
                command_group.ext_codeplay_enqueue_native_command(
                    [=, this](sycl::interop_handle handle) {
                    auto stream = handle.get_native_queue<sycl::backend::ext_oneapi_cuda>();
                    check(cufftSetStream(handle_, stream), "cufftSetStream");
                    if (double_precision_) {
                        check(
                            cufftExecZ2Z(
                                handle_,
                                reinterpret_cast<cufftDoubleComplex *>(const_cast<void *>(input)),
                                reinterpret_cast<cufftDoubleComplex *>(output),
                                fft_direction),
                            "cufftExecZ2Z");
                    } else {
                        check(
                            cufftExecC2C(
                                handle_,
                                reinterpret_cast<cufftComplex *>(const_cast<void *>(input)),
                                reinterpret_cast<cufftComplex *>(output),
                                fft_direction),
                            "cufftExecC2C");
                    }
                });
            });
#endif
        }

    private:
        sycl::queue queue_;
        cufftHandle handle_{};
        syclfft::direction direction_{syclfft::direction::forward};
        bool double_precision_{};
    };

    bool supports(const sycl::queue &queue, char *error, std::size_t capacity)
    {
        if (!is_cuda(queue)) {
            set_error(error, capacity, "queue is not CUDA-backed");
            return false;
        }
        const auto result = cudaSetDevice(native_device_id(queue));
        if (result != cudaSuccess) {
            set_error(error, capacity, cudaGetErrorString(result));
            return false;
        }
        return true;
    }

    void *create(
        sycl::queue &queue, const syclfft::detail::cufft_plan_config_v1 &config, char *error,
        std::size_t capacity)
    {
        try {
            if (config.abi_version != SYCLFFT_CUFFT_PROVIDER_ABI_VERSION) {
                throw std::runtime_error("cuFFT provider ABI mismatch");
            }
            return new cufft_plan_impl(queue, config);
        } catch (const std::exception &ex) {
            set_error(error, capacity, ex.what());
            return nullptr;
        }
    }

    void destroy(void *plan)
    {
        delete static_cast<cufft_plan_impl *>(plan);
    }

    sycl::event execute(
        void *plan, const void *input, void *output, std::span<const sycl::event> dependencies,
        char *error, std::size_t capacity)
    {
        try {
            if (!plan || !input || !output) {
                throw std::runtime_error("invalid cuFFT execution argument");
            }
            return static_cast<cufft_plan_impl *>(plan)->execute(input, output, dependencies);
        } catch (const std::exception &ex) {
            set_error(error, capacity, ex.what());
            return {};
        }
    }

    const syclfft::detail::cufft_provider_v1 provider_api{
        .abi_version = SYCLFFT_CUFFT_PROVIDER_ABI_VERSION,
        .name = "cufft",
        .supports = &supports,
        .create = &create,
        .destroy = &destroy,
        .execute = &execute,
    };

} // namespace

SYCLFFT_CUFFT_PROVIDER_EXPORT const syclfft::detail::cufft_provider_v1 *
    syclfft_get_cufft_provider_v1()
{
    return &provider_api;
}
