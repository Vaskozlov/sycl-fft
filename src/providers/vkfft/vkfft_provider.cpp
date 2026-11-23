#include "vkfft_compiler.hpp"

#include <cerrno>
#include <exception>
#include <limits>
#include <locale.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <syclfft/detail/cuda_queue_utils.hpp>
#include <syclfft/detail/cufft_provider_abi.hpp>
#include <syclfft/detail/provider_error_buffer.hpp>
#include <system_error>
#include <utility>
#include <vector>

// The macros in this header redirect VkFFT driver calls.
// clang-format off
#include "vkfft_cuda_driver.hpp"
#include <vkFFT.h>
// clang-format on

namespace
{
    using plan_config = syclfft::detail::cufft_plan_config_v2;

    void check_driver(CUresult result, const char *operation)
    {
        if (result != CUDA_SUCCESS) {
            const char *detail = nullptr;
            cuGetErrorString(result, &detail);
            throw std::runtime_error(
                std::string(operation) + ": " + (detail ? detail : "CUDA driver error"));
        }
    }

    void check_vkfft(VkFFTResult result, const char *operation)
    {
        if (result != VKFFT_SUCCESS) {
            throw std::runtime_error(std::string(operation) + ": " + getVkFFTErrorString(result));
        }
    }

    CUcontext stream_context(CUstream stream)
    {
        CUcontext context{};
        if (stream) {
            check_driver(cuStreamGetCtx(stream, &context), "cuStreamGetCtx");
        } else {
            check_driver(cuCtxGetCurrent(&context), "cuCtxGetCurrent");
        }
        if (!context) {
            throw std::runtime_error("CUDA SYCL queue has no native context");
        }
        return context;
    }

    class current_cuda_context
    {
    public:
        explicit current_cuda_context(CUcontext context)
        {
            CUcontext previous{};
            check_driver(cuCtxGetCurrent(&previous), "cuCtxGetCurrent");
            if (previous != context) {
                check_driver(cuCtxPushCurrent(context), "cuCtxPushCurrent");
                pushed_ = true;
            }
        }

        ~current_cuda_context()
        {
            if (pushed_) {
                CUcontext popped{};
                cuCtxPopCurrent(&popped);
            }
        }

        current_cuda_context(const current_cuda_context &) = delete;
        current_cuda_context &operator=(const current_cuda_context &) = delete;

    private:
        bool pushed_{};
    };

    class c_numeric_locale
    {
    public:
        c_numeric_locale()
        {
            const auto active = uselocale(nullptr);
            if (!active) {
                throw std::system_error(errno, std::generic_category(), "uselocale");
            }
            auto baseline = duplocale(active);
            if (!baseline) {
                throw std::system_error(errno, std::generic_category(), "duplocale");
            }
            locale_ = newlocale(LC_NUMERIC_MASK, "C", baseline);
            if (!locale_) {
                const auto error = errno;
                freelocale(baseline);
                throw std::system_error(error, std::generic_category(), "newlocale");
            }
            previous_ = uselocale(locale_);
            if (!previous_) {
                const auto error = errno;
                freelocale(locale_);
                throw std::system_error(error, std::generic_category(), "uselocale");
            }
        }

        ~c_numeric_locale()
        {
            uselocale(previous_);
            freelocale(locale_);
        }

        c_numeric_locale(const c_numeric_locale &) = delete;
        c_numeric_locale &operator=(const c_numeric_locale &) = delete;

    private:
        locale_t previous_{};
        locale_t locale_{};
    };

    template <class Function>
    sycl::event enqueue_cuda(
        sycl::queue &queue, Function function, const std::vector<sycl::event> &dependencies)
    {
        return queue.AdaptiveCpp_enqueue_custom_operation(
            [function = std::move(function)](sycl::interop_handle &interop) {
            function(interop.get_native_queue<sycl::backend::cuda>());
        }, dependencies);
    }

    struct config_and_bytes
    {
        VkFFTConfiguration configuration{};
        std::size_t bytes{};
    };

    config_and_bytes configure(const plan_config &config)
    {
        if (config.abi_version != SYCLFFT_CUFFT_PROVIDER_ABI_VERSION) {
            throw std::runtime_error("VkFFT provider ABI mismatch");
        }
        if (config.rank == 0 || config.rank > config.lengths.size() || config.batch_count == 0) {
            throw std::runtime_error("Invalid VkFFT rank or batch count");
        }
        if (config.transform_direction != syclfft::direction::forward
            && config.transform_direction != syclfft::direction::backward) {
            throw std::runtime_error("Invalid VkFFT direction");
        }

        config_and_bytes result;
        auto &vk_config = result.configuration;
        std::size_t count = config.batch_count;
        // VkFFT orders dimensions from fastest to slowest. Unit axes leave the layout unchanged.
        for (std::size_t axis = config.rank; axis > 0; --axis) {
            const auto length = config.lengths[axis - 1];
            if (length == 0 || count > std::numeric_limits<std::size_t>::max() / length) {
                throw std::runtime_error("Invalid VkFFT dimensions or element count");
            }
            count *= length;
            if (length > 1) {
                vk_config.size[vk_config.FFTdim++] = length;
            }
        }
        const auto element_bytes = 2 * (config.double_precision ? sizeof(double) : sizeof(float));
        if (count > std::numeric_limits<std::size_t>::max() / element_bytes) {
            throw std::runtime_error("VkFFT buffer size overflows size_t");
        }
        result.bytes = count * element_bytes;
        vk_config.numberBatches = config.batch_count;
        vk_config.doublePrecision = config.double_precision;
        vk_config.useLUT = 1;
        vk_config.useLUT_4step = 1;
        vk_config.isInputFormatted = config.transform_placement == syclfft::placement::out_of_place;
        vk_config.makeForwardPlanOnly = config.transform_direction == syclfft::direction::forward;
        vk_config.makeInversePlanOnly = config.transform_direction == syclfft::direction::backward;
        return result;
    }

    class vkfft_state
    {
    public:
        vkfft_state() = default;
        vkfft_state(const vkfft_state &) = delete;
        vkfft_state &operator=(const vkfft_state &) = delete;

        ~vkfft_state()
        {
            if (initialized_) {
                try {
                    const current_cuda_context current{context_};
                    deleteVkFFT(&application_);
                } catch (...) {
                }
            }
        }

        void initialize(CUstream stream, VkFFTConfiguration config, pfUINT bytes)
        {
            context_ = stream_context(stream);
            const current_cuda_context current{context_};
            check_driver(cuCtxGetDevice(&device_), "cuCtxGetDevice");
            stream_ = stream;
            bytes_ = bytes;
            config.device = &device_;
            config.stream = &stream_;
            config.num_streams = 1;
            config.bufferSize = &bytes_;
            if (config.isInputFormatted) {
                config.inputBufferSize = &bytes_;
            }
            config.compileCudaPtx = &syclfft::detail::compile_cuda_ptx;
            config.compileCudaPtxUserData = &compiler_error_;
            const c_numeric_locale numeric_locale;
            const auto result = initializeVkFFT(&application_, config);
            if (result != VKFFT_SUCCESS && !compiler_error_.empty()) {
                throw std::runtime_error("VkFFT PTX compilation failed: " + compiler_error_);
            }
            check_vkfft(result, "initializeVkFFT");
            initialized_ = true;
        }

        void append(CUstream stream, const void *input, void *output)
        {
            if (stream != stream_ && stream_context(stream) != context_) {
                throw std::runtime_error("VkFFT execution CUDA context differs from plan context");
            }
            const current_cuda_context current{context_};
            stream_ = stream;
            // VkFFT retains these addresses in its configuration and compiled axes across appends.
            input_ = const_cast<void *>(input);
            buffer_ = output;
            VkFFTLaunchParams parameters{};
            parameters.buffer = &buffer_;
            if (application_.configuration.isInputFormatted) {
                parameters.inputBuffer = &input_;
            }
            check_vkfft(
                VkFFTAppend(
                    &application_,
                    application_.configuration.makeInversePlanOnly ? 1 : -1,
                    &parameters),
                "VkFFTAppend");
        }

        [[nodiscard]] bool initialized() const noexcept
        {
            return initialized_;
        }

    private:
        VkFFTApplication application_{};
        CUcontext context_{};
        CUdevice device_{};
        CUstream stream_{};
        pfUINT bytes_{};
        void *input_{};
        void *buffer_{};
        bool initialized_{};
        std::string compiler_error_;
    };

    class vkfft_plan
    {
    public:
        vkfft_plan(sycl::queue queue, const plan_config &config)
          : queue_(std::move(queue))
        {
            if (!syclfft::detail::is_cuda_queue(queue_)) {
                throw std::runtime_error("queue is not CUDA-backed");
            }
            if (const auto *error = syclfft::detail::vkfft_cuda::driver_error()) {
                throw std::runtime_error(error);
            }

            const auto settings = configure(config);
            bytes_ = settings.bytes;
            if (settings.configuration.FFTdim == 0) {
                return;
            }

            struct preparation
            {
                std::unique_ptr<vkfft_state> state = std::make_unique<vkfft_state>();
                std::exception_ptr error;
            };

            auto pending = std::make_shared<preparation>();
            auto planned = enqueue_cuda(queue_, [pending, settings](CUstream stream) {
                try {
                    pending->state->initialize(stream, settings.configuration, settings.bytes);
                } catch (...) {
                    pending->error = std::current_exception();
                }
            }, {});
            planned.wait_and_throw();
            if (pending->error) {
                std::rethrow_exception(pending->error);
            }
            if (!pending->state->initialized()) {
                throw std::runtime_error("VkFFT initialization callback did not complete");
            }
            state_ = std::move(pending->state);
        }

        sycl::event
            execute(const void *input, void *output, syclfft::span<const sycl::event> dependencies)
        {
            const std::vector<sycl::event> deps(dependencies.begin(), dependencies.end());
            if (!state_) {
                return queue_.submit([&](sycl::handler &handler) {
                    handler.depends_on(deps);
                    if (input == output) {
                        handler.single_task([] {});
                    } else {
                        handler.memcpy(output, input, bytes_);
                    }
                });
            }

            auto * const state = state_.get();
            return enqueue_cuda(queue_, [state, input, output](CUstream stream) {
                try {
                    state->append(stream, input, output);
                } catch (const std::exception &error) {
                    hipsycl::rt::register_error(
                        __acpp_here(), hipsycl::rt::error_info{error.what()});
                }
            }, deps);
        }

    private:
        sycl::queue queue_;
        std::size_t bytes_{};
        std::unique_ptr<vkfft_state> state_;
    };

    bool supports(const sycl::queue &queue, char *error, std::size_t capacity)
    {
        try {
            if (!syclfft::detail::is_cuda_queue(queue)) {
                throw std::runtime_error("queue is not CUDA-backed");
            }
            syclfft::detail::require_cuda_ptx_compiler();
            if (const auto *message = syclfft::detail::vkfft_cuda::probe_driver()) {
                throw std::runtime_error(message);
            }
            return true;
        } catch (const std::exception &ex) {
            syclfft::detail::set_error_buffer(error, capacity, ex.what());
            return false;
        }
    }

    void *create(sycl::queue &queue, const plan_config &config, char *error, std::size_t capacity)
    {
        try {
            return new vkfft_plan(queue, config);
        } catch (const std::exception &ex) {
            syclfft::detail::set_error_buffer(error, capacity, ex.what());
            return nullptr;
        }
    }

    void destroy(void *plan)
    {
        delete static_cast<vkfft_plan *>(plan);
    }

    sycl::event execute(
        void *plan, const void *input, void *output, syclfft::span<const sycl::event> dependencies,
        char *error, std::size_t capacity)
    {
        try {
            if (!plan || !input || !output) {
                throw std::runtime_error("Invalid VkFFT execution argument");
            }
            return static_cast<vkfft_plan *>(plan)->execute(input, output, dependencies);
        } catch (const std::exception &ex) {
            syclfft::detail::set_error_buffer(error, capacity, ex.what());
            return {};
        }
    }

    const syclfft::detail::cufft_provider_v2 provider_api{
        SYCLFFT_CUFFT_PROVIDER_ABI_VERSION,
        "vkfft",
        &supports,
        &create,
        &destroy,
        &execute,
    };

} // namespace

SYCLFFT_CUFFT_PROVIDER_EXPORT const syclfft::detail::cufft_provider_v2 *
    syclfft_get_vkfft_provider_v2()
{
    return &provider_api;
}
