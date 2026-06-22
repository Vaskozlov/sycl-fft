#include <syclfft/detail/cufft_provider_abi.hpp>
#include <syclfft/detail/provider_loader.hpp>
#include <syclfft/syclfft.hpp>

#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
#    include <syclfft/host.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace syclfft
{
    namespace
    {

        bool is_power_of_two(std::size_t value)
        {
            return value != 0 && (value & (value - 1)) == 0;
        }

        std::size_t next_power_of_two(std::size_t value)
        {
            std::size_t result = 1;
            while (result < value) {
                if (result > std::numeric_limits<std::size_t>::max() / 2) {
                    throw exception(
                        error_code::invalid_argument,
                        "Bluestein convolution size overflows size_t");
                }
                result *= 2;
            }
            return result;
        }

        std::vector<std::size_t> supported_factors(std::size_t value)
        {
            std::vector<std::size_t> result;
            for (const std::size_t radix : {2u, 3u, 5u, 7u}) {
                while (value % radix == 0) {
                    result.push_back(radix);
                    value /= radix;
                }
            }
            if (value != 1) {
                result.clear();
            }
            return result;
        }

        std::size_t transform_size(const std::vector<std::size_t> &shape)
        {
            std::size_t result = 1;
            for (const auto length : shape) {
                result *= length;
            }
            return result;
        }

        std::size_t axis_stride(const std::vector<std::size_t> &shape, std::size_t axis)
        {
            std::size_t result = 1;
            for (std::size_t index = axis + 1; index < shape.size(); ++index) {
                result *= shape[index];
            }
            return result;
        }

        template <class Scalar>
        void validate_pointer(
            const complex<Scalar> *pointer, const sycl::queue &queue, const char *name)
        {
            if (!pointer) {
                throw exception(
                    error_code::invalid_pointer, std::string(name) + " must be non-null");
            }
            if (sycl::get_pointer_type(pointer, queue.get_context()) == sycl::usm::alloc::unknown) {
                throw exception(
                    error_code::invalid_pointer,
                    std::string(name) + " is not a USM allocation in the plan context");
            }
        }

        template <class Scalar>
        sycl::event enqueue_copy_scaled(
            sycl::queue &queue, const complex<Scalar> *input, complex<Scalar> *output,
            std::size_t count, Scalar scale, const std::vector<sycl::event> &dependencies)
        {
            return queue.submit([&](sycl::handler &handler) {
                handler.depends_on(dependencies);
                handler.parallel_for(sycl::range<1>{count}, [=](sycl::id<1> index) {
                    output[index[0]] = input[index[0]] * scale;
                });
            });
        }

        template <class Scalar>
        sycl::event enqueue_mixed_axis(
            sycl::queue &queue, const complex<Scalar> *input, complex<Scalar> *output,
            std::size_t total, std::size_t length, std::size_t stride,
            const complex<Scalar> *twiddles, const std::vector<std::size_t> &factors,
            const std::vector<sycl::event> &dependencies)
        {
            std::array<std::size_t, 64> radices{};
            if (factors.size() > radices.size()) {
                throw exception(error_code::planning_failed, "Too many mixed-radix stages");
            }
            std::copy(factors.begin(), factors.end(), radices.begin());
            const auto factor_count = factors.size();
            sycl::event current = queue.submit([&](sycl::handler &handler) {
                handler.depends_on(dependencies);
                handler.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> item) {
                    const std::size_t linear = item[0];
                    const std::size_t coordinate = (linear / stride) % length;
                    const std::size_t base = linear - coordinate * stride;
                    std::size_t source_coordinate = 0;
                    std::size_t remaining = length;
                    std::size_t digits = coordinate;
                    for (std::size_t stage = 0; stage < factor_count; ++stage) {
                        const auto radix = radices[stage];
                        remaining /= radix;
                        source_coordinate += (digits % radix) * remaining;
                        digits /= radix;
                    }
                    output[linear] = input[base + source_coordinate * stride];
                });
            });

            const std::size_t line_count = total / length;
            std::size_t stage_size = 1;
            for (const auto radix : factors) {
                const auto previous_size = stage_size;
                stage_size *= radix;
                const auto captured_stage_size = stage_size;
                const auto captured_radix = radix;
                const auto prior = current;
                current = queue.submit([&](sycl::handler &handler) {
                    handler.depends_on(prior);
                    handler.parallel_for(
                        sycl::range<1>{line_count * (length / captured_radix)},
                        [=](sycl::id<1> item) {
                        const std::size_t butterfly = item[0];
                        const std::size_t line = butterfly / (length / captured_radix);
                        const std::size_t in_line = butterfly % (length / captured_radix);
                        const std::size_t group = in_line / previous_size;
                        const std::size_t j = in_line % previous_size;
                        const std::size_t outer = line / stride;
                        const std::size_t inner = line % stride;
                        const std::size_t base = outer * length * stride + inner
                                                 + group * captured_stage_size * stride
                                                 + j * stride;
                        complex<Scalar> values[7];
                        for (std::size_t q = 0; q < captured_radix; ++q) {
                            values[q] =
                                output[base + q * previous_size * stride]
                                * twiddles[(q * j * (length / captured_stage_size)) % length];
                        }
                        for (std::size_t p = 0; p < captured_radix; ++p) {
                            complex<Scalar> sum{Scalar{0}, Scalar{0}};
                            for (std::size_t q = 0; q < captured_radix; ++q) {
                                sum += values[q]
                                       * twiddles[(q * p * (length / captured_radix)) % length];
                            }
                            output[base + p * previous_size * stride] = sum;
                        }
                    });
                });
            }
            return current;
        }

        template <class Scalar>
        sycl::event enqueue_radix2_axis(
            sycl::queue &queue, const complex<Scalar> *input, complex<Scalar> *output,
            std::size_t total, std::size_t length, std::size_t stride,
            const complex<Scalar> *twiddles, bool conjugate_twiddles,
            const std::vector<sycl::event> &dependencies)
        {
            unsigned bits = 0;
            for (std::size_t value = length; value > 1; value >>= 1) {
                ++bits;
            }

            sycl::event current = queue.submit([&](sycl::handler &handler) {
                handler.depends_on(dependencies);
                handler.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> item) {
                    const std::size_t linear = item[0];
                    const std::size_t coordinate = (linear / stride) % length;
                    std::size_t reversed = 0;
                    std::size_t value = coordinate;
                    for (unsigned bit = 0; bit < bits; ++bit) {
                        reversed = (reversed << 1) | (value & 1);
                        value >>= 1;
                    }
                    const std::size_t base = linear - coordinate * stride;
                    output[linear] = input[base + reversed * stride];
                });
            });

            const std::size_t line_count = total / length;
            for (std::size_t stage = 2; stage <= length; stage <<= 1) {
                const std::size_t stage_value = stage;
                const auto prior = current;
                current = queue.submit([&](sycl::handler &handler) {
                    handler.depends_on(prior);
                    handler.parallel_for(
                        sycl::range<1>{line_count * (length / 2)}, [=](sycl::id<1> item) {
                        const std::size_t pair = item[0];
                        const std::size_t line = pair / (length / 2);
                        const std::size_t pair_in_line = pair % (length / 2);
                        const std::size_t half = stage_value / 2;
                        const std::size_t group = pair_in_line / half;
                        const std::size_t j = pair_in_line % half;
                        const std::size_t outer = line / stride;
                        const std::size_t inner = line % stride;
                        const std::size_t base = outer * length * stride + inner;
                        const std::size_t first = base + (group * stage_value + j) * stride;
                        const std::size_t second = first + half * stride;
                        auto twiddle = twiddles[j * (length / stage_value)];
                        if (conjugate_twiddles) {
                            twiddle = {twiddle.real(), -twiddle.imag()};
                        }
                        const auto lhs = output[first];
                        const auto rhs = output[second] * twiddle;
                        output[first] = lhs + rhs;
                        output[second] = lhs - rhs;
                    });
                });
            }
            return current;
        }

        template <class Scalar>
        void host_radix2_fft(std::vector<complex<Scalar>> &values, direction dir)
        {
            const auto length = values.size();
            for (std::size_t index = 1, reversed = 0; index < length; ++index) {
                std::size_t bit = length >> 1;
                while (reversed & bit) {
                    reversed ^= bit;
                    bit >>= 1;
                }
                reversed ^= bit;
                if (index < reversed) {
                    std::swap(values[index], values[reversed]);
                }
            }
            const Scalar sign = dir == direction::forward ? Scalar{-1} : Scalar{1};
            const Scalar two_pi = Scalar{6.283185307179586476925286766559};
            for (std::size_t stage = 2; stage <= length; stage *= 2) {
                for (std::size_t base = 0; base < length; base += stage) {
                    for (std::size_t j = 0; j < stage / 2; ++j) {
                        const auto angle =
                            sign * two_pi * static_cast<Scalar>(j) / static_cast<Scalar>(stage);
                        const complex<Scalar> twiddle{std::cos(angle), std::sin(angle)};
                        const auto lhs = values[base + j];
                        const auto rhs = values[base + j + stage / 2] * twiddle;
                        values[base + j] = lhs + rhs;
                        values[base + j + stage / 2] = lhs - rhs;
                    }
                }
            }
        }

        template <class Scalar>
        sycl::event enqueue_bluestein_axis(
            sycl::queue &queue, const complex<Scalar> *input, complex<Scalar> *first,
            complex<Scalar> *second, std::size_t total, std::size_t length, std::size_t stride,
            std::size_t convolution_length, const complex<Scalar> *chirp,
            const complex<Scalar> *kernel_spectrum, const complex<Scalar> *convolution_twiddles,
            const std::vector<sycl::event> &dependencies)
        {
            const auto line_count = total / length;
            const auto expanded_count = line_count * convolution_length;

            auto current = queue.submit([&](sycl::handler &handler) {
                handler.depends_on(dependencies);
                handler.parallel_for(sycl::range<1>{expanded_count}, [=](sycl::id<1> item) {
                    const auto expanded = static_cast<std::size_t>(item[0]);
                    const auto line = expanded / convolution_length;
                    const auto coordinate = expanded % convolution_length;
                    if (coordinate < length) {
                        const auto outer = line / stride;
                        const auto inner = line % stride;
                        const auto source = outer * length * stride + inner + coordinate * stride;
                        first[expanded] = input[source] * chirp[coordinate];
                    } else {
                        first[expanded] = {Scalar{0}, Scalar{0}};
                    }
                });
            });

            current = enqueue_radix2_axis(
                queue,
                first,
                second,
                expanded_count,
                convolution_length,
                1,
                convolution_twiddles,
                false,
                {current});
            auto prior = current;
            current = queue.submit([&](sycl::handler &handler) {
                handler.depends_on(prior);
                handler.parallel_for(sycl::range<1>{expanded_count}, [=](sycl::id<1> item) {
                    const auto index = static_cast<std::size_t>(item[0]);
                    second[index] *= kernel_spectrum[index % convolution_length];
                });
            });
            current = enqueue_radix2_axis(
                queue,
                second,
                first,
                expanded_count,
                convolution_length,
                1,
                convolution_twiddles,
                true,
                {current});
            const auto inverse_scale = Scalar{1} / static_cast<Scalar>(convolution_length);
            prior = current;
            current = queue.submit([&](sycl::handler &handler) {
                handler.depends_on(prior);
                handler.parallel_for(sycl::range<1>{total}, [=](sycl::id<1> item) {
                    const auto linear = static_cast<std::size_t>(item[0]);
                    const auto coordinate = (linear / stride) % length;
                    const auto line = (linear / (length * stride)) * stride + linear % stride;
                    second[linear] = first[line * convolution_length + coordinate]
                                     * complex<Scalar>{
                                         chirp[coordinate].real() * inverse_scale,
                                         chirp[coordinate].imag() * inverse_scale};
                });
            });
            return current;
        }

        bool is_cuda_queue(const sycl::queue &queue)
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

        struct loaded_cufft_provider
        {
            const detail::cufft_provider_v1 *api{};
            std::shared_ptr<void> module;
        };

        loaded_cufft_provider
            try_load_cufft(const sycl::queue &queue, bool required, std::string *reason = nullptr)
        {
            if (!is_cuda_queue(queue)) {
                if (reason) {
                    *reason = "queue does not use CUDA";
                }
                if (required) {
                    throw exception(
                        error_code::provider_unavailable,
                        "cuFFT requires a CUDA-backed SYCL queue");
                }
                return {};
            }
#if defined(SYCLFFT_HAS_CUFFT_PROVIDER)
            try {
                auto symbol = detail::load_provider_symbol(
                    "syclfft_provider_cufft", "syclfft_get_cufft_provider_v1");
                auto getter = reinterpret_cast<detail::get_cufft_provider_v1_fn>(symbol.symbol);
                const auto *api = getter();
                if (!api || api->abi_version != SYCLFFT_CUFFT_PROVIDER_ABI_VERSION) {
                    throw exception(error_code::plugin_error, "cuFFT provider ABI mismatch");
                }
                std::array<char, 1024> error{};
                if (!api->supports(queue, error.data(), error.size())) {
                    if (reason) {
                        *reason = error.data();
                    }
                    if (required) {
                        throw exception(
                            error_code::provider_unavailable,
                            error[0] ? error.data() : "cuFFT provider rejected the queue");
                    }
                    return {};
                }
                return {api, std::move(symbol.module)};
            } catch (const std::exception &ex) {
                if (reason) {
                    *reason = ex.what();
                }
                if (required) {
                    throw;
                }
                return {};
            }
#else
            if (reason) {
                *reason = "provider was not built";
            }
            if (required) {
                throw exception(error_code::provider_unavailable, "cuFFT provider was not built");
            }
            return {};
#endif
        }

    } // namespace

    template <class Scalar>
    class plan<Scalar>::impl
    {
    public:
        struct axis_twiddle_table
        {
            std::size_t axis{};
            complex_type *twiddles{};
        };

        struct bluestein_table
        {
            std::size_t axis{};
            std::size_t convolution_length{};
            complex_type *chirp{};
            complex_type *kernel_spectrum{};
            complex_type *convolution_twiddles{};
        };

        impl(
            sycl::queue queue, std::vector<std::size_t> lengths, std::size_t batch_count,
            syclfft::direction dir, plan_options opts)
          : queue_(std::move(queue))
          , lengths_(std::move(lengths))
          , batch_count_(batch_count)
          , element_count_(detail::checked_element_count(lengths_, batch_count_))
          , transform_size_(transform_size(lengths_))
          , direction_(dir)
          , options_(opts)
        {
            if (direction_ != syclfft::direction::forward
                && direction_ != syclfft::direction::backward) {
                throw exception(error_code::invalid_argument, "Invalid FFT direction");
            }
            if constexpr (std::is_same_v<Scalar, double>) {
                if (!queue_.get_device().has(sycl::aspect::fp64)) {
                    throw exception(
                        error_code::unsupported_precision,
                        "The selected SYCL device does not support fp64");
                }
            }

            if (options_.preferred_provider == provider::automatic) {
                cufft_ = try_load_cufft(queue_, false);
                selected_provider_ = cufft_.api ? provider::cufft : provider::portable_sycl;
            } else if (options_.preferred_provider == provider::portable_sycl) {
                selected_provider_ = provider::portable_sycl;
            } else if (options_.preferred_provider == provider::cufft) {
                cufft_ = try_load_cufft(queue_, true);
                selected_provider_ = provider::cufft;
            } else if (options_.preferred_provider == provider::fftw) {
#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
                if (!queue_.get_device().is_cpu()) {
                    throw exception(
                        error_code::provider_unavailable,
                        "FFTW can only be selected for a SYCL CPU queue");
                }
                selected_provider_ = provider::fftw;
                host_plan_.emplace(
                    host::plan_many_dft<Scalar>(lengths_, batch_count_, direction_, options_));
#else
                throw exception(
                    error_code::provider_unavailable,
                    "FFTW provider was not built with the SYCL component");
#endif
            } else {
                throw exception(
                    error_code::provider_unavailable,
                    std::string(detail::provider_name(options_.preferred_provider))
                        + " provider is not implemented in v0.1");
            }

            if (selected_provider_ == provider::cufft) {
                detail::cufft_plan_config_v1 config{};
                config.double_precision = std::is_same_v<Scalar, double>;
                config.transform_direction = direction_;
                config.transform_placement = options_.placement;
                config.rank = static_cast<std::uint32_t>(lengths_.size());
                std::copy(lengths_.begin(), lengths_.end(), config.lengths.begin());
                config.batch_count = batch_count_;
                std::array<char, 1024> error{};
                cufft_handle_ = cufft_.api->create(queue_, config, error.data(), error.size());
                if (!cufft_handle_) {
                    throw exception(
                        error_code::planning_failed,
                        error[0] ? error.data() : "cuFFT plan creation failed");
                }
            } else if (selected_provider_ == provider::portable_sycl) {
                if (!queue_.get_device().has(sycl::aspect::usm_device_allocations)) {
                    throw exception(
                        error_code::provider_unavailable,
                        "Portable provider requires device USM allocations");
                }
                try {
                    const auto upload_table = [this](const std::vector<complex_type> &host) {
                        auto *device = sycl::malloc_device<complex_type>(host.size(), queue_);
                        if (!device) {
                            throw std::bad_alloc{};
                        }
                        table_allocations_.push_back(device);
                        queue_.memcpy(device, host.data(), host.size() * sizeof(complex_type))
                            .wait_and_throw();
                        return device;
                    };
                    const auto make_twiddles = [](std::size_t length, Scalar sign) {
                        std::vector<complex_type> result(length);
                        const Scalar two_pi = Scalar{6.283185307179586476925286766559};
                        for (std::size_t index = 0; index < length; ++index) {
                            const auto angle = sign * two_pi * static_cast<Scalar>(index)
                                               / static_cast<Scalar>(length);
                            result[index] = {std::cos(angle), std::sin(angle)};
                        }
                        return result;
                    };

                    scratch_elements_ = element_count_;
                    for (std::size_t axis = 0; axis < lengths_.size(); ++axis) {
                        const auto length = lengths_[axis];
                        const Scalar sign =
                            direction_ == direction::forward ? Scalar{-1} : Scalar{1};
                        if (!supported_factors(length).empty() || length == 1) {
                            auto twiddles = make_twiddles(length, sign);
                            axis_twiddle_tables_.push_back({axis, upload_table(twiddles)});
                            continue;
                        }
                        if (length > std::numeric_limits<std::size_t>::max() / 2 + 1) {
                            throw exception(
                                error_code::invalid_argument,
                                "Bluestein convolution size overflows size_t");
                        }
                        const auto convolution_length = next_power_of_two(2 * length - 1);
                        const auto line_count = element_count_ / length;
                        if (line_count
                            > std::numeric_limits<std::size_t>::max() / convolution_length) {
                            throw exception(
                                error_code::invalid_argument,
                                "Bluestein scratch size overflows size_t");
                        }
                        scratch_elements_ =
                            std::max(scratch_elements_, line_count * convolution_length);

                        std::vector<complex_type> kernel(
                            convolution_length, complex_type{Scalar{0}, Scalar{0}});
                        const Scalar pi = Scalar{3.1415926535897932384626433832795};
                        std::vector<complex_type> chirp(length);
                        for (std::size_t index = 0; index < length; ++index) {
                            const auto square =
                                static_cast<Scalar>(index) * static_cast<Scalar>(index);
                            const auto angle = -sign * pi * square / static_cast<Scalar>(length);
                            const complex_type value{std::cos(angle), std::sin(angle)};
                            kernel[index] = value;
                            if (index != 0) {
                                kernel[convolution_length - index] = value;
                            }
                            chirp[index] = {value.real(), -value.imag()};
                        }
                        host_radix2_fft(kernel, direction::forward);
                        auto convolution_twiddles = make_twiddles(convolution_length, Scalar{-1});
                        bluestein_tables_.push_back(
                            {axis,
                             convolution_length,
                             upload_table(chirp),
                             upload_table(kernel),
                             upload_table(convolution_twiddles)});
                    }

                    scratch_a_ = sycl::malloc_device<complex_type>(scratch_elements_, queue_);
                    scratch_b_ = sycl::malloc_device<complex_type>(scratch_elements_, queue_);
                    if (!scratch_a_ || !scratch_b_) {
                        throw std::bad_alloc{};
                    }
                } catch (...) {
                    if (scratch_a_) {
                        sycl::free(scratch_a_, queue_);
                    }
                    if (scratch_b_) {
                        sycl::free(scratch_b_, queue_);
                    }
                    for (auto *allocation : table_allocations_) {
                        sycl::free(allocation, queue_);
                    }
                    throw exception(
                        error_code::planning_failed,
                        "Unable to allocate or initialize portable FFT scratch memory");
                }
            }
        }

        ~impl()
        {
            try {
                if (last_event_) {
                    last_event_->wait_and_throw();
                }
            } catch (...) {
            }
            if (scratch_a_) {
                sycl::free(scratch_a_, queue_);
            }
            if (scratch_b_) {
                sycl::free(scratch_b_, queue_);
            }
            for (auto *allocation : table_allocations_) {
                sycl::free(allocation, queue_);
            }
            if (cufft_handle_) {
                cufft_.api->destroy(cufft_handle_);
            }
        }

        sycl::event execute(
            const complex_type *input, complex_type *output,
            std::span<const sycl::event> dependencies, bool one_pointer)
        {
            validate_pointer(input, queue_, "input");
            validate_pointer(output, queue_, "output");
            if (options_.placement == placement::in_place && !one_pointer) {
                throw exception(
                    error_code::invalid_argument,
                    "In-place plan must be executed with execute(inout)");
            }
            if (options_.placement == placement::out_of_place && one_pointer) {
                throw exception(
                    error_code::invalid_argument,
                    "Out-of-place plan requires input and output arguments");
            }
            if (options_.placement == placement::out_of_place && input == output) {
                throw exception(
                    error_code::invalid_argument,
                    "Out-of-place plan requires distinct allocations");
            }

            std::scoped_lock lock(mutex_);
            std::vector<sycl::event> all_dependencies(dependencies.begin(), dependencies.end());
            if (last_event_) {
                all_dependencies.push_back(*last_event_);
            }

            if (selected_provider_ == provider::fftw) {
#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
                for (auto dependency : all_dependencies) {
                    dependency.wait_and_throw();
                }
                queue_.wait_and_throw();
                if (sycl::get_pointer_type(input, queue_.get_context()) == sycl::usm::alloc::device
                    || sycl::get_pointer_type(output, queue_.get_context())
                           == sycl::usm::alloc::device) {
                    throw exception(
                        error_code::invalid_pointer,
                        "Blocking FFTW execution requires host or shared USM");
                }
                if (one_pointer) {
                    host_plan_->execute(reinterpret_cast<std::complex<Scalar> *>(output));
                } else {
                    host_plan_->execute(
                        reinterpret_cast<const std::complex<Scalar> *>(input),
                        reinterpret_cast<std::complex<Scalar> *>(output));
                }
                auto completed =
                    queue_.submit([](sycl::handler &handler) { handler.single_task([]{}); });
                last_event_ = completed;
                return completed;
#endif
            }

            if (selected_provider_ == provider::cufft) {
                std::array<char, 1024> error{};
                auto current = cufft_.api->execute(
                    cufft_handle_, input, output, all_dependencies, error.data(), error.size());
                if (error[0]) {
                    throw exception(error_code::execution_failed, error.data());
                }
                const auto scale = static_cast<Scalar>(detail::normalization_scale(
                    options_.normalization, direction_, transform_size_));
                if (scale != Scalar{1}) {
                    std::vector<sycl::event> scale_dependency{current};
                    current = enqueue_copy_scaled(
                        queue_, output, output, element_count_, scale, scale_dependency);
                }
                last_event_ = current;
                return current;
            }

            const complex_type *current_input = input;
            complex_type *current_output = scratch_a_;
            sycl::event current;
            bool have_event = false;
            for (std::size_t axis = 0; axis < lengths_.size(); ++axis) {
                std::vector<sycl::event> axis_dependencies;
                if (have_event) {
                    axis_dependencies.push_back(current);
                } else {
                    axis_dependencies = all_dependencies;
                }

                const auto length = lengths_[axis];
                const auto stride = axis_stride(lengths_, axis);
                const auto factors = supported_factors(length);
                if (!factors.empty() || length == 1) {
                    const auto table = std::find_if(
                        axis_twiddle_tables_.begin(),
                        axis_twiddle_tables_.end(),
                        [axis](const axis_twiddle_table &candidate) {
                        return candidate.axis == axis;
                    });
                    if (table == axis_twiddle_tables_.end()) {
                        throw exception(
                            error_code::invalid_state, "Missing mixed-radix twiddle table");
                    }
                    current_output = current_input == scratch_a_ ? scratch_b_ : scratch_a_;
                    if (is_power_of_two(length)) {
                        current = enqueue_radix2_axis(
                            queue_,
                            current_input,
                            current_output,
                            element_count_,
                            length,
                            stride,
                            table->twiddles,
                            false,
                            axis_dependencies);
                    } else {
                        current = enqueue_mixed_axis(
                            queue_,
                            current_input,
                            current_output,
                            element_count_,
                            length,
                            stride,
                            table->twiddles,
                            factors,
                            axis_dependencies);
                    }
                } else {
                    const auto table = std::find_if(
                        bluestein_tables_.begin(),
                        bluestein_tables_.end(),
                        [axis](const bluestein_table &candidate) {
                        return candidate.axis == axis;
                    });
                    if (table == bluestein_tables_.end()) {
                        throw exception(error_code::invalid_state, "Missing Bluestein plan table");
                    }
                    auto *first = current_input == scratch_a_ ? scratch_b_ : scratch_a_;
                    auto *second = first == scratch_a_ ? scratch_b_ : scratch_a_;
                    current = enqueue_bluestein_axis(
                        queue_,
                        current_input,
                        first,
                        second,
                        element_count_,
                        length,
                        stride,
                        table->convolution_length,
                        table->chirp,
                        table->kernel_spectrum,
                        table->convolution_twiddles,
                        axis_dependencies);
                    current_output = second;
                }
                have_event = true;
                current_input = current_output;
            }

            const auto scale = static_cast<Scalar>(
                detail::normalization_scale(options_.normalization, direction_, transform_size_));
            std::vector<sycl::event> final_dependency{current};
            current = enqueue_copy_scaled(
                queue_, current_input, output, element_count_, scale, final_dependency);
            last_event_ = current;
            return current;
        }

        sycl::queue queue_;
        std::vector<std::size_t> lengths_;
        std::size_t batch_count_{};
        std::size_t element_count_{};
        std::size_t transform_size_{};
        std::size_t scratch_elements_{};
        syclfft::direction direction_{syclfft::direction::forward};
        plan_options options_{};
        provider selected_provider_{provider::portable_sycl};
        complex_type *scratch_a_{};
        complex_type *scratch_b_{};
        std::vector<complex_type *> table_allocations_;
        std::vector<axis_twiddle_table> axis_twiddle_tables_;
        std::vector<bluestein_table> bluestein_tables_;
        std::optional<sycl::event> last_event_;
        std::mutex mutex_;
        loaded_cufft_provider cufft_;
        void *cufft_handle_{};
#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
        std::optional<host::plan<Scalar>> host_plan_;
#endif
    };

    template <class Scalar>
    plan<Scalar>::plan(
        sycl::queue queue, std::vector<std::size_t> lengths, std::size_t batch_count,
        syclfft::direction direction, plan_options options)
      : impl_(
            std::make_unique<impl>(
                std::move(queue), std::move(lengths), batch_count, direction, options))
    {}

    template <class Scalar>
    plan<Scalar>::plan(plan &&) noexcept = default;
    template <class Scalar>
    plan<Scalar> &plan<Scalar>::operator=(plan &&) noexcept = default;
    template <class Scalar>
    plan<Scalar>::~plan() = default;

    template <class Scalar>
    sycl::event plan<Scalar>::execute(complex_type *inout)
    {
        if (!impl_) {
            throw exception(error_code::invalid_state, "Cannot execute a moved-from plan");
        }
        return impl_->execute(inout, inout, {}, true);
    }

    template <class Scalar>
    sycl::event plan<Scalar>::execute(complex_type *inout, const sycl::event &dependency)
    {
        if (!impl_) {
            throw exception(error_code::invalid_state, "Cannot execute a moved-from plan");
        }
        return impl_->execute(inout, inout, std::span<const sycl::event>{&dependency, 1}, true);
    }

    template <class Scalar>
    sycl::event
        plan<Scalar>::execute(complex_type *inout, std::span<const sycl::event> dependencies)
    {
        if (!impl_) {
            throw exception(error_code::invalid_state, "Cannot execute a moved-from plan");
        }
        return impl_->execute(inout, inout, dependencies, true);
    }

    template <class Scalar>
    sycl::event plan<Scalar>::execute(const complex_type *input, complex_type *output)
    {
        if (!impl_) {
            throw exception(error_code::invalid_state, "Cannot execute a moved-from plan");
        }
        return impl_->execute(input, output, {}, false);
    }

    template <class Scalar>
    sycl::event plan<Scalar>::execute(
        const complex_type *input, complex_type *output, const sycl::event &dependency)
    {
        if (!impl_) {
            throw exception(error_code::invalid_state, "Cannot execute a moved-from plan");
        }
        return impl_->execute(input, output, std::span<const sycl::event>{&dependency, 1}, false);
    }

    template <class Scalar>
    sycl::event plan<Scalar>::execute(
        const complex_type *input, complex_type *output, std::span<const sycl::event> dependencies)
    {
        if (!impl_) {
            throw exception(error_code::invalid_state, "Cannot execute a moved-from plan");
        }
        return impl_->execute(input, output, dependencies, false);
    }

    template <class Scalar>
    std::span<const std::size_t> plan<Scalar>::shape() const noexcept
    {
        return impl_ ? std::span<const std::size_t>{impl_->lengths_}
                     : std::span<const std::size_t>{};
    }

    template <class Scalar>
    std::size_t plan<Scalar>::batch_count() const noexcept
    {
        return impl_ ? impl_->batch_count_ : 0;
    }

    template <class Scalar>
    syclfft::direction plan<Scalar>::direction() const noexcept
    {
        return impl_ ? impl_->direction_ : syclfft::direction::forward;
    }

    template <class Scalar>
    syclfft::provider plan<Scalar>::selected_provider() const noexcept
    {
        return impl_ ? impl_->selected_provider_ : provider::automatic;
    }

    template <class Scalar>
    std::size_t plan<Scalar>::scratch_size_bytes() const noexcept
    {
        return impl_ && impl_->selected_provider_ == provider::portable_sycl
                   ? 2 * impl_->scratch_elements_ * sizeof(complex_type)
                   : 0;
    }

    std::vector<provider_status> query_providers(const sycl::queue &queue)
    {
        std::vector<provider_status> result;
        const auto portable_available =
            queue.get_device().has(sycl::aspect::usm_device_allocations);
        result.push_back(
            {provider::portable_sycl,
             "portable_sycl",
             true,
             portable_available,
             portable_available ? "" : "device USM allocations are unsupported"});
        std::string cufft_reason;
        const auto cufft = try_load_cufft(queue, false, &cufft_reason);
#if defined(SYCLFFT_HAS_CUFFT_PROVIDER)
        result.push_back(
            {provider::cufft, "cufft", true, cufft.api != nullptr, cufft.api ? "" : cufft_reason});
#else
        result.push_back({provider::cufft, "cufft", false, false, cufft_reason});
#endif
#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
        if (!queue.get_device().is_cpu()) {
            result.push_back({provider::fftw, "fftw", true, false, "queue is not a CPU device"});
        } else {
            try {
                (void)detail::load_host_provider("syclfft_provider_fftw");
                result.push_back({provider::fftw, "fftw", true, true, {}});
            } catch (const std::exception &ex) {
                result.push_back({provider::fftw, "fftw", true, false, ex.what()});
            }
        }
#else
        result.push_back({provider::fftw, "fftw", false, false, "provider was not built"});
#endif
        result.push_back(
            {provider::rocfft, "rocfft", false, false, "reserved for a future provider"});
        result.push_back(
            {provider::onemkl, "onemkl", false, false, "reserved for a future provider"});
        return result;
    }

    template class SYCLFFT_SYCL_EXPORT plan<float>;
    template class SYCLFFT_SYCL_EXPORT plan<double>;

} // namespace syclfft
