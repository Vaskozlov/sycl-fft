#include "cpu_radix2.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <string>
#include <syclfft/detail/cuda_queue_utils.hpp>
#include <syclfft/detail/cufft_provider_abi.hpp>
#include <syclfft/detail/provider_loader.hpp>
#include <syclfft/detail/trace.hpp>
#include <syclfft/syclfft.hpp>
#include <type_traits>
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

        constexpr unsigned local_fft_group_size = 256;

        enum class radix2_execution
        {
            staged,
            cpu_batch,
            gpu_work_group
        };

        template <class Scalar>
        radix2_execution
            select_radix2_execution(const sycl::queue &queue, const std::vector<std::size_t> &shape)
        {
            const auto length = shape.front();
            if (shape.size() != 1 || length < 1024 || length > 4096 || !is_power_of_two(length)) {
                return radix2_execution::staged;
            }
            const auto device = queue.get_device();
            if (device.is_cpu()) {
                return radix2_execution::cpu_batch;
            }
            if (detail::is_cuda_queue(queue)
                && device.get_info<sycl::info::device::max_work_group_size>()
                       >= local_fft_group_size
                && device.get_info<sycl::info::device::max_work_item_sizes<1>>()[0]
                       >= local_fft_group_size
                && device.get_info<sycl::info::device::local_mem_size>()
                       >= length * sizeof(complex<Scalar>)) {
                return radix2_execution::gpu_work_group;
            }
            return radix2_execution::staged;
        }

        constexpr double pi = 3.141592653589793238462643383279502884;

        template <class Scalar>
        constexpr syclfft_scalar_v1 scalar_id()
        {
            if constexpr (std::is_same_v<Scalar, float>) {
                return SYCLFFT_SCALAR_FLOAT_V1;
            }
            return SYCLFFT_SCALAR_DOUBLE_V1;
        }

        std::int32_t direction_id(syclfft::direction direction)
        {
            switch (direction) {
            case syclfft::direction::forward:
                return SYCLFFT_DIRECTION_FORWARD_V1;
            case syclfft::direction::backward:
                return SYCLFFT_DIRECTION_BACKWARD_V1;
            }
            throw exception(error_code::invalid_argument, "Invalid FFT direction");
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

        std::size_t axis_stride(const std::vector<std::size_t> &shape, std::size_t axis)
        {
            std::size_t result = 1;
            for (std::size_t index = axis + 1; index < shape.size(); ++index) {
                result *= shape[index];
            }
            return result;
        }

#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
        template <class Scalar>
        class fftw_plan_adapter
        {
        public:
            using complex_type = complex<Scalar>;

            fftw_plan_adapter(
                std::vector<std::size_t> lengths, std::size_t batch_count, syclfft::direction dir,
                plan_options opts)
              : lengths_(std::move(lengths))
              , batch_count_(batch_count)
              , element_count_(detail::checked_element_count(lengths_, batch_count_))
              , direction_(dir)
              , options_(opts)
            {
                detail::trace("sycl: loading FFTW provider");
                auto loaded = detail::load_host_provider("syclfft_provider_fftw");
                detail::trace("sycl: FFTW provider loaded");
                api_ = loaded.api;
                module_ = std::move(loaded.module);

                syclfft_host_plan_config_v1 config{};
                config.abi_version = SYCLFFT_PROVIDER_ABI_VERSION;
                config.scalar = scalar_id<Scalar>();
                config.direction = direction_id(direction_);
                config.rank = static_cast<std::uint32_t>(lengths_.size());
                config.batch_count = static_cast<std::uint64_t>(batch_count_);
                config.in_place =
                    static_cast<std::uint32_t>(options_.placement == placement::in_place);
                config.measure =
                    static_cast<std::uint32_t>(options_.planning == planning_mode::measure);
                for (std::size_t i = 0; i < lengths_.size(); ++i) {
                    config.lengths[i] = static_cast<std::uint64_t>(lengths_[i]);
                }

                std::array<char, 1024> error{};
                detail::trace("sycl: calling FFTW provider create");
                handle_ = api_->create(&config, error.data(), error.size());
                detail::trace("sycl: FFTW provider create returned");
                if (!handle_) {
                    throw exception(
                        error_code::planning_failed,
                        error[0] ? error.data() : "FFTW plan creation failed");
                }
            }

            fftw_plan_adapter(const fftw_plan_adapter &) = delete;
            fftw_plan_adapter &operator=(const fftw_plan_adapter &) = delete;
            fftw_plan_adapter(fftw_plan_adapter &&) = delete;
            fftw_plan_adapter &operator=(fftw_plan_adapter &&) = delete;

            ~fftw_plan_adapter()
            {
                if (handle_) {
                    api_->destroy(handle_);
                }
            }

            void execute(const complex_type *input, complex_type *output, bool one_pointer)
            {
                if (!input || !output) {
                    throw exception(
                        error_code::invalid_pointer, "FFT input and output must be non-null");
                }
                if (options_.placement == placement::in_place && !one_pointer) {
                    throw exception(
                        error_code::invalid_argument,
                        "In-place plan must be executed with execute(inout)");
                }
                if (options_.placement == placement::out_of_place && one_pointer) {
                    throw exception(
                        error_code::invalid_argument,
                        "Out-of-place plan requires distinct input and output arguments");
                }

                std::scoped_lock lock(mutex_);
                std::array<char, 1024> error{};
                if (api_->execute(
                        handle_,
                        // FFTW's new-array API is not const-correct for preserved C2C input.
                        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
                        const_cast<complex_type *>(input),
                        output,
                        error.data(),
                        error.size())
                    != 0) {
                    throw exception(
                        error_code::execution_failed,
                        error[0] ? error.data() : "FFTW execution failed");
                }

                const auto scale = static_cast<Scalar>(detail::normalization_scale(
                    options_.normalization, direction_, detail::transform_size(lengths_)));
                if (scale != Scalar{1}) {
                    for (std::size_t i = 0; i < element_count_; ++i) {
                        output[i] *= scale;
                    }
                }
            }

        private:
            std::vector<std::size_t> lengths_;
            std::size_t batch_count_{};
            std::size_t element_count_{};
            syclfft::direction direction_{syclfft::direction::forward};
            plan_options options_{};
            const syclfft_host_provider_v1 *api_{};
            std::shared_ptr<void> module_;
            void *handle_{};
            std::mutex mutex_;
        };
#endif

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

        struct staged_submission
        {
            sycl::queue &queue;
            std::optional<sycl::event> &last_event;

            template <class Function>
            sycl::event submit(Function &&function)
            {
                // Keep the most recent successful submission even if a later stage throws.
                last_event = queue.submit(std::forward<Function>(function));
                return *last_event;
            }
        };

        template <class Scalar>
        sycl::event enqueue_copy_scaled(
            staged_submission &stages, const complex<Scalar> *input, complex<Scalar> *output,
            std::size_t count, Scalar scale, const std::vector<sycl::event> &dependencies)
        {
            return stages.submit([&](sycl::handler &handler) {
                handler.depends_on(dependencies);
                handler.parallel_for(sycl::range<1>{count}, [=](sycl::id<1> index) {
                    output[index[0]] = input[index[0]] * scale;
                });
            });
        }

        template <class Scalar>
        sycl::event enqueue_mixed_axis(
            staged_submission &stages, const complex<Scalar> *input, complex<Scalar> *output,
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
            sycl::event current = stages.submit([&](sycl::handler &handler) {
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
                current = stages.submit([&](sycl::handler &handler) {
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
            staged_submission &stages, const complex<Scalar> *input, complex<Scalar> *output,
            std::size_t total, std::size_t length, std::size_t stride,
            const complex<Scalar> *twiddles, bool conjugate_twiddles,
            const std::vector<sycl::event> &dependencies)
        {
            unsigned bits = 0;
            for (std::size_t value = length; value > 1; value >>= 1) {
                ++bits;
            }

            sycl::event current = stages.submit([&](sycl::handler &handler) {
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
                current = stages.submit([&](sycl::handler &handler) {
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

        std::uint32_t reverse_bits(std::uint32_t value)
        {
            value = ((value & 0x55555555u) << 1) | ((value >> 1) & 0x55555555u);
            value = ((value & 0x33333333u) << 2) | ((value >> 2) & 0x33333333u);
            value = ((value & 0x0f0f0f0fu) << 4) | ((value >> 4) & 0x0f0f0f0fu);
            value = ((value & 0x00ff00ffu) << 8) | ((value >> 8) & 0x00ff00ffu);
            return (value << 16) | (value >> 16);
        }

        template <class Scalar>
        std::array<complex<Scalar>, 4> local_butterfly4(
            complex<Scalar> a, complex<Scalar> b, complex<Scalar> c, complex<Scalar> d,
            bool backward)
        {
            const auto sum_ab = a + b;
            const auto sum_cd = c + d;
            const auto diff_ab = a - b;
            const auto diff_cd = c - d;
            const auto rotated = backward ? complex<Scalar>{-diff_cd.imag(), diff_cd.real()}
                                          : complex<Scalar>{diff_cd.imag(), -diff_cd.real()};
            return {sum_ab + sum_cd, diff_ab + rotated, sum_ab - sum_cd, diff_ab - rotated};
        }

        template <unsigned Length, class Scalar>
        sycl::event enqueue_local_radix4(
            sycl::queue &queue, const complex<Scalar> *input, complex<Scalar> *output,
            std::size_t batch_count, const complex<Scalar> *twiddles, Scalar scale, bool backward,
            const std::vector<sycl::event> &dependencies)
        {
            // Only the low log2(Length / 4) bits select the first butterfly.
            constexpr unsigned reverse_shift = Length == 1024 ? 24 : Length == 2048 ? 23 : 22;
            return queue.submit([&](sycl::handler &handler) {
                handler.depends_on(dependencies);
                sycl::local_accessor<complex<Scalar>, 1> values{sycl::range<1>{Length}, handler};
                handler.parallel_for(
                    sycl::nd_range<1>{
                        sycl::range<1>{batch_count * local_fft_group_size},
                        sycl::range<1>{local_fft_group_size}},
                    [=](sycl::nd_item<1> item) {
                    const auto lane = static_cast<unsigned>(item.get_local_linear_id());
                    const auto base = item.get_group_linear_id() * Length;
                    // Fuse bit reversal with the initial radix-4 butterfly.
                    for (unsigned i = lane; i < Length / 4; i += local_fft_group_size) {
                        const auto first = (reverse_bits(i) >> reverse_shift) * 4;
                        const auto samples = local_butterfly4(
                            input[base + i],
                            input[base + i + Length / 2],
                            input[base + i + Length / 4],
                            input[base + i + 3 * Length / 4],
                            backward);
                        for (unsigned leg = 0; leg < 4; ++leg) {
                            values[first + leg] = samples[leg];
                        }
                    }
                    item.barrier(sycl::access::fence_space::global_and_local);

                    for (unsigned quarter = 4; quarter <= Length / 4; quarter *= 4) {
                        const unsigned mask = quarter - 1;
                        const unsigned step = Length / (4 * quarter);
                        for (unsigned quad = lane; quad < Length / 4;
                             quad += local_fft_group_size) {
                            const unsigned j = quad & mask;
                            const unsigned first = ((quad & ~mask) << 2) | j;
                            const unsigned twiddle = j * step;
                            const auto samples = local_butterfly4(
                                values[first],
                                values[first + quarter] * twiddles[2 * twiddle],
                                values[first + 2 * quarter] * twiddles[twiddle],
                                values[first + 3 * quarter] * twiddles[3 * twiddle],
                                backward);
                            for (unsigned leg = 0; leg < 4; ++leg) {
                                const auto index = first + leg * quarter;
                                if (4 * quarter == Length) {
                                    output[base + index] = samples[leg] * scale;
                                } else {
                                    values[index] = samples[leg];
                                }
                            }
                        }
                        if (4 * quarter != Length) {
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                    }

                    if constexpr (Length == 2048) {
                        for (unsigned i = lane; i < Length / 2; i += local_fft_group_size) {
                            const auto lhs = values[i];
                            const auto rhs = values[i + Length / 2] * twiddles[i];
                            output[base + i] = (lhs + rhs) * scale;
                            output[base + i + Length / 2] = (lhs - rhs) * scale;
                        }
                    }
                });
            });
        }

        template <class Scalar>
        sycl::event enqueue_gpu_work_group_fft(
            sycl::queue &queue, const complex<Scalar> *input, complex<Scalar> *output,
            unsigned length, std::size_t batch_count, const complex<Scalar> *twiddles, Scalar scale,
            bool backward, const std::vector<sycl::event> &dependencies)
        {
            switch (length) {
            case 1024:
                return enqueue_local_radix4<1024>(
                    queue, input, output, batch_count, twiddles, scale, backward, dependencies);
            case 2048:
                return enqueue_local_radix4<2048>(
                    queue, input, output, batch_count, twiddles, scale, backward, dependencies);
            case 4096:
                return enqueue_local_radix4<4096>(
                    queue, input, output, batch_count, twiddles, scale, backward, dependencies);
            default:
                throw exception(error_code::invalid_state, "Unsupported work-group FFT length");
            }
        }

        template <class Scalar>
        sycl::event enqueue_bluestein_axis(
            staged_submission &stages, const complex<Scalar> *input, complex<Scalar> *first,
            complex<Scalar> *second, std::size_t total, std::size_t length, std::size_t stride,
            std::size_t convolution_length, const complex<Scalar> *chirp,
            const complex<Scalar> *kernel_spectrum, const complex<Scalar> *convolution_twiddles,
            const std::vector<sycl::event> &dependencies)
        {
            const auto line_count = total / length;
            const auto expanded_count = line_count * convolution_length;

            auto current = stages.submit([&](sycl::handler &handler) {
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
                stages,
                first,
                second,
                expanded_count,
                convolution_length,
                1,
                convolution_twiddles,
                false,
                {current});
            auto prior = current;
            current = stages.submit([&](sycl::handler &handler) {
                handler.depends_on(prior);
                handler.parallel_for(sycl::range<1>{expanded_count}, [=](sycl::id<1> item) {
                    const auto index = static_cast<std::size_t>(item[0]);
                    second[index] *= kernel_spectrum[index % convolution_length];
                });
            });
            current = enqueue_radix2_axis(
                stages,
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
            current = stages.submit([&](sycl::handler &handler) {
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

        template <class Scalar>
        class portable_plan
        {
            using complex_type = complex<Scalar>;

            struct free_usm
            {
                const sycl::queue *queue{};

                void operator()(complex_type *pointer) const
                {
                    sycl::free(pointer, *queue);
                }
            };

            using device_buffer = std::unique_ptr<complex_type, free_usm>;

            struct axis_plan
            {
                std::size_t length{};
                std::size_t stride{};
                std::size_t convolution_length{};
                std::vector<std::size_t> factors;
                device_buffer twiddles;
                device_buffer chirp;
                device_buffer kernel_spectrum;
            };

        public:
            portable_plan(
                sycl::queue &queue, const std::vector<std::size_t> &lengths,
                std::size_t element_count, std::size_t batch_count, direction dir)
              : queue_(queue)
              , rank_(lengths.size())
              , element_count_(element_count)
              , batch_count_(batch_count)
              , direction_(dir)
            {
                if (!queue_.get_device().has(sycl::aspect::usm_device_allocations)) {
                    throw exception(
                        error_code::provider_unavailable,
                        "Portable provider requires device USM allocations");
                }
                execution_ = select_radix2_execution<Scalar>(queue_, lengths);
                prepare_axes(lengths);
                try {
                    const double sign = direction_ == direction::forward ? -1.0 : 1.0;
                    for (std::size_t i = 0; i < rank_; ++i) {
                        auto &axis = axes_[i];
                        if (axis.convolution_length) {
                            prepare_bluestein(axis, sign);
                        } else {
                            axis.twiddles = upload_table(make_twiddles(axis.length, sign));
                        }
                    }
                    if (scratch_elements_ != 0) {
                        scratch_a_ = allocate_device(scratch_elements_);
                        scratch_b_ = allocate_device(scratch_elements_);
                    }
                } catch (...) {
                    throw exception(
                        error_code::planning_failed,
                        "Unable to allocate or initialize portable FFT scratch memory");
                }
            }

            std::size_t scratch_size_bytes() const noexcept
            {
                return 2 * scratch_elements_ * sizeof(complex_type);
            }

            sycl::event enqueue(
                const complex_type *input, complex_type *output, Scalar scale,
                const std::vector<sycl::event> &dependencies,
                std::optional<sycl::event> &last_event)
            {
                const auto &first_axis = axes_.front();
                switch (execution_) {
                case radix2_execution::cpu_batch:
                    last_event = detail::enqueue_cpu_radix2(
                        queue_,
                        input,
                        output,
                        first_axis.length,
                        batch_count_,
                        first_axis.twiddles.get(),
                        scale,
                        dependencies);
                    return *last_event;
                case radix2_execution::gpu_work_group:
                    last_event = enqueue_gpu_work_group_fft(
                        queue_,
                        input,
                        output,
                        static_cast<unsigned>(first_axis.length),
                        batch_count_,
                        first_axis.twiddles.get(),
                        scale,
                        direction_ == direction::backward,
                        dependencies);
                    return *last_event;
                case radix2_execution::staged:
                    return enqueue_staged(input, output, scale, dependencies, last_event);
                }
                throw exception(
                    error_code::invalid_state, "Invalid portable FFT execution strategy");
            }

        private:
            void prepare_axes(const std::vector<std::size_t> &lengths)
            {
                constexpr auto max_scratch_elements =
                    std::numeric_limits<std::size_t>::max() / sizeof(complex_type) / 2;
                scratch_elements_ = execution_ == radix2_execution::staged ? element_count_ : 0;
                if (scratch_elements_ > max_scratch_elements) {
                    throw exception(
                        error_code::invalid_argument, "FFT scratch size overflows size_t");
                }
                for (std::size_t i = 0; i < rank_; ++i) {
                    auto &axis = axes_[i];
                    axis.length = lengths[i];
                    axis.stride = axis_stride(lengths, i);
                    if (is_power_of_two(axis.length)) {
                        continue;
                    }
                    axis.factors = supported_factors(axis.length);
                    if (!axis.factors.empty()) {
                        continue;
                    }
                    axis.convolution_length = next_power_of_two(2 * axis.length - 1);
                    const auto line_count = element_count_ / axis.length;
                    if (line_count > max_scratch_elements / axis.convolution_length) {
                        throw exception(
                            error_code::invalid_argument,
                            "Bluestein scratch size overflows size_t");
                    }
                    scratch_elements_ =
                        std::max(scratch_elements_, line_count * axis.convolution_length);
                }
            }

            device_buffer allocate_device(std::size_t count)
            {
                auto *pointer = sycl::malloc_device<complex_type>(count, queue_);
                if (!pointer) {
                    throw std::bad_alloc{};
                }
                return device_buffer{pointer, free_usm{&queue_}};
            }

            device_buffer upload_table(const std::vector<complex_type> &host)
            {
                auto device = allocate_device(host.size());
                queue_.memcpy(device.get(), host.data(), host.size() * sizeof(complex_type))
                    .wait_and_throw();
                return device;
            }

            static std::vector<complex_type> make_twiddles(std::size_t length, double sign)
            {
                std::vector<complex_type> twiddles(length);
                for (std::size_t index = 0; index < length; ++index) {
                    const auto angle =
                        sign * 2.0 * pi * static_cast<double>(index) / static_cast<double>(length);
                    twiddles[index] = {
                        static_cast<Scalar>(std::cos(angle)), static_cast<Scalar>(std::sin(angle))};
                }
                return twiddles;
            }

            void prepare_bluestein(axis_plan &axis, double sign)
            {
                const auto length = axis.length;
                const auto convolution_length = axis.convolution_length;
                std::vector<complex_type> kernel(convolution_length);
                std::vector<complex_type> chirp(length);
                const auto period = 2 * length;
                std::size_t phase_index = 0;
                for (std::size_t index = 0; index < length; ++index) {
                    // Keep index squared modulo 2N using successive odd differences.
                    const auto angle =
                        -sign * pi * static_cast<double>(phase_index) / static_cast<double>(length);
                    const complex_type value{
                        static_cast<Scalar>(std::cos(angle)), static_cast<Scalar>(std::sin(angle))};
                    kernel[index] = value;
                    if (index != 0) {
                        kernel[convolution_length - index] = value;
                    }
                    chirp[index] = {value.real(), -value.imag()};
                    phase_index = (phase_index + 2 * index + 1) % period;
                }
                const auto twiddles = make_twiddles(convolution_length, -1.0);
                detail::fft_radix2_line(
                    kernel.data(), kernel.data(), convolution_length, twiddles.data(), Scalar{1});
                axis.chirp = upload_table(chirp);
                axis.kernel_spectrum = upload_table(kernel);
                axis.twiddles = upload_table(twiddles);
            }

            sycl::event enqueue_staged(
                const complex_type *input, complex_type *output, Scalar scale,
                const std::vector<sycl::event> &dependencies,
                std::optional<sycl::event> &last_event)
            {
                staged_submission stages{queue_, last_event};
                auto axis_dependencies = dependencies;
                const auto *current_input = input;
                for (std::size_t i = 0; i < rank_; ++i) {
                    const auto &axis = axes_[i];
                    auto *first =
                        current_input == scratch_a_.get() ? scratch_b_.get() : scratch_a_.get();
                    if (axis.convolution_length) {
                        auto *second =
                            first == scratch_a_.get() ? scratch_b_.get() : scratch_a_.get();
                        const auto event = enqueue_bluestein_axis(
                            stages,
                            current_input,
                            first,
                            second,
                            element_count_,
                            axis.length,
                            axis.stride,
                            axis.convolution_length,
                            axis.chirp.get(),
                            axis.kernel_spectrum.get(),
                            axis.twiddles.get(),
                            axis_dependencies);
                        axis_dependencies.assign(1, event);
                        current_input = second;
                    } else {
                        const auto event = axis.factors.empty() ? enqueue_radix2_axis(
                                                                      stages,
                                                                      current_input,
                                                                      first,
                                                                      element_count_,
                                                                      axis.length,
                                                                      axis.stride,
                                                                      axis.twiddles.get(),
                                                                      false,
                                                                      axis_dependencies)
                                                                : enqueue_mixed_axis(
                                                                      stages,
                                                                      current_input,
                                                                      first,
                                                                      element_count_,
                                                                      axis.length,
                                                                      axis.stride,
                                                                      axis.twiddles.get(),
                                                                      axis.factors,
                                                                      axis_dependencies);
                        axis_dependencies.assign(1, event);
                        current_input = first;
                    }
                }
                return enqueue_copy_scaled(
                    stages, current_input, output, element_count_, scale, axis_dependencies);
            }

            sycl::queue &queue_;
            std::size_t rank_;
            std::size_t element_count_;
            std::size_t batch_count_;
            direction direction_;
            radix2_execution execution_{radix2_execution::staged};
            std::size_t scratch_elements_{};
            std::array<axis_plan, 3> axes_;
            device_buffer scratch_a_;
            device_buffer scratch_b_;
        };

        struct loaded_cuda_provider
        {
            const detail::cufft_provider_v2 *api{};
            std::shared_ptr<void> module;
        };

        void retain_cuda_module(const std::shared_ptr<void> &module)
        {
            struct process_modules
            {
                std::mutex mutex;
                std::vector<std::shared_ptr<void>> handles;
            };

            // The SYCL runtime can destroy native callbacks after their events complete,
            // including during runtime teardown. Their code must stay mapped until process exit.
            static auto &modules = *new process_modules;
            std::scoped_lock lock(modules.mutex);
            auto &handles = modules.handles;
            if (std::find(handles.begin(), handles.end(), module) == handles.end()) {
                handles.push_back(module);
            }
        }

        bool cuda_provider_built(provider id)
        {
#if defined(SYCLFFT_HAS_CUFFT_PROVIDER)
            if (id == provider::cufft) {
                return true;
            }
#endif
#if defined(SYCLFFT_HAS_VKFFT_PROVIDER)
            if (id == provider::vkfft) {
                return true;
            }
#endif
            (void)id;
            return false;
        }

        loaded_cuda_provider try_load_cuda(
            const sycl::queue &queue, provider id, bool required, std::string *reason = nullptr)
        {
            const std::string name = detail::provider_name(id);
            try {
                if (!detail::is_cuda_queue(queue)) {
                    throw exception(
                        error_code::provider_unavailable, name + " requires a CUDA queue");
                }
                if (!cuda_provider_built(id)) {
                    throw exception(
                        error_code::provider_unavailable, name + " provider was not built");
                }
                auto symbol = detail::load_provider_symbol(
                    "syclfft_provider_" + name, "syclfft_get_" + name + "_provider_v2");
                auto getter = reinterpret_cast<detail::get_cufft_provider_v2_fn>(symbol.symbol);
                const auto *api = getter();
                if (!api || api->abi_version != SYCLFFT_CUFFT_PROVIDER_ABI_VERSION) {
                    throw exception(error_code::plugin_error, name + " provider ABI mismatch");
                }
                std::array<char, 1024> error{};
                if (!api->supports(queue, error.data(), error.size())) {
                    throw exception(
                        error_code::provider_unavailable,
                        error[0] ? error.data() : name + " provider rejected the queue");
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
        }

    } // namespace

    template <class Scalar>
    class plan<Scalar>::impl
    {
        sycl::queue queue_;

    public:
        impl(
            sycl::queue queue, std::vector<std::size_t> lengths, std::size_t batch_count,
            syclfft::direction dir, plan_options opts)
          : queue_(std::move(queue))
          , lengths_(std::move(lengths))
          , batch_count_(batch_count)
          , direction_(dir)
          , element_count_(detail::checked_element_count(lengths_, batch_count_))
          , transform_size_(detail::transform_size(lengths_))
          , options_(opts)
        {
            constexpr auto max_complex_elements =
                std::numeric_limits<std::size_t>::max() / sizeof(complex_type);
            if (element_count_ > max_complex_elements) {
                throw exception(error_code::invalid_argument, "FFT buffer size overflows size_t");
            }
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

            initialize_provider();
            if (selected_provider_ == provider::portable_sycl) {
                portable_.emplace(queue_, lengths_, element_count_, batch_count_, direction_);
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
            if (cuda_handle_) {
                cuda_.api->destroy(cuda_handle_);
            }
        }

        sycl::event execute(
            const complex_type *input, complex_type *output,
            syclfft::span<const sycl::event> dependencies, bool one_pointer)
        {
            validate_execution(input, output, one_pointer);
            std::scoped_lock lock(mutex_);
            std::vector<sycl::event> all_dependencies(dependencies.begin(), dependencies.end());
            if (last_event_) {
                all_dependencies.push_back(*last_event_);
            }

#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
            if (fftw_plan_) {
                last_event_ = execute_fftw(input, output, one_pointer, all_dependencies);
                return *last_event_;
            }
#endif
            const auto scale = static_cast<Scalar>(
                detail::normalization_scale(options_.normalization, direction_, transform_size_));
            if (cuda_.api) {
                return execute_cuda(input, output, scale, all_dependencies);
            }
            return portable_->enqueue(input, output, scale, all_dependencies, last_event_);
        }

        std::vector<std::size_t> lengths_;
        std::size_t batch_count_{};
        syclfft::direction direction_{syclfft::direction::forward};
        provider selected_provider_{provider::portable_sycl};

        std::size_t scratch_size_bytes() const noexcept
        {
            return portable_ ? portable_->scratch_size_bytes() : 0;
        }

    private:
        void initialize_provider()
        {
            switch (options_.preferred_provider) {
            case provider::automatic:
                if (detail::is_cuda_queue(queue_)) {
                    for (const auto id : {provider::cufft, provider::vkfft}) {
                        if (cuda_provider_built(id) && create_cuda_plan(id, false)) {
                            return;
                        }
                    }
                }
                return;
            case provider::portable_sycl:
                return;
            case provider::cufft:
            case provider::vkfft:
                create_cuda_plan(options_.preferred_provider, true);
                return;
            case provider::fftw:
#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
                if (!queue_.get_device().is_cpu()) {
                    throw exception(
                        error_code::provider_unavailable,
                        "FFTW can only be selected for a SYCL CPU queue");
                }
                fftw_plan_.emplace(lengths_, batch_count_, direction_, options_);
                selected_provider_ = provider::fftw;
                return;
#else
                throw exception(
                    error_code::provider_unavailable,
                    "FFTW provider was not built with the SYCL component");
#endif
            default:
                throw exception(
                    error_code::provider_unavailable,
                    std::string(detail::provider_name(options_.preferred_provider))
                        + " provider is not implemented in v0.1");
            }
        }

        bool create_cuda_plan(provider id, bool required)
        {
            auto backend = try_load_cuda(queue_, id, required);
            if (!backend.api) {
                return false;
            }
            retain_cuda_module(backend.module);
            detail::cufft_plan_config_v2 config{};
            config.double_precision = std::is_same_v<Scalar, double>;
            config.transform_direction = direction_;
            config.transform_placement = options_.placement;
            config.rank = static_cast<std::uint32_t>(lengths_.size());
            std::copy(lengths_.begin(), lengths_.end(), config.lengths.begin());
            config.batch_count = batch_count_;
            std::array<char, 1024> error{};
            auto *handle = backend.api->create(queue_, config, error.data(), error.size());
            if (!handle) {
                if (required) {
                    throw exception(
                        error_code::planning_failed,
                        error[0] ? error.data() : "CUDA FFT plan creation failed");
                }
                return false;
            }
            selected_provider_ = id;
            cuda_ = std::move(backend);
            cuda_handle_ = handle;
            return true;
        }

        void validate_execution(
            const complex_type *input, complex_type *output, bool one_pointer) const
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
            const auto input_address = reinterpret_cast<std::uintptr_t>(input);
            const auto output_address = reinterpret_cast<std::uintptr_t>(output);
            if (options_.placement == placement::out_of_place) {
                const auto distance = input_address < output_address
                                          ? output_address - input_address
                                          : input_address - output_address;
                if (distance < element_count_ * sizeof(complex_type)) {
                    throw exception(
                        error_code::invalid_argument,
                        "Out-of-place FFT input and output must not overlap");
                }
            }
            if (selected_provider_ == provider::vkfft && transform_size_ > 1
                && (input_address % sizeof(complex_type) != 0
                    || output_address % sizeof(complex_type) != 0)) {
                throw exception(
                    error_code::invalid_pointer,
                    "VkFFT input and output require " + std::to_string(sizeof(complex_type))
                        + "-byte alignment");
            }
        }

#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
        sycl::event execute_fftw(
            const complex_type *input, complex_type *output, bool one_pointer,
            const std::vector<sycl::event> &dependencies)
        {
            for (auto dependency : dependencies) {
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
            fftw_plan_->execute(input, output, one_pointer);
            return queue_.submit([](sycl::handler &handler) { handler.single_task([]{}); });
        }
#endif

        sycl::event execute_cuda(
            const complex_type *input, complex_type *output, Scalar scale,
            const std::vector<sycl::event> &dependencies)
        {
            std::array<char, 1024> error{};
            auto current = cuda_.api->execute(
                cuda_handle_, input, output, dependencies, error.data(), error.size());
            if (error[0]) {
                throw exception(error_code::execution_failed, error.data());
            }
            last_event_ = std::move(current);
            if (scale == Scalar{1}) {
                return *last_event_;
            }
            staged_submission stages{queue_, last_event_};
            return enqueue_copy_scaled(
                stages, output, output, element_count_, scale, {*last_event_});
        }

        std::size_t element_count_{};
        std::size_t transform_size_{};
        plan_options options_{};
        std::optional<portable_plan<Scalar>> portable_;
        std::optional<sycl::event> last_event_;
        std::mutex mutex_;
        loaded_cuda_provider cuda_;
        void *cuda_handle_{};
#if defined(SYCLFFT_HAS_FFTW_PROVIDER)
        std::optional<fftw_plan_adapter<Scalar>> fftw_plan_;
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
        return impl_->execute(inout, inout, syclfft::span<const sycl::event>{&dependency, 1}, true);
    }

    template <class Scalar>
    sycl::event
        plan<Scalar>::execute(complex_type *inout, syclfft::span<const sycl::event> dependencies)
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
        return impl_->execute(
            input, output, syclfft::span<const sycl::event>{&dependency, 1}, false);
    }

    template <class Scalar>
    sycl::event plan<Scalar>::execute(
        const complex_type *input, complex_type *output,
        syclfft::span<const sycl::event> dependencies)
    {
        if (!impl_) {
            throw exception(error_code::invalid_state, "Cannot execute a moved-from plan");
        }
        return impl_->execute(input, output, dependencies, false);
    }

    template <class Scalar>
    syclfft::span<const std::size_t> plan<Scalar>::shape() const noexcept
    {
        return impl_ ? syclfft::span<const std::size_t>{impl_->lengths_}
                     : syclfft::span<const std::size_t>{};
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
        return impl_ ? impl_->scratch_size_bytes() : 0;
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
        for (const auto id : {provider::cufft, provider::vkfft}) {
            std::string reason;
            const auto loaded = try_load_cuda(queue, id, false, &reason);
            result.push_back(
                {id,
                 detail::provider_name(id),
                 cuda_provider_built(id),
                 loaded.api != nullptr,
                 std::move(reason)});
        }
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
