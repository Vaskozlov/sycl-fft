#include "test_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <syclfft/span.hpp>
#include <syclfft/syclfft.hpp>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

    /* C++17 replacement for std::bit_cast (C++20) */
    template <class To, class From>
    To portability_bit_cast(const From &src) noexcept
    {
        static_assert(sizeof(To) == sizeof(From), "bit_cast size mismatch");
        static_assert(
            std::is_trivially_copyable<From>::value, "bit_cast requires trivially copyable");
        static_assert(
            std::is_trivially_copyable<To>::value, "bit_cast requires trivially copyable");
        To dst{};
        std::memcpy(&dst, &src, sizeof(To));
        return dst;
    }

    template <class T>
    class usm_buffer
    {
    public:
        usm_buffer(std::size_t count, sycl::queue &queue, bool require_host_accessible = false)
          : queue_(&queue)
          , count_(count)
        {
            const auto device = queue.get_device();
            if (!require_host_accessible && device.has(sycl::aspect::usm_device_allocations)) {
                pointer_ = sycl::malloc_device<T>(count, queue);
            } else if (device.has(sycl::aspect::usm_host_allocations)) {
                pointer_ = sycl::malloc_host<T>(count, queue);
            } else if (device.has(sycl::aspect::usm_shared_allocations)) {
                pointer_ = sycl::malloc_shared<T>(count, queue);
            }
            if (!pointer_) {
                throw std::runtime_error(
                    require_host_accessible ? "test device has no host-accessible USM"
                                            : "test device has no usable USM allocation");
            }
        }

        ~usm_buffer()
        {
            sycl::free(pointer_, *queue_);
        }

        usm_buffer(const usm_buffer &) = delete;
        usm_buffer &operator=(const usm_buffer &) = delete;

        [[nodiscard]] T *get() const noexcept
        {
            return pointer_;
        }

        void write(syclfft::span<const T> values)
        {
            test_utils::check(values.size() == count_, "USM upload size mismatch");
            queue_->memcpy(pointer_, values.data(), count_ * sizeof(T)).wait_and_throw();
        }

        [[nodiscard]] std::vector<T> read() const
        {
            std::vector<T> values(count_);
            queue_->memcpy(values.data(), pointer_, count_ * sizeof(T)).wait_and_throw();
            return values;
        }

    private:
        sycl::queue *queue_{};
        std::size_t count_{};
        T *pointer_{};
    };

    template <class Scalar>
    std::vector<std::complex<Scalar>> direct_dft(
        const std::vector<std::complex<Scalar>> &input, syclfft::direction direction,
        syclfft::normalization normalization)
    {
        const auto count = input.size();
        std::vector<std::complex<Scalar>> output(count);
        const auto sign = direction == syclfft::direction::forward ? Scalar{-1} : Scalar{1};
        const auto pi = std::acos(Scalar{-1});
        for (std::size_t k = 0; k < count; ++k) {
            for (std::size_t n = 0; n < count; ++n) {
                const auto angle =
                    sign * Scalar{2} * pi * static_cast<Scalar>(k * n) / static_cast<Scalar>(count);
                output[k] += input[n] * std::complex<Scalar>{std::cos(angle), std::sin(angle)};
            }
        }
        Scalar scale = 1;
        if (normalization == syclfft::normalization::orthogonal) {
            scale = Scalar{1} / std::sqrt(static_cast<Scalar>(count));
        } else if (
            (normalization == syclfft::normalization::forward
             && direction == syclfft::direction::forward)
            || (normalization == syclfft::normalization::backward
                && direction == syclfft::direction::backward)) {
            scale = Scalar{1} / static_cast<Scalar>(count);
        }
        for (auto &value : output) {
            value *= scale;
        }
        return output;
    }

    template <class Scalar>
    std::vector<std::complex<Scalar>> reference_many(
        std::vector<std::complex<Scalar>> values, syclfft::span<const std::size_t> shape,
        std::size_t batch_count, syclfft::direction direction)
    {
        std::size_t transform_size = 1;
        for (const auto n : shape) {
            transform_size *= n;
        }
        const auto sign = direction == syclfft::direction::forward ? Scalar{-1} : Scalar{1};
        const auto pi = std::acos(Scalar{-1});
        for (std::size_t axis = 0; axis < shape.size(); ++axis) {
            auto output = values;
            std::size_t stride = 1;
            for (std::size_t i = axis + 1; i < shape.size(); ++i) {
                stride *= shape[i];
            }
            const auto length = shape[axis];
            for (std::size_t batch = 0; batch < batch_count; ++batch) {
                const auto offset = batch * transform_size;
                for (std::size_t linear = 0; linear < transform_size; ++linear) {
                    const auto coordinate = (linear / stride) % length;
                    const auto base = linear - coordinate * stride;
                    std::complex<Scalar> sum{};
                    for (std::size_t k = 0; k < length; ++k) {
                        const auto angle = sign * Scalar{2} * pi
                                           * static_cast<Scalar>(coordinate * k)
                                           / static_cast<Scalar>(length);
                        sum += values[offset + base + k * stride]
                               * std::complex<Scalar>{std::cos(angle), std::sin(angle)};
                    }
                    output[offset + linear] = sum;
                }
            }
            values = std::move(output);
        }
        return values;
    }

    template <class Scalar>
    void run_many(
        sycl::queue &queue, std::vector<std::size_t> shape, std::size_t batch_count,
        syclfft::provider provider = syclfft::provider::portable_sycl,
        syclfft::direction direction = syclfft::direction::forward)
    {
        std::size_t count = batch_count;
        for (const auto n : shape) {
            count *= n;
        }
        std::vector<std::complex<Scalar>> source(count);
        for (std::size_t i = 0; i < count; ++i) {
            source[i] = {
                static_cast<Scalar>((i * 11 + 2) % 17) / Scalar{8},
                static_cast<Scalar>((i * 3 + 1) % 13) / Scalar{7}};
        }
        const auto expected = reference_many(source, shape, batch_count, direction);
        std::vector<syclfft::complex<Scalar>> device_source(count);
        for (std::size_t i = 0; i < count; ++i) {
            device_source[i] = {source[i].real(), source[i].imag()};
        }
        usm_buffer<syclfft::complex<Scalar>> input(count, queue);
        usm_buffer<syclfft::complex<Scalar>> output(count, queue);
        input.write(device_source);
        auto fft = syclfft::plan_many_dft<Scalar>(
            queue,
            shape,
            batch_count,
            direction,
            test_utils::make_plan_options(
                syclfft::placement::out_of_place, syclfft::normalization::none, provider));
        fft.execute(input.get(), output.get()).wait_and_throw();
        const auto actual_values = output.read();
        const auto tolerance = std::is_same_v<Scalar, float> ? Scalar{8e-4} : Scalar{5e-11};
        for (std::size_t i = 0; i < count; ++i) {
            const std::complex<Scalar> actual{actual_values[i].real(), actual_values[i].imag()};
            test_utils::check(
                std::abs(actual - expected[i]) <= tolerance * (Scalar{1} + std::abs(expected[i])),
                "multidimensional or batch mismatch at element " + std::to_string(i));
        }
        test_utils::check(input.read() == device_source, "out-of-place batch modified input");
    }

    template <class Scalar>
    void run_1d(
        sycl::queue &queue, std::size_t count, syclfft::direction direction,
        syclfft::normalization normalization, syclfft::placement placement,
        syclfft::provider provider = syclfft::provider::portable_sycl)
    {
        std::vector<std::complex<Scalar>> source(count);
        for (std::size_t i = 0; i < count; ++i) {
            source[i] = {
                static_cast<Scalar>((i * 5 + 1) % 9) / Scalar{4},
                static_cast<Scalar>((i * 3 + 2) % 7) / Scalar{5}};
        }
        const auto expected = direct_dft(source, direction, normalization);
        const bool out_of_place = placement == syclfft::placement::out_of_place;
        std::vector<syclfft::complex<Scalar>> device_source(out_of_place ? 2 * count : count);
        for (std::size_t i = 0; i < count; ++i) {
            device_source[i] = {source[i].real(), source[i].imag()};
            if (out_of_place) {
                device_source[count + i] = {-source[i].real(), -source[i].imag()};
            }
        }
        usm_buffer<syclfft::complex<Scalar>> input(device_source.size(), queue);
        std::unique_ptr<usm_buffer<syclfft::complex<Scalar>>> output_storage;
        std::unique_ptr<usm_buffer<syclfft::complex<Scalar>>> repeated_storage;
        if (placement == syclfft::placement::out_of_place) {
            output_storage = std::make_unique<usm_buffer<syclfft::complex<Scalar>>>(count, queue);
            repeated_storage = std::make_unique<usm_buffer<syclfft::complex<Scalar>>>(count, queue);
        }
        auto *output = output_storage ? output_storage->get() : input.get();
        auto *repeated_output = repeated_storage ? repeated_storage->get() : nullptr;
        auto fft = syclfft::plan_dft_1d<Scalar>(
            queue,
            count,
            direction,
            test_utils::make_plan_options(placement, normalization, provider));
        auto dependency = queue.memcpy(
            input.get(),
            device_source.data(),
            device_source.size() * sizeof(syclfft::complex<Scalar>));
        auto event = placement == syclfft::placement::in_place
                         ? fft.execute(input.get(), dependency)
                         : fft.execute(input.get(), output, dependency);
        sycl::event repeated;
        if (repeated_output) {
            repeated = fft.execute(input.get() + count, repeated_output);
        }
        event.wait_and_throw();
        if (repeated_output) {
            repeated.wait_and_throw();
        }
        const auto actual_values = output_storage ? output_storage->read() : input.read();
        const auto repeated_values =
            repeated_storage ? repeated_storage->read() : std::vector<syclfft::complex<Scalar>>{};
        const auto tolerance = std::is_same_v<Scalar, float> ? Scalar{4e-4} : Scalar{2e-11};
        for (std::size_t i = 0; i < count; ++i) {
            const std::complex<Scalar> actual{actual_values[i].real(), actual_values[i].imag()};
            test_utils::check(
                std::abs(actual - expected[i]) <= tolerance * (Scalar{1} + std::abs(expected[i])),
                "1D transform mismatch at element " + std::to_string(i));
            if (repeated_output) {
                const std::complex<Scalar> repeated_actual{
                    repeated_values[i].real(), repeated_values[i].imag()};
                test_utils::check(
                    std::abs(repeated_actual + expected[i])
                        <= tolerance * (Scalar{1} + std::abs(expected[i])),
                    "serialized repeated execution mismatch");
            }
        }
        test_utils::check(fft.selected_provider() == provider, "wrong explicit provider");
        if (placement == syclfft::placement::out_of_place) {
            test_utils::check(
                input.read() == device_source, "out-of-place transform modified input");
        }
    }

    void run_prime_length_tone(sycl::queue &queue, syclfft::provider provider)
    {
        constexpr std::size_t count = 65537;
        constexpr std::size_t frequency = 17;
        const double two_pi = 2.0 * std::acos(-1.0);
        std::vector<syclfft::complex<float>> source(count);
        for (std::size_t i = 0; i < count; ++i) {
            const auto phase = two_pi * static_cast<double>((i * frequency) % count) / count;
            source[i] = {static_cast<float>(std::cos(phase)), static_cast<float>(std::sin(phase))};
        }
        usm_buffer<syclfft::complex<float>> input(count, queue);
        usm_buffer<syclfft::complex<float>> output(count, queue);
        input.write(source);
        auto fft = syclfft::plan_dft_1d<float>(
            queue,
            count,
            syclfft::direction::forward,
            test_utils::make_plan_options(
                syclfft::placement::out_of_place, syclfft::normalization::forward, provider));
        if (provider == syclfft::provider::portable_sycl) {
            constexpr std::size_t convolution_length = 262144;
            test_utils::check(
                fft.scratch_size_bytes()
                    == 2 * convolution_length * sizeof(syclfft::complex<float>),
                "wrong Bluestein scratch size");
        }
        fft.execute(input.get(), output.get()).wait_and_throw();
        const auto actual = output.read();
        double squared_error = 0;
        for (std::size_t i = 0; i < count; ++i) {
            const std::complex<double> bin{actual[i].real(), actual[i].imag()};
            squared_error += std::norm(bin - (i == frequency ? 1.0 : 0.0));
        }
        const auto error = std::sqrt(squared_error);
        std::cout << syclfft::detail::provider_name(provider)
                  << " prime-length tone relative L2 error: " << error << '\n';
        test_utils::check(
            error <= 5e-6, "prime-length tone relative L2 error: " + std::to_string(error));
    }

    template <class Scalar>
    void run_batched_tones(sycl::queue &queue, std::size_t length, syclfft::provider provider)
    {
        using complex_type = syclfft::complex<Scalar>;
        const std::array<std::size_t, 3> frequencies{1, length / 4 + 1, length - 1};
        constexpr std::array<std::complex<double>, 3> amplitudes{
            std::complex<double>{1.0,    -0.25},
            {-0.375, 0.75 },
            {0.125,  2.0  }
        };
        const auto count = length * frequencies.size();
        const auto two_pi = 2.0 * std::acos(-1.0);
        std::vector<complex_type> source(count);
        for (std::size_t batch = 0; batch < frequencies.size(); ++batch) {
            for (std::size_t i = 0; i < length; ++i) {
                const auto phase =
                    two_pi * static_cast<double>((i * frequencies[batch]) % length) / length;
                const auto sample = amplitudes[batch] * std::polar(1.0, phase);
                source[batch * length + i] = complex_type{
                    static_cast<Scalar>(sample.real()), static_cast<Scalar>(sample.imag())};
            }
        }

        usm_buffer<complex_type> input(count, queue);
        usm_buffer<complex_type> output(count, queue);
        input.write(source);
        sycl::queue consumer_queue{queue.get_context(), queue.get_device()};
        std::vector<complex_type> actual(count);
        const auto consume = [&](const complex_type *data, const sycl::event &done) {
            consumer_queue.memcpy(actual.data(), data, count * sizeof(complex_type), done)
                .wait_and_throw();
        };
        for (const auto direction : {syclfft::direction::forward, syclfft::direction::backward}) {
            const bool forward = direction == syclfft::direction::forward;
            const auto normalization =
                forward ? syclfft::normalization::none : syclfft::normalization::orthogonal;
            const auto placement =
                forward ? syclfft::placement::out_of_place : syclfft::placement::in_place;
            sycl::event done;
            {
                auto fft = syclfft::plan_many_dft<Scalar>(
                    queue,
                    {length},
                    frequencies.size(),
                    direction,
                    test_utils::make_plan_options(placement, normalization, provider));
                done = forward ? fft.execute(input.get(), output.get()) : fft.execute(input.get());
                if (forward) {
                    consume(output.get(), done);
                }
            }
            if (!forward) {
                consume(input.get(), done);
            }
            const double peak_scale = forward ? length : std::sqrt(static_cast<double>(length));
            for (std::size_t batch = 0; batch < frequencies.size(); ++batch) {
                const auto frequency = forward ? frequencies[batch] : length - frequencies[batch];
                const auto peak = amplitudes[batch] * peak_scale;
                double squared_error = 0;
                for (std::size_t i = 0; i < length; ++i) {
                    const auto value = actual[batch * length + i];
                    const std::complex<double> bin{value.real(), value.imag()};
                    squared_error += std::norm(bin - (i == frequency ? peak : 0.0));
                }
                const auto error = std::sqrt(squared_error / std::norm(peak));
                test_utils::check(
                    error <= (std::is_same_v<Scalar, float> ? 5e-6 : 5e-12),
                    std::string(syclfft::detail::provider_name(provider)) + " "
                        + std::to_string(length) + "-point " + (forward ? "forward" : "backward")
                        + " tone mismatch in batch " + std::to_string(batch) + ": "
                        + std::to_string(error));
            }
        }
    }

    void check_plan_size_overflow(sycl::queue &queue)
    {
        constexpr auto max_elements =
            std::numeric_limits<std::size_t>::max() / sizeof(syclfft::complex<float>);
        const auto portable_options = test_utils::make_plan_options(
            syclfft::placement::out_of_place,
            syclfft::normalization::none,
            syclfft::provider::portable_sycl);
        test_utils::expect_error(syclfft::error_code::invalid_argument, [&] {
            (void)syclfft::plan_many_dft<float>(
                queue, {1}, max_elements + 1, syclfft::direction::forward);
        });
        test_utils::expect_error(syclfft::error_code::invalid_argument, [&] {
            (void)syclfft::plan_many_dft<float>(
                queue, {1}, max_elements / 2 + 1, syclfft::direction::forward, portable_options);
        });
        constexpr std::size_t length = 11;
        constexpr std::size_t convolution_length = 32;
        test_utils::expect_error(syclfft::error_code::invalid_argument, [&] {
            (void)syclfft::plan_many_dft<float>(
                queue,
                {length},
                max_elements / (2 * convolution_length) + 1,
                syclfft::direction::forward,
                portable_options);
        });
    }

    void check_out_of_place_ranges(sycl::queue &queue)
    {
        using complex_type = syclfft::complex<float>;
        constexpr std::size_t length = 8;
        constexpr std::size_t count = 2 * length;
        usm_buffer<complex_type> storage(2 * count, queue);
        auto * const start = storage.get();
        auto fft = syclfft::plan_many_dft<float>(queue, {length}, 2, syclfft::direction::forward);
        test_utils::expect_error(
            syclfft::error_code::invalid_argument, [&] { fft.execute(start, start); });
        test_utils::expect_error(
            syclfft::error_code::invalid_argument, [&] { fft.execute(start, start + length); });
        test_utils::expect_error(
            syclfft::error_code::invalid_argument, [&] { fft.execute(start + length, start); });

        std::vector<complex_type> source(2 * count);
        source[0] = {1.0f, 0.5f};
        source[length] = {-0.25f, 1.0f};
        storage.write(source);
        fft.execute(start, start + count).wait_and_throw();
        const auto actual = storage.read();
        for (std::size_t i = 0; i < count; ++i) {
            test_utils::check(
                sycl::ext::cplx::abs(actual[count + i] - source[i - i % length]) <= 1e-6f,
                "FFT with adjacent disjoint buffers produced the wrong spectrum");
        }
    }

    template <class Scalar>
    void check_vkfft_pointer_alignment(sycl::queue &queue)
    {
        using complex_type = syclfft::complex<Scalar>;
        constexpr std::size_t length = 8;
        usm_buffer<Scalar> storage(2 * length + 1, queue);
        usm_buffer<complex_type> aligned(length, queue);
        auto *scalar_aligned = storage.get();
        if (reinterpret_cast<std::uintptr_t>(scalar_aligned) % sizeof(complex_type) == 0) {
            ++scalar_aligned;
        }
        auto *weakly_aligned = reinterpret_cast<complex_type *>(scalar_aligned);
        test_utils::check(
            reinterpret_cast<std::uintptr_t>(weakly_aligned) % alignof(complex_type) == 0
                && reinterpret_cast<std::uintptr_t>(weakly_aligned) % sizeof(complex_type) != 0,
            "test pointer must satisfy complex ABI alignment and miss VkFFT vector alignment");
        test_utils::check(
            sycl::get_pointer_type(weakly_aligned, queue.get_context())
                != sycl::usm::alloc::unknown,
            "test pointer must belong to the plan's USM context");
        const auto options = test_utils::make_plan_options(
            syclfft::placement::out_of_place,
            syclfft::normalization::none,
            syclfft::provider::vkfft);
        auto fft =
            syclfft::plan_dft_1d<Scalar>(queue, length, syclfft::direction::forward, options);
        test_utils::expect_error(syclfft::error_code::invalid_pointer, [&] {
            fft.execute(weakly_aligned, aligned.get());
        });
        test_utils::expect_error(syclfft::error_code::invalid_pointer, [&] {
            fft.execute(aligned.get(), weakly_aligned);
        });

        const complex_type expected{Scalar{0.5}, Scalar{-0.25}};
        auto identity =
            syclfft::plan_dft_1d<Scalar>(queue, 1, syclfft::direction::forward, options);
        const auto uploaded = queue.memcpy(weakly_aligned, &expected, sizeof(complex_type));
        const auto copied = identity.execute(weakly_aligned, aligned.get(), uploaded);
        complex_type actual{};
        queue.memcpy(&actual, aligned.get(), sizeof(complex_type), copied).wait_and_throw();
        test_utils::check(actual == expected, "VkFFT identity must accept scalar-aligned input");
    }

    void check_automatic_provider(sycl::queue &queue, syclfft::provider expected_provider)
    {
        using value_type = syclfft::complex<float>;
        constexpr std::size_t length = 8;
        const value_type expected{0.5f, -0.25f};
        std::array<value_type, length> source{};
        source[0] = expected;
        usm_buffer<value_type> input(length, queue);
        usm_buffer<value_type> output(length, queue);
        input.write(source);
        auto fft = syclfft::plan_dft_1d<float>(queue, length, syclfft::direction::forward);
        test_utils::check(
            fft.selected_provider() == expected_provider, "wrong automatic provider selection");
        fft.execute(input.get(), output.get()).wait_and_throw();
        for (const auto &value : output.read()) {
            test_utils::check(
                sycl::ext::cplx::abs(value - expected) <= 1e-6f,
                "automatic provider produced an incorrect spectrum");
        }
    }

    void check_complex_abi()
    {
        static_assert(sizeof(syclfft::complex<float>) == 2 * sizeof(float));
        static_assert(alignof(syclfft::complex<float>) == alignof(float));
        static_assert(std::is_trivially_copyable_v<syclfft::complex<float>>);

        struct native_complex_float
        {
            float real;
            float imag;
        };

        static_assert(sizeof(native_complex_float) == sizeof(syclfft::complex<float>));
        static_assert(alignof(native_complex_float) == alignof(syclfft::complex<float>));
        const std::complex<float> host_value{2.5f, -1.25f};
        const syclfft::complex<float> sycl_value{host_value};
        const std::complex<float> converted = sycl_value;
        test_utils::check(converted == host_value, "SyclCPLX std::complex conversion failed");
        const auto native_value = portability_bit_cast<native_complex_float>(sycl_value);
        test_utils::check(
            native_value.real == host_value.real() && native_value.imag == host_value.imag(),
            "SyclCPLX vendor complex layout mismatch");
    }

    enum class test_mode
    {
        normal,
        portable_unavailable,
        require_cufft,
        require_vkfft,
        require_vkfft_only,
        vkfft_unavailable,
        vkfft_plan_fallback
    };

    test_mode parse_mode(int argc, char **argv)
    {
        if (argc == 1) {
            return test_mode::normal;
        }
        test_utils::check(argc == 2, "unknown test argument");
        const std::pair<std::string_view, test_mode> modes[]{
            {"--expect-portable-unavailable", test_mode::portable_unavailable},
            {"--require-cufft",               test_mode::require_cufft       },
            {"--require-vkfft",               test_mode::require_vkfft       },
            {"--require-vkfft-only",          test_mode::require_vkfft_only  },
            {"--expect-vkfft-unavailable",    test_mode::vkfft_unavailable   },
            {"--expect-vkfft-plan-fallback",  test_mode::vkfft_plan_fallback },
        };
        for (const auto &[name, mode] : modes) {
            if (name == argv[1]) {
                return mode;
            }
        }
        throw std::runtime_error("unknown test argument");
    }

    const syclfft::provider_status &provider_status(
        const std::vector<syclfft::provider_status> &providers, syclfft::provider id)
    {
        const auto status = std::find_if(
            providers.begin(), providers.end(), [id](const auto &item) { return item.id == id; });
        test_utils::check(status != providers.end(), "provider status is missing");
        return *status;
    }

    void expect_provider_plan_error(
        sycl::queue &queue, syclfft::provider provider, syclfft::error_code error)
    {
        const auto options = test_utils::make_plan_options(
            syclfft::placement::out_of_place, syclfft::normalization::none, provider);
        test_utils::expect_error(error, [&] {
            (void)syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward, options);
        });
    }

    void check_portable_unavailable(sycl::queue &queue, const syclfft::provider_status &portable)
    {
        test_utils::check(
            portable.built && !portable.available && !portable.reason.empty(),
            "expected a built but unavailable portable provider with a diagnostic");
        expect_provider_plan_error(
            queue, syclfft::provider::portable_sycl, syclfft::error_code::provider_unavailable);
        std::cout << "Portable provider correctly unavailable on "
                  << queue.get_device().get_info<sycl::info::device::name>() << ": "
                  << portable.reason << '\n';
    }

    bool run_cuda_provider_probe(
        sycl::queue &queue, test_mode mode, const syclfft::provider_status &cufft,
        const syclfft::provider_status &vkfft)
    {
        const bool planning_fallback = mode == test_mode::vkfft_plan_fallback;
        const bool missing_helper = mode == test_mode::vkfft_unavailable;
        const bool only_vkfft = mode == test_mode::require_vkfft_only;
        if (mode == test_mode::require_cufft) {
            test_utils::check(cufft.available, "cuFFT provider unavailable: " + cufft.reason);
        }
        if (mode == test_mode::require_vkfft || only_vkfft || planning_fallback) {
            test_utils::check(vkfft.available, "VkFFT provider unavailable: " + vkfft.reason);
        }
        if (only_vkfft || planning_fallback || missing_helper) {
            test_utils::check(!cufft.available, "test requires cuFFT to be unavailable");
        }
        if (missing_helper) {
            test_utils::check(
                vkfft.built && !vkfft.available
                    && vkfft.reason.find("compiler helper") != std::string::npos,
                "expected VkFFT discovery to diagnose a missing compiler helper");
        }
        if (vkfft.available) {
            for (int i = 0; i < 3; ++i) {
                const auto queried = syclfft::query_providers(queue);
                test_utils::check(
                    provider_status(queried, syclfft::provider::vkfft).available,
                    "VkFFT unavailable after repeated query");
            }
        }
        if (!planning_fallback && !missing_helper) {
            return false;
        }
        expect_provider_plan_error(
            queue,
            syclfft::provider::vkfft,
            planning_fallback ? syclfft::error_code::planning_failed
                              : syclfft::error_code::provider_unavailable);
        check_automatic_provider(queue, syclfft::provider::portable_sycl);
        std::cout << "VkFFT " << (planning_fallback ? "planning" : "discovery")
                  << " fallback passed\n";
        return true;
    }

    constexpr std::array normalizations{
        syclfft::normalization::none,
        syclfft::normalization::forward,
        syclfft::normalization::backward,
        syclfft::normalization::orthogonal};

    void run_portable_tests(sycl::queue &queue)
    {
        run_prime_length_tone(queue, syclfft::provider::portable_sycl);
        run_batched_tones<float>(queue, 1024, syclfft::provider::portable_sycl);
        run_batched_tones<float>(queue, 2048, syclfft::provider::portable_sycl);
        run_batched_tones<float>(queue, 4096, syclfft::provider::portable_sycl);
        run_batched_tones<float>(queue, 4093, syclfft::provider::portable_sycl);
        for (const auto normalization : normalizations) {
            run_1d<float>(
                queue,
                8,
                syclfft::direction::forward,
                normalization,
                syclfft::placement::out_of_place,
                syclfft::provider::portable_sycl);
        }
        run_1d<float>(
            queue,
            7,
            syclfft::direction::backward,
            syclfft::normalization::none,
            syclfft::placement::in_place);
        run_1d<float>(
            queue,
            6,
            syclfft::direction::forward,
            syclfft::normalization::none,
            syclfft::placement::out_of_place);
        run_1d<float>(
            queue,
            15,
            syclfft::direction::backward,
            syclfft::normalization::none,
            syclfft::placement::out_of_place);
        run_1d<float>(
            queue,
            35,
            syclfft::direction::forward,
            syclfft::normalization::none,
            syclfft::placement::out_of_place);
        run_1d<float>(
            queue,
            11,
            syclfft::direction::backward,
            syclfft::normalization::none,
            syclfft::placement::in_place);
        run_many<float>(queue, {3, 4}, 2);
        run_many<float>(queue, {2, 3, 5}, 1);
        run_many<float>(queue, {11, 3}, 1);
        run_many<float>(queue, {11, 13, 1}, 2);
        if (queue.get_device().has(sycl::aspect::fp64)) {
            run_1d<double>(
                queue,
                11,
                syclfft::direction::forward,
                syclfft::normalization::orthogonal,
                syclfft::placement::out_of_place);
            run_many<double>(queue, {2, 11, 3}, 2);
            run_batched_tones<double>(queue, 1024, syclfft::provider::portable_sycl);
            run_batched_tones<double>(queue, 2048, syclfft::provider::portable_sycl);
            run_batched_tones<double>(queue, 4096, syclfft::provider::portable_sycl);
        } else {
            test_utils::expect_error(syclfft::error_code::unsupported_precision, [&] {
                (void)syclfft::plan_dft_1d<double>(queue, 8, syclfft::direction::forward);
            });
        }
    }

    void run_cufft_tests(sycl::queue &queue)
    {
        constexpr auto provider = syclfft::provider::cufft;
        for (const auto normalization : normalizations) {
            run_1d<float>(
                queue,
                8,
                syclfft::direction::forward,
                normalization,
                syclfft::placement::out_of_place,
                provider);
        }
        run_1d<float>(
            queue,
            11,
            syclfft::direction::backward,
            syclfft::normalization::backward,
            syclfft::placement::in_place,
            provider);
        run_many<float>(queue, {3, 4}, 2, provider);
        run_many<float>(queue, {2, 3, 5}, 2, provider);
        run_prime_length_tone(queue, provider);
        run_batched_tones<float>(queue, 4096, provider);

        if (queue.get_device().has(sycl::aspect::fp64)) {
            run_1d<double>(
                queue,
                11,
                syclfft::direction::backward,
                syclfft::normalization::orthogonal,
                syclfft::placement::out_of_place,
                provider);
            run_many<double>(queue, {2, 11, 3}, 2, provider);
        }
    }

    void run_vkfft_tests(sycl::queue &queue)
    {
        const auto backend = syclfft::provider::vkfft;
        check_vkfft_pointer_alignment<float>(queue);
        run_1d<float>(
            queue,
            1,
            syclfft::direction::forward,
            syclfft::normalization::forward,
            syclfft::placement::out_of_place,
            backend);
        run_1d<float>(
            queue,
            1,
            syclfft::direction::backward,
            syclfft::normalization::none,
            syclfft::placement::in_place,
            backend);
        run_1d<float>(
            queue,
            127,
            syclfft::direction::backward,
            syclfft::normalization::backward,
            syclfft::placement::out_of_place,
            backend);
        run_many<float>(queue, {3, 4}, 2, backend);
        run_many<float>(queue, {3, 1, 4}, 2, backend, syclfft::direction::backward);
        run_batched_tones<float>(queue, 4096, backend);
        run_batched_tones<float>(queue, 4093, backend);
        if (queue.get_device().has(sycl::aspect::fp64)) {
            check_vkfft_pointer_alignment<double>(queue);
            run_batched_tones<double>(queue, 4096, backend);
            run_many<double>(queue, {2, 11, 3}, 2, backend, syclfft::direction::backward);
        }
    }

    void run_fftw_tests(sycl::queue &queue)
    {
        std::vector<std::complex<float>> source(8);
        source[1] = {1.0f, -0.5f};
        const auto expected =
            direct_dft(source, syclfft::direction::forward, syclfft::normalization::none);
        std::vector<syclfft::complex<float>> device_source(8);
        for (std::size_t i = 0; i < 8; ++i) {
            device_source[i] = source[i];
        }
        usm_buffer<syclfft::complex<float>> input(8, queue, true);
        usm_buffer<syclfft::complex<float>> output(8, queue, true);
        input.write(device_source);
        auto fft = syclfft::plan_dft_1d<float>(
            queue,
            8,
            syclfft::direction::forward,
            test_utils::make_plan_options(
                syclfft::placement::out_of_place,
                syclfft::normalization::none,
                syclfft::provider::fftw));
        fft.execute(input.get(), output.get()).wait_and_throw();
        const auto actual = output.read();
        for (std::size_t i = 0; i < 8; ++i) {
            test_utils::check(
                std::abs(static_cast<std::complex<float>>(actual[i]) - expected[i]) < 1e-4f,
                "explicit FFTW SYCL path mismatch");
        }
    }

    void check_async_execution(sycl::queue &queue)
    {
        {
            usm_buffer<syclfft::complex<float>> input(13, queue);
            usm_buffer<syclfft::complex<float>> spectrum(13, queue);
            usm_buffer<syclfft::complex<float>> restored(13, queue);
            std::vector<syclfft::complex<float>> source(13);
            for (std::size_t i = 0; i < 13; ++i) {
                source[i] = {static_cast<float>(i) / 7.0f, static_cast<float>(i % 3) / 5.0f};
            }
            input.write(source);
            auto forward = syclfft::plan_dft_1d<float>(queue, 13, syclfft::direction::forward);
            auto backward = syclfft::plan_dft_1d<float>(
                queue,
                13,
                syclfft::direction::backward,
                test_utils::make_plan_options(
                    syclfft::placement::out_of_place, syclfft::normalization::backward));
            const auto forward_done = forward.execute(input.get(), spectrum.get());
            backward.execute(spectrum.get(), restored.get(), forward_done).wait_and_throw();
            const auto actual = restored.read();
            for (std::size_t i = 0; i < 13; ++i) {
                test_utils::check(
                    std::abs(
                        static_cast<std::complex<float>>(actual[i])
                        - static_cast<std::complex<float>>(source[i]))
                        < 2e-3f,
                    "automatic provider round trip mismatch");
            }
        }

        {
            usm_buffer<syclfft::complex<float>> input_a(8, queue);
            usm_buffer<syclfft::complex<float>> input_b(8, queue);
            usm_buffer<syclfft::complex<float>> output_a(8, queue);
            usm_buffer<syclfft::complex<float>> output_b(8, queue);
            std::vector<syclfft::complex<float>> source_a(8);
            std::vector<syclfft::complex<float>> source_b(8);
            for (std::size_t i = 0; i < 8; ++i) {
                source_a[i] = {i == 0 ? 1.0f : 0.0f, 0.0f};
                source_b[i] = {i == 1 ? 1.0f : 0.0f, 0.0f};
            }
            input_a.write(source_a);
            input_b.write(source_b);
            auto first = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
            auto second = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
            auto first_done = first.execute(input_a.get(), output_a.get());
            auto second_done = second.execute(input_b.get(), output_b.get());
            first_done.wait_and_throw();
            second_done.wait_and_throw();
            const auto actual_a = output_a.read();
            const auto actual_b = output_b.read();
            test_utils::check(
                std::abs(actual_a[0].real() - 1.0f) < 1e-5f
                    && std::abs(actual_b[0].real() - 1.0f) < 1e-5f,
                "concurrent plan execution failed");
        }

        {
            usm_buffer<syclfft::complex<float>> input(8, queue);
            usm_buffer<syclfft::complex<float>> output(8, queue);
            auto *input_pointer = input.get();
            auto first_half = queue.submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> i) {
                    input_pointer[i[0]] = {i[0] == 0 ? 1.0f : 0.0f, 0.0f};
                });
            });
            auto second_half = queue.submit([&](sycl::handler &handler) {
                handler.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> i) {
                    input_pointer[i[0] + 4] = {0.0f, 0.0f};
                });
            });
            const std::array dependencies{first_half, second_half};
            auto fft = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
            fft.execute(input.get(), output.get(), syclfft::span<const sycl::event>{dependencies})
                .wait_and_throw();
            const auto actual = output.read();
            for (std::size_t i = 0; i < 8; ++i) {
                test_utils::check(
                    std::abs(actual[i].real() - 1.0f) < 1e-5f && std::abs(actual[i].imag()) < 1e-5f,
                    "event-span dependency ordering failed");
            }
        }
    }

    void check_argument_errors(sycl::queue &queue, bool cufft_available)
    {
        check_plan_size_overflow(queue);
        check_out_of_place_ranges(queue);

        const sycl::context foreign_context{queue.get_device()};
        if (foreign_context != queue.get_context()) {
            sycl::queue foreign_queue{foreign_context, queue.get_device()};
            usm_buffer<syclfft::complex<float>> foreign_input(8, foreign_queue);
            usm_buffer<syclfft::complex<float>> output(8, queue);
            if (sycl::get_pointer_type(foreign_input.get(), queue.get_context())
                == sycl::usm::alloc::unknown) {
                test_utils::expect_error(syclfft::error_code::invalid_pointer, [&] {
                    auto fft = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
                    fft.execute(foreign_input.get(), output.get());
                });
            }
        }

        if (!cufft_available) {
            expect_provider_plan_error(
                queue, syclfft::provider::cufft, syclfft::error_code::provider_unavailable);
        }
        test_utils::expect_error(syclfft::error_code::invalid_argument, [&] {
            (void)syclfft::plan_dft_1d<float>(queue, 8, static_cast<syclfft::direction>(0));
        });
        test_utils::expect_error(syclfft::error_code::invalid_pointer, [&] {
            auto fft = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
            fft.execute(nullptr, nullptr);
        });
        test_utils::expect_error(syclfft::error_code::invalid_argument, [&] {
            const auto options = test_utils::make_plan_options(syclfft::placement::in_place);
            auto fft = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward, options);
            usm_buffer<syclfft::complex<float>> input(8, queue);
            usm_buffer<syclfft::complex<float>> output(8, queue);
            fft.execute(input.get(), output.get());
        });
        test_utils::expect_error(syclfft::error_code::invalid_state, [&] {
            auto original = syclfft::plan_dft_1d<float>(queue, 4, syclfft::direction::forward);
            auto moved = std::move(original);
            (void)moved;
            // Deliberately verify the documented moved-from plan contract.
            original.execute(nullptr, nullptr); // NOLINT(bugprone-use-after-move)
        });
    }

} // namespace

int main(int argc, char **argv)
try {
    check_complex_abi();
    const auto mode = parse_mode(argc, argv);
    sycl::queue queue;
    const auto providers = syclfft::query_providers(queue);
    const auto &portable = provider_status(providers, syclfft::provider::portable_sycl);
    if (mode == test_mode::portable_unavailable) {
        check_portable_unavailable(queue, portable);
        return 0;
    }

    const auto &cufft = provider_status(providers, syclfft::provider::cufft);
    const auto &vkfft = provider_status(providers, syclfft::provider::vkfft);
    if (run_cuda_provider_probe(queue, mode, cufft, vkfft)) {
        return 0;
    }
    test_utils::check(portable.available, "portable provider unavailable: " + portable.reason);

    run_portable_tests(queue);
    auto expected_provider = syclfft::provider::portable_sycl;
    if (vkfft.available) {
        run_vkfft_tests(queue);
        expected_provider = syclfft::provider::vkfft;
    }
    if (cufft.available) {
        run_cufft_tests(queue);
        expected_provider = syclfft::provider::cufft;
    }
    if (provider_status(providers, syclfft::provider::fftw).available) {
        run_fftw_tests(queue);
    }
    check_automatic_provider(queue, expected_provider);
    check_async_execution(queue);
    check_argument_errors(queue, cufft.available);

    std::cout << "SYCL tests passed on " << queue.get_device().get_info<sycl::info::device::name>()
              << '\n';
    return 0;
} catch (const std::exception &ex) {
    std::cerr << "SYCL tests failed: " << ex.what() << '\n';
    return 1;
}
