#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <syclfft/syclfft.hpp>
#include <type_traits>
#include <vector>

namespace
{

    void check(bool condition, const std::string &message)
    {
        if (!condition) {
            throw std::runtime_error(message);
        }
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

        void write(std::span<const T> values)
        {
            check(values.size() == count_, "USM upload size mismatch");
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
        const auto scale = static_cast<Scalar>(
            syclfft::detail::normalization_scale(normalization, direction, count));
        for (auto &value : output) {
            value *= scale;
        }
        return output;
    }

    template <class Scalar>
    std::vector<std::complex<Scalar>> reference_many(
        std::vector<std::complex<Scalar>> values, std::span<const std::size_t> shape,
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
    void run_many(sycl::queue &queue, std::vector<std::size_t> shape, std::size_t batch_count)
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
        const auto expected =
            reference_many(source, shape, batch_count, syclfft::direction::forward);
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
            syclfft::direction::forward,
            {.preferred_provider = syclfft::provider::portable_sycl});
        fft.execute(input.get(), output.get()).wait_and_throw();
        const auto actual_values = output.read();
        const auto tolerance = std::is_same_v<Scalar, float> ? Scalar{8e-4} : Scalar{5e-11};
        for (std::size_t i = 0; i < count; ++i) {
            const std::complex<Scalar> actual{actual_values[i].real(), actual_values[i].imag()};
            check(
                std::abs(actual - expected[i]) <= tolerance * (Scalar{1} + std::abs(expected[i])),
                "multidimensional or batch mismatch at element " + std::to_string(i));
        }
    }

    template <class Function>
    void expect_error(syclfft::error_code code, Function &&function)
    {
        try {
            function();
        } catch (const syclfft::exception &ex) {
            check(ex.code() == code, "wrong exception code");
            return;
        }
        throw std::runtime_error("expected syclfft::exception");
    }

    template <class Scalar>
    void run_1d(
        sycl::queue &queue, std::size_t count, syclfft::direction direction,
        syclfft::normalization normalization, syclfft::placement placement)
    {
        std::vector<std::complex<Scalar>> source(count);
        for (std::size_t i = 0; i < count; ++i) {
            source[i] = {
                static_cast<Scalar>((i * 5 + 1) % 9) / Scalar{4},
                static_cast<Scalar>((i * 3 + 2) % 7) / Scalar{5}};
        }
        const auto expected = direct_dft(source, direction, normalization);
        std::vector<syclfft::complex<Scalar>> device_source(count);
        for (std::size_t i = 0; i < count; ++i) {
            device_source[i] = {source[i].real(), source[i].imag()};
        }
        usm_buffer<syclfft::complex<Scalar>> input(count, queue);
        std::unique_ptr<usm_buffer<syclfft::complex<Scalar>>> output_storage;
        std::unique_ptr<usm_buffer<syclfft::complex<Scalar>>> repeated_storage;
        if (placement == syclfft::placement::out_of_place) {
            output_storage = std::make_unique<usm_buffer<syclfft::complex<Scalar>>>(count, queue);
            repeated_storage = std::make_unique<usm_buffer<syclfft::complex<Scalar>>>(count, queue);
        }
        auto *output = output_storage ? output_storage->get() : input.get();
        auto *repeated_output = repeated_storage ? repeated_storage->get() : nullptr;
        input.write(device_source);
        auto fft = syclfft::plan_dft_1d<Scalar>(
            queue,
            count,
            direction,
            {.placement = placement,
             .normalization = normalization,
             .preferred_provider = syclfft::provider::portable_sycl});
        auto dependency = queue.submit([&](sycl::handler &handler) { handler.single_task([] {}); });
        auto event = placement == syclfft::placement::in_place
                         ? fft.execute(input.get(), dependency)
                         : fft.execute(input.get(), output, dependency);
        sycl::event repeated;
        if (repeated_output) {
            repeated = fft.execute(input.get(), repeated_output);
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
            if (std::abs(actual - expected[i]) > tolerance * (Scalar{1} + std::abs(expected[i]))) {
                throw std::runtime_error("portable mismatch at element " + std::to_string(i));
            }
            if (repeated_output) {
                const std::complex<Scalar> repeated_actual{
                    repeated_values[i].real(), repeated_values[i].imag()};
                check(
                    std::abs(repeated_actual - expected[i])
                        <= tolerance * (Scalar{1} + std::abs(expected[i])),
                    "serialized repeated execution mismatch");
            }
        }
        check(
            fft.selected_provider() == syclfft::provider::portable_sycl, "wrong portable provider");
        check(
            fft.scratch_size_bytes() >= 2 * count * sizeof(syclfft::complex<Scalar>),
            "portable scratch size is too small");
    }

} // namespace

int main(int argc, char **argv)
try {
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
    check(converted == host_value, "SyclCPLX std::complex conversion failed");
    const auto native_value = std::bit_cast<native_complex_float>(sycl_value);
    check(
        native_value.real == host_value.real() && native_value.imag == host_value.imag(),
        "SyclCPLX vendor complex layout mismatch");

    sycl::queue queue;
    if (argc == 2 && std::string_view{argv[1]} == "--expect-portable-unavailable") {
        const auto providers = syclfft::query_providers(queue);
        const auto portable = std::find_if(
            providers.begin(), providers.end(), [](const syclfft::provider_status &status) {
            return status.id == syclfft::provider::portable_sycl;
        });
        check(portable != providers.end(), "portable provider status is missing");
        check(portable->built, "portable provider was not built");
        check(
            !portable->available, "portable provider unexpectedly available on a non-USM runtime");
        check(!portable->reason.empty(), "unavailable portable provider has no diagnostic reason");
        expect_error(syclfft::error_code::provider_unavailable, [&] {
            (void)syclfft::plan_dft_1d<float>(
                queue,
                8,
                syclfft::direction::forward,
                {.preferred_provider = syclfft::provider::portable_sycl});
        });
        std::cout << "Portable provider correctly unavailable on "
                  << queue.get_device().get_info<sycl::info::device::name>() << ": "
                  << portable->reason << '\n';
        return 0;
    }
    check(argc == 1, "unknown test argument");
    for (auto normalization :
         {syclfft::normalization::none,
          syclfft::normalization::forward,
          syclfft::normalization::backward,
          syclfft::normalization::orthogonal}) {
        run_1d<float>(
            queue, 8, syclfft::direction::forward, normalization, syclfft::placement::out_of_place);
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
    if (queue.get_device().has(sycl::aspect::fp64)) {
        run_1d<double>(
            queue,
            11,
            syclfft::direction::forward,
            syclfft::normalization::orthogonal,
            syclfft::placement::out_of_place);
    } else {
        expect_error(syclfft::error_code::unsupported_precision, [&] {
            (void)syclfft::plan_dft_1d<double>(queue, 8, syclfft::direction::forward);
        });
    }

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
            {.normalization = syclfft::normalization::backward});
        const auto forward_done = forward.execute(input.get(), spectrum.get());
        backward.execute(spectrum.get(), restored.get(), forward_done).wait_and_throw();
        const auto actual = restored.read();
        for (std::size_t i = 0; i < 13; ++i) {
            check(
                std::abs(
                    static_cast<std::complex<float>>(actual[i])
                    - static_cast<std::complex<float>>(source[i]))
                    < 2e-3f,
                "portable round trip mismatch");
        }
    }

    const auto providers = syclfft::query_providers(queue);
    check(!providers.empty() && providers.front().available, "portable provider unavailable");
    {
        auto automatic = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
        check(
            automatic.selected_provider() == syclfft::provider::portable_sycl,
            "automatic CPU selection did not safely fall back to portable SYCL");
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
        check(
            std::abs(actual_a[0].real() - 1.0f) < 1e-5f
                && std::abs(actual_b[0].real() - 1.0f) < 1e-5f,
            "concurrent plan execution failed");
    }

    const sycl::context foreign_context{queue.get_device()};
    if (foreign_context != queue.get_context()) {
        sycl::queue foreign_queue{foreign_context, queue.get_device()};
        usm_buffer<syclfft::complex<float>> foreign_input(8, foreign_queue);
        usm_buffer<syclfft::complex<float>> output(8, queue);
        if (sycl::get_pointer_type(foreign_input.get(), queue.get_context())
            == sycl::usm::alloc::unknown) {
            expect_error(syclfft::error_code::invalid_pointer, [&] {
                auto fft = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
                fft.execute(foreign_input.get(), output.get());
            });
        }
    }
    const auto fftw_status = std::find_if(
        providers.begin(), providers.end(), [](const syclfft::provider_status &status) {
        return status.id == syclfft::provider::fftw;
    });
    if (fftw_status != providers.end() && fftw_status->available) {
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
            queue, 8, syclfft::direction::forward, {.preferred_provider = syclfft::provider::fftw});
        fft.execute(input.get(), output.get()).wait_and_throw();
        const auto actual = output.read();
        for (std::size_t i = 0; i < 8; ++i) {
            check(
                std::abs(static_cast<std::complex<float>>(actual[i]) - expected[i]) < 1e-4f,
                "explicit FFTW SYCL path mismatch");
        }
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
            handler.parallel_for(
                sycl::range<1>{4}, [=](sycl::id<1> i) { input_pointer[i[0] + 4] = {0.0f, 0.0f}; });
        });
        const std::array dependencies{first_half, second_half};
        auto fft = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
        fft.execute(input.get(), output.get(), std::span<const sycl::event>{dependencies})
            .wait_and_throw();
        const auto actual = output.read();
        for (std::size_t i = 0; i < 8; ++i) {
            check(
                std::abs(actual[i].real() - 1.0f) < 1e-5f && std::abs(actual[i].imag()) < 1e-5f,
                "event-span dependency ordering failed");
        }
    }

    expect_error(syclfft::error_code::provider_unavailable, [&] {
        (void)syclfft::plan_dft_1d<float>(
            queue,
            8,
            syclfft::direction::forward,
            {.preferred_provider = syclfft::provider::cufft});
    });
    expect_error(syclfft::error_code::invalid_argument, [&] {
        (void)syclfft::plan_dft_1d<float>(queue, 8, static_cast<syclfft::direction>(0));
    });
    expect_error(syclfft::error_code::invalid_pointer, [&] {
        auto fft = syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
        fft.execute(nullptr, nullptr);
    });
    expect_error(syclfft::error_code::invalid_argument, [&] {
        auto fft = syclfft::plan_dft_1d<float>(
            queue, 8, syclfft::direction::forward, {.placement = syclfft::placement::in_place});
        usm_buffer<syclfft::complex<float>> input(8, queue);
        usm_buffer<syclfft::complex<float>> output(8, queue);
        fft.execute(input.get(), output.get());
    });
    expect_error(syclfft::error_code::invalid_state, [&] {
        auto original = syclfft::plan_dft_1d<float>(queue, 4, syclfft::direction::forward);
        auto moved = std::move(original);
        (void)moved;
        // Deliberately verify the documented moved-from plan contract.
        original.execute(nullptr, nullptr); // NOLINT(bugprone-use-after-move)
    });

    std::cout << "SYCL tests passed on " << queue.get_device().get_info<sycl::info::device::name>()
              << '\n';
    return 0;
} catch (const std::exception &ex) {
    std::cerr << "SYCL tests failed: " << ex.what() << '\n';
    return 1;
}
