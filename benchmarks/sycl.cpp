#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <complex>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <syclfft/detail/cuda_queue_utils.hpp>
#include <syclfft/syclfft.hpp>
#include <type_traits>
#include <vector>

namespace
{
    using clock = std::chrono::steady_clock;

    double elapsed_us(clock::time_point start)
    {
        return std::chrono::duration<double, std::micro>(clock::now() - start).count();
    }

    std::size_t parse_positive(std::string_view value)
    {
        std::size_t result{};
        const auto end = value.data() + value.size();
        const auto parsed = std::from_chars(value.data(), end, result);
        if (parsed.ec != std::errc{} || parsed.ptr != end || result == 0) {
            throw std::invalid_argument("Expected a positive integer: " + std::string(value));
        }
        return result;
    }

    std::vector<std::size_t> parse_shape(std::string_view shape)
    {
        std::vector<std::size_t> lengths;
        for (;;) {
            const auto separator = shape.find('x');
            lengths.push_back(parse_positive(shape.substr(0, separator)));
            if (separator == std::string_view::npos) {
                return lengths;
            }
            shape.remove_prefix(separator + 1);
        }
    }

    sycl::queue make_queue(std::string_view kind)
    {
        if (kind == "cpu") {
            return sycl::queue{sycl::cpu_selector_v};
        }
        if (kind == "cuda") {
            for (const auto &device : sycl::device::get_devices(sycl::info::device_type::gpu)) {
                sycl::queue queue{device};
                if (syclfft::detail::is_cuda_queue(queue)) {
                    return queue;
                }
            }
            throw std::runtime_error("No CUDA device was found");
        }
        throw std::invalid_argument("Device must be cpu or cuda");
    }

    syclfft::provider parse_provider(std::string_view name, const sycl::queue &queue)
    {
        if (name == "automatic") {
            return syclfft::provider::automatic;
        }
        for (const auto &status : syclfft::query_providers(queue)) {
            if (status.name == name) {
                return status.id;
            }
        }
        throw std::invalid_argument("Unknown provider: " + std::string(name));
    }

    template <class T>
    auto allocate(sycl::queue &queue, std::size_t count, bool shared)
    {
        auto free = [&queue](T *pointer) { sycl::free(pointer, queue); };
        std::unique_ptr<T[], decltype(free)> buffer{
            shared ? sycl::malloc_shared<T>(count, queue) : sycl::malloc_device<T>(count, queue),
            free};
        if (!buffer) {
            throw std::bad_alloc{};
        }
        return buffer;
    }

    template <class Scalar>
    double check_samples(
        sycl::queue &queue, const std::vector<std::size_t> &lengths,
        const std::vector<syclfft::complex<Scalar>> &input, const syclfft::complex<Scalar> *output,
        std::size_t batch)
    {
        using reference_complex = std::complex<double>;
        const auto points = input.size() / batch;
        std::vector<syclfft::complex<Scalar>> result(points);
        double max_error = 0;
        for (const auto row : {std::size_t{0}, batch - 1}) {
            const auto offset = row * points;
            queue.memcpy(result.data(), output + offset, points * sizeof(result.front()))
                .wait_and_throw();
            for (const auto bin : {std::size_t{0}, points / 3, points - 1}) {
                std::vector<std::vector<reference_complex>> phases;
                auto frequency = bin;
                for (auto axis = lengths.rbegin(); axis != lengths.rend(); ++axis) {
                    auto &table = phases.emplace_back();
                    const double step =
                        -2.0 * std::acos(-1.0) * (static_cast<double>(frequency % *axis) / *axis);
                    frequency /= *axis;
                    for (std::size_t coordinate = 0; coordinate < *axis; ++coordinate) {
                        table.push_back(std::polar(1.0, step * coordinate));
                    }
                }
                reference_complex expected{};
                for (std::size_t point = 0; point < points; ++point) {
                    const auto value = input[offset + point];
                    reference_complex term{value.real(), value.imag()};
                    auto index = point;
                    for (const auto &table : phases) {
                        term *= table[index % table.size()];
                        index /= table.size();
                    }
                    expected += term;
                }
                const reference_complex actual{result[bin].real(), result[bin].imag()};
                const double error = std::abs(actual - expected) / std::sqrt(points);
                if (!std::isfinite(error)) {
                    return std::numeric_limits<double>::infinity();
                }
                max_error = std::max(max_error, error);
            }
            if (batch == 1) {
                break;
            }
        }
        return max_error;
    }

    template <class Submit>
    void measure(std::size_t samples, std::size_t burst, Submit submit)
    {
        std::vector<double> timings;
        timings.reserve(samples);
        for (std::size_t sample = 0; sample < samples; ++sample) {
            const auto start = clock::now();
            auto done = submit();
            for (std::size_t index = 1; index < burst; ++index) {
                done = submit();
            }
            done.wait_and_throw();
            timings.push_back(elapsed_us(start) / burst);
        }
        std::sort(timings.begin(), timings.end());
        for (const auto percentile : {10, 50, 90}) {
            std::cout << ',' << timings[(samples - 1) * percentile / 100];
        }
    }

    template <class Scalar>
    bool
        run(sycl::queue &queue, syclfft::provider provider, std::string_view device,
            std::string_view precision, std::string_view shape, std::size_t batch,
            std::size_t samples, std::size_t burst)
    {
        using complex = syclfft::complex<Scalar>;
        const auto lengths = parse_shape(shape);
        const auto count = syclfft::detail::checked_element_count(lengths, batch);
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(complex)) {
            throw std::invalid_argument("FFT allocation size overflows size_t");
        }
        const bool shared = queue.get_device().is_cpu();
        auto input = allocate<complex>(queue, count, shared);
        auto output = allocate<complex>(queue, count, shared);
        std::vector<complex> host_input(count);
        std::mt19937 rng{20260622};
        const auto component = [&rng] {
            return static_cast<Scalar>(rng() & 0xffffu) / Scalar{32768} - Scalar{1};
        };
        for (auto &value : host_input) {
            value = {component(), component()};
        }
        queue.memcpy(input.get(), host_input.data(), count * sizeof(complex)).wait_and_throw();

        syclfft::plan_options options;
        options.preferred_provider = provider;
        const auto start_plan = clock::now();
        auto plan = syclfft::plan_many_dft<Scalar>(
            queue, lengths, batch, syclfft::direction::forward, options);
        const auto plan_us = elapsed_us(start_plan);
        const auto submit = [&] { return plan.execute(input.get(), output.get()); };
        const auto start_first = clock::now();
        submit().wait_and_throw();
        const auto first_us = elapsed_us(start_first);
        for (int index = 0; index < 5; ++index) {
            submit().wait_and_throw();
        }

        std::cout << std::setprecision(8) << device << ','
                  << syclfft::detail::provider_name(plan.selected_provider()) << ',' << precision
                  << ',' << shape << ',' << batch << ',' << samples << ',' << burst << ','
                  << plan.scratch_size_bytes() << ',' << plan_us << ',' << first_us;
        measure(samples, 1, submit);
        measure(samples, burst, submit);
        const auto error = check_samples(queue, lengths, host_input, output.get(), batch);
        std::cout << ',' << error << '\n';
        const auto limit = std::is_same_v<Scalar, float> ? 5e-4 : 1e-10;
        return error <= limit;
    }
} // namespace

int main(int argc, char **argv)
try {
    if (argc < 5 || argc > 8) {
        std::cerr << "Usage: syclfft_benchmark cpu|cuda PROVIDER float|double Nx... "
                     "[batch=1] [samples=15] [burst=8]\n";
        return 1;
    }
    const std::string_view device = argv[1];
    const std::string_view precision = argv[3];
    if (precision != "float" && precision != "double") {
        throw std::invalid_argument("Precision must be float or double");
    }
    const auto batch = argc > 5 ? parse_positive(argv[5]) : 1;
    const auto samples = argc > 6 ? parse_positive(argv[6]) : 15;
    const auto burst = argc > 7 ? parse_positive(argv[7]) : 8;
    auto queue = make_queue(device);
    const auto provider = parse_provider(argv[2], queue);
    std::cerr << queue.get_device().get_info<sycl::info::device::name>() << " ("
              << queue.get_device().get_info<sycl::info::device::driver_version>() << ")\n";
    std::cout << "device,provider,precision,shape,batch,samples,burst,scratch_bytes,plan_us,"
                 "first_us,latency_p10_us,latency_p50_us,latency_p90_us,burst_p10_us,burst_p50_us,"
                 "burst_p90_us,max_scaled_dft_error\n";
    const auto passed =
        precision == "float"
            ? run<float>(queue, provider, device, precision, argv[4], batch, samples, burst)
            : run<double>(queue, provider, device, precision, argv[4], batch, samples, burst);
    if (!passed) {
        std::cerr << "FFT output exceeded the sampled DFT error limit\n";
    }
    return passed ? 0 : 2;
} catch (const std::exception &error) {
    std::cerr << "FFT benchmark failed: " << error.what() << '\n';
    return 1;
}
