#pragma once

#include <cstddef>
#include <syclfft/syclfft.hpp>
#include <vector>

namespace syclfft::detail
{
    template <class Scalar>
    void fft_radix2_line(
        const complex<Scalar> *input, complex<Scalar> *output, std::size_t length,
        const complex<Scalar> *twiddles, Scalar scale)
    {
        const bool in_place = input == output;
        std::size_t reversed = 0;
        for (std::size_t coordinate = 0; coordinate < length; ++coordinate) {
            if (in_place) {
                if (coordinate < reversed) {
                    const auto value = output[coordinate];
                    output[coordinate] = output[reversed];
                    output[reversed] = value;
                }
            } else {
                output[coordinate] = input[reversed];
            }
            std::size_t bit = length >> 1;
            while (reversed & bit) {
                reversed ^= bit;
                bit >>= 1;
            }
            reversed ^= bit;
        }

        for (std::size_t stage = 2; stage <= length; stage <<= 1) {
            const auto half = stage / 2;
            const auto twiddle_stride = length / stage;
            for (std::size_t base = 0; base < length; base += stage) {
                for (std::size_t j = 0; j < half; ++j) {
                    const auto first = base + j;
                    const auto second = first + half;
                    const auto lhs = output[first];
                    const auto rhs = output[second] * twiddles[j * twiddle_stride];
                    output[first] = lhs + rhs;
                    output[second] = lhs - rhs;
                }
            }
        }
        if (scale != Scalar{1}) {
            for (std::size_t coordinate = 0; coordinate < length; ++coordinate) {
                output[coordinate] *= scale;
            }
        }
    }

    template <class Scalar>
    sycl::event enqueue_cpu_radix2(
        sycl::queue &queue, const complex<Scalar> *input, complex<Scalar> *output,
        std::size_t length, std::size_t batch_count, const complex<Scalar> *twiddles, Scalar scale,
        const std::vector<sycl::event> &dependencies)
    {
        const auto command_group = [&](sycl::handler &handler) {
            handler.depends_on(dependencies);
            handler.parallel_for(sycl::range<1>{batch_count}, [=](sycl::id<1> item) {
                const auto offset = item[0] * length;
                fft_radix2_line(input + offset, output + offset, length, twiddles, scale);
            });
        };
#if defined(SYCLFFT_USE_ADAPTIVECPP)
        // Smaller batches may otherwise have too few groups to fill the CPU threads.
        if (batch_count >= 8 && batch_count < 128) {
            using preferred_group_size =
                sycl::property::command_group::AdaptiveCpp_prefer_group_size<1>;
            return queue.submit(
                sycl::property_list{preferred_group_size{sycl::range<1>{1}}}, command_group);
        }
#endif
        return queue.submit(command_group);
    }
} // namespace syclfft::detail
