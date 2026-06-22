#include <exception>
#include <iostream>
#include <memory>
#include <new>
#include <syclfft/syclfft.hpp>

int main()
try {
    sycl::queue queue;
    const auto deleter = [&queue](syclfft::complex<float> *pointer) {
        if (pointer) {
            sycl::free(pointer, queue);
        }
    };
    std::unique_ptr<syclfft::complex<float>[], decltype(deleter)> input {
        sycl::malloc_shared<syclfft::complex<float>>(8, queue), deleter
    };
    std::unique_ptr<syclfft::complex<float>[], decltype(deleter)> output {
        sycl::malloc_shared<syclfft::complex<float>>(8, queue), deleter
    };
    if (!input || !output) {
        throw std::bad_alloc{};
    }
    for (std::size_t i = 0; i < 8; ++i) {
        input[i] = {i == 0 ? 1.0f : 0.0f, 0.0f};
    }

    auto fft = syclfft::plan_dft_1d<float>(
        queue, 8, syclfft::direction::forward, {.placement = syclfft::placement::out_of_place});
    fft.execute(input.get(), output.get()).wait_and_throw();
    std::cout << "provider=" << static_cast<int>(fft.selected_provider())
              << ", X[0]=" << output[0].real() << "+" << output[0].imag() << "i\n";
    return 0;
} catch (const std::exception &ex) {
    std::cerr << "SYCL FFT example failed: " << ex.what() << '\n';
    return 1;
}
