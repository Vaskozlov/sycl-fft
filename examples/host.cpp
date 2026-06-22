#include <complex>
#include <exception>
#include <iostream>
#include <syclfft/host.hpp>
#include <vector>

int main()
try {
    std::vector<std::complex<float>> input(8), output(8);
    input[0] = {1.0f, 0.0f};

    syclfft::plan_options options;
    options.placement = syclfft::placement::out_of_place;
    auto fft = syclfft::host::plan_dft_1d<float>(8, syclfft::direction::forward, options);
    fft.execute(input.data(), output.data());

    std::cout << "provider=fftw, X[0]=" << output[0] << '\n';
    return 0;
} catch (const std::exception &ex) {
    std::cerr << "host FFT example failed: " << ex.what() << '\n';
    return 1;
}
