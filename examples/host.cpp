#include <syclfft/host.hpp>

#include <complex>
#include <iostream>
#include <vector>

int main() {
  std::vector<std::complex<float>> input(8), output(8);
  input[0] = {1.0f, 0.0f};

  auto fft = syclfft::host::plan_dft_1d<float>(
      8, syclfft::direction::forward,
      {.placement = syclfft::placement::out_of_place});
  fft.execute(input.data(), output.data());

  std::cout << "provider=fftw, X[0]=" << output[0] << '\n';
}
