#include <syclfft/syclfft.hpp>

#include <iostream>

int main() {
  sycl::queue queue;
  auto *input = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
  auto *output = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
  for (std::size_t i = 0; i < 8; ++i)
    input[i] = {i == 0 ? 1.0f : 0.0f, 0.0f};

  auto fft = syclfft::plan_dft_1d<float>(
      queue, 8, syclfft::direction::forward,
      {.placement = syclfft::placement::out_of_place});
  fft.execute(input, output).wait_and_throw();
  std::cout << "provider=" << static_cast<int>(fft.selected_provider())
            << ", X[0]=" << output[0].real() << "+" << output[0].imag()
            << "i\n";

  sycl::free(input, queue);
  sycl::free(output, queue);
}
