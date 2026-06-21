#include <syclfft/syclfft.hpp>

int main() {
  sycl::queue queue;
  auto *data = sycl::malloc_shared<syclfft::complex<float>>(4, queue);
  data[0] = {1.0f, 0.0f};
  for (int i = 1; i < 4; ++i)
    data[i] = {0.0f, 0.0f};
  auto plan =
      syclfft::plan_dft_1d<float>(queue, 4, syclfft::direction::forward,
                                  {.placement = syclfft::placement::in_place});
  plan.execute(data).wait_and_throw();
  const bool correct = data[0].real() == 1.0f && data[0].imag() == 0.0f;
  sycl::free(data, queue);
  return correct ? 0 : 1;
}
