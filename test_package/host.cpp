#include <syclfft/host.hpp>

#include <complex>
#include <vector>

int main() {
  std::vector<std::complex<float>> input(4), output(4);
  input[0] = {1.0f, 0.0f};
  auto plan = syclfft::host::plan_dft_1d<float>(4, syclfft::direction::forward);
  plan.execute(input.data(), output.data());
  return output[0] == std::complex<float>{1.0f, 0.0f} ? 0 : 1;
}
