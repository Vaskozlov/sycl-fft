#include <complex>
#include <iostream>
#include <syclfft/host.hpp>
#include <vector>

int main()
{
    std::vector<std::complex<float>> input(4), output(4);
    input[0] = {1.0f, 0.0f};
    std::cerr << "creating installed host plan\n";
    {
        auto plan = syclfft::host::plan_dft_1d<float>(4, syclfft::direction::forward);
        std::cerr << "executing installed host plan\n";
        plan.execute(input.data(), output.data());
        std::cerr << "destroying installed host plan\n";
    }
    std::cerr << "installed host plan completed\n";
    return output[0] == std::complex<float>{1.0f, 0.0f} ? 0 : 1;
}
