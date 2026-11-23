#include <algorithm>
#include <iostream>
#include <syclfft/syclfft.hpp>

int main()
{
    using complex = syclfft::complex<float>;
    constexpr std::size_t length = 4;
    sycl::queue queue;
    auto *data = sycl::malloc_shared<complex>(length, queue);
    if (!data) {
        return 1;
    }
    std::fill_n(data, length, complex{});
    data[0] = {1.0f, 0.0f};

    syclfft::plan_options options;
    options.placement = syclfft::placement::in_place;
    auto plan = syclfft::plan_dft_1d<float>(queue, length, syclfft::direction::forward, options);
    plan.execute(data).wait_and_throw();
    std::cout << "Automatic provider: " << syclfft::detail::provider_name(plan.selected_provider())
              << '\n';
    const bool correct = std::all_of(data, data + length, [](const auto &value) {
        return value.real() == 1.0f && value.imag() == 0.0f;
    });
    sycl::free(data, queue);
    queue.wait_and_throw();
    return correct ? 0 : 1;
}
