#include <exception>
#include <iostream>
#include <syclfft/detail/provider_loader.hpp>

int main()
try {
    try {
        (void)syclfft::detail::load_host_provider("syclfft_provider_fftw");
    } catch (const syclfft::exception &ex) {
        if (ex.code() == syclfft::error_code::plugin_error) {
            return 0;
        }
        std::cerr << "unexpected error: " << ex.what() << '\n';
        return 1;
    }
    std::cerr << "provider with incompatible ABI was accepted\n";
    return 1;
} catch (const std::exception &ex) {
    std::cerr << "plugin ABI test failed: " << ex.what() << '\n';
    return 1;
}
