#include <exception>
#include <iostream>
#include <syclfft/host.hpp>

int main()
try {
    try {
        (void)syclfft::host::plan_dft_1d<float>(4, syclfft::direction::forward);
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
