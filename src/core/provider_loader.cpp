#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <sstream>
#include <syclfft/detail/provider_loader.hpp>
#include <syclfft/detail/trace.hpp>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace syclfft::detail
{
    namespace
    {

        std::vector<std::string> split_paths(const char *value)
        {
            std::vector<std::string> result;
            if (!value) {
                return result;
            }
#if defined(_WIN32)
            constexpr char separator = ';';
#else
            constexpr char separator = ':';
#endif
            std::stringstream stream(value);
            std::string item;
            while (std::getline(stream, item, separator)) {
                if (!item.empty()) {
                    result.push_back(item);
                }
            }
            return result;
        }

        std::filesystem::path library_directory()
        {
#if defined(_WIN32)
            HMODULE module = nullptr;
            if (!GetModuleHandleExA(
                    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                        | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                    reinterpret_cast<LPCSTR>(&library_directory),
                    &module)) {
                return {};
            }
            std::vector<char> path(32768);
            const auto length =
                GetModuleFileNameA(module, path.data(), static_cast<DWORD>(path.size()));
            if (length == 0) {
                return {};
            }
            return std::filesystem::path(std::string(path.data(), length)).parent_path();
#else
            Dl_info info{};
            if (dladdr(reinterpret_cast<void *>(&library_directory), &info) == 0
                || !info.dli_fname) {
                return {};
            }
            return std::filesystem::path(info.dli_fname).parent_path();
#endif
        }

        std::string module_filename(const std::string &base)
        {
#if defined(_WIN32)
            return base + ".dll";
#elif defined(__APPLE__)
            return base + ".so";
#else
            return base + ".so";
#endif
        }

#if defined(_WIN32)
        std::shared_ptr<void> load_pinned_module(const std::filesystem::path &path)
        {
            // Let Windows own module lifetime. PIN prevents provider teardown
            // before process termination without holding a user-space mutex
            // across LoadLibrary.
            const auto key = path.string();
            trace("core: calling LoadLibrary for provider");
            HMODULE raw = LoadLibraryA(key.c_str());
            trace("core: LoadLibrary returned");
            if (!raw) {
                return {};
            }

            HMODULE pinned = nullptr;
            trace("core: pinning provider module");
            const auto pinned_module = GetModuleHandleExA(
                GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_PIN,
                reinterpret_cast<LPCSTR>(raw),
                &pinned);
            trace("core: provider module pin returned");
            if (!pinned_module) {
                FreeLibrary(raw);
                return {};
            }
            FreeLibrary(raw);
            return std::shared_ptr<void>(pinned, [](void *) noexcept {});
        }
#endif

    } // namespace

    std::size_t
        checked_element_count(const std::vector<std::size_t> &lengths, std::size_t batch_count)
    {
        if (lengths.empty() || lengths.size() > 3) {
            throw exception(error_code::invalid_argument, "FFT rank must be between 1 and 3");
        }
        if (batch_count == 0) {
            throw exception(error_code::invalid_argument, "FFT batch count must be non-zero");
        }
        std::size_t count = batch_count;
        for (const auto length : lengths) {
            if (length == 0) {
                throw exception(error_code::invalid_argument, "FFT dimensions must be non-zero");
            }
            if (count > std::numeric_limits<std::size_t>::max() / length) {
                throw exception(error_code::invalid_argument, "FFT element count overflows size_t");
            }
            count *= length;
        }
        return count;
    }

    double normalization_scale(normalization mode, direction dir, std::size_t transform_size)
    {
        switch (mode) {
        case normalization::none:
            return 1.0;
        case normalization::forward:
            return dir == direction::forward ? 1.0 / static_cast<double>(transform_size) : 1.0;
        case normalization::backward:
            return dir == direction::backward ? 1.0 / static_cast<double>(transform_size) : 1.0;
        case normalization::orthogonal:
            return 1.0 / std::sqrt(static_cast<double>(transform_size));
        }
        return 1.0;
    }

    const char *provider_name(provider value) noexcept
    {
        switch (value) {
        case provider::automatic:
            return "automatic";
        case provider::portable_sycl:
            return "portable_sycl";
        case provider::fftw:
            return "fftw";
        case provider::cufft:
            return "cufft";
        case provider::rocfft:
            return "rocfft";
        case provider::onemkl:
            return "onemkl";
        }
        return "unknown";
    }

    std::vector<std::string> provider_search_paths()
    {
        auto result = split_paths(std::getenv("SYCLFFT_PLUGIN_PATH"));
        const auto directory = library_directory();
        if (!directory.empty()) {
            result.push_back((directory / "syclfft" / "providers").string());
        }
        return result;
    }

    loaded_host_provider load_host_provider(const std::string &provider_file)
    {
        trace("core: resolving host provider symbol");
        auto loaded = load_provider_symbol(provider_file, "syclfft_get_host_provider_v1");
        auto symbol = reinterpret_cast<syclfft_get_host_provider_v1_fn>(loaded.symbol);
        trace("core: calling host provider entry point");
        const auto *api = symbol();
        trace("core: host provider entry point returned");
        if (!api || api->abi_version != SYCLFFT_PROVIDER_ABI_VERSION) {
            throw exception(
                error_code::plugin_error,
                "Provider '" + provider_file + "' has an incompatible ABI");
        }
        return loaded_host_provider{api, std::move(loaded.module)};
    }

    loaded_provider_symbol
        load_provider_symbol(const std::string &provider_file, const std::string &symbol_name)
    {
        std::string errors;
        for (const auto &directory : provider_search_paths()) {
            const auto path = std::filesystem::path(directory) / module_filename(provider_file);
#if defined(_WIN32)
            auto module = load_pinned_module(path);
            if (!module) {
                errors += path.string() + ": load failed; ";
                continue;
            }
            auto raw = static_cast<HMODULE>(module.get());
            void *symbol = reinterpret_cast<void *>(GetProcAddress(raw, symbol_name.c_str()));
#else
            void *raw = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!raw) {
                const char *message = dlerror();
                errors += path.string() + ": " + (message ? message : "load failed") + "; ";
                continue;
            }
            auto module = std::shared_ptr<void>(raw, [](void *handle) {
                if (handle) {
                    dlclose(handle);
                }
            });
            void *symbol = dlsym(raw, symbol_name.c_str());
#endif
            if (!symbol) {
                errors += path.string() + ": entry point '" + symbol_name + "' missing; ";
                continue;
            }
            trace("core: provider symbol resolved");
            return loaded_provider_symbol{symbol, std::move(module)};
        }
        throw exception(
            error_code::provider_unavailable,
            "Unable to load provider '" + provider_file + "': " + errors);
    }

} // namespace syclfft::detail
