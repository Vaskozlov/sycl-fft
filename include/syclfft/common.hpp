#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#    if defined(SYCLFFT_BUILDING_CORE)
#        define SYCLFFT_CORE_EXPORT __declspec(dllexport)
#    else
#        define SYCLFFT_CORE_EXPORT __declspec(dllimport)
#    endif
#    if defined(SYCLFFT_BUILDING_HOST)
#        define SYCLFFT_HOST_EXPORT __declspec(dllexport)
#    else
#        define SYCLFFT_HOST_EXPORT __declspec(dllimport)
#    endif
#    if defined(SYCLFFT_BUILDING_SYCL)
#        define SYCLFFT_SYCL_EXPORT __declspec(dllexport)
#    else
#        define SYCLFFT_SYCL_EXPORT __declspec(dllimport)
#    endif
#else
#    define SYCLFFT_CORE_EXPORT __attribute__((visibility("default")))
#    define SYCLFFT_HOST_EXPORT __attribute__((visibility("default")))
#    define SYCLFFT_SYCL_EXPORT __attribute__((visibility("default")))
#endif

namespace syclfft
{

    enum class direction
    {
        forward = -1,
        backward = 1
    };
    enum class placement
    {
        in_place,
        out_of_place
    };
    enum class normalization
    {
        none,
        forward,
        backward,
        orthogonal
    };
    enum class planning_mode
    {
        estimate,
        measure
    };
    enum class provider
    {
        automatic,
        portable_sycl,
        fftw,
        cufft,
        rocfft,
        onemkl
    };

    struct plan_options
    {
        syclfft::placement placement = syclfft::placement::out_of_place;
        syclfft::normalization normalization = syclfft::normalization::none;
        planning_mode planning = planning_mode::estimate;
        provider preferred_provider = provider::automatic;
    };

    struct provider_status
    {
        provider id{};
        std::string name;
        bool built{};
        bool available{};
        std::string reason;
    };

    enum class error_code
    {
        invalid_argument,
        invalid_state,
        invalid_pointer,
        unsupported_precision,
        provider_unavailable,
        planning_failed,
        execution_failed,
        plugin_error,
    };

    class SYCLFFT_CORE_EXPORT exception : public std::runtime_error
    {
    public:
        exception(error_code code, const std::string &message)
          : std::runtime_error(message)
          , code_(code)
        {}

        [[nodiscard]] error_code code() const noexcept
        {
            return code_;
        }

    private:
        error_code code_;
    };

    namespace detail
    {
        SYCLFFT_CORE_EXPORT std::size_t
            checked_element_count(const std::vector<std::size_t> &lengths, std::size_t batch_count);
        SYCLFFT_CORE_EXPORT double
            normalization_scale(normalization mode, direction dir, std::size_t transform_size);
        SYCLFFT_CORE_EXPORT const char *provider_name(provider value) noexcept;
    } // namespace detail

} // namespace syclfft
