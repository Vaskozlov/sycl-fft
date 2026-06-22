#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <syclfft/common.hpp>

#define SYCLFFT_CUFFT_PROVIDER_ABI_VERSION 2u

namespace syclfft::detail
{

    struct cufft_plan_config_v2
    {
        std::uint32_t abi_version{SYCLFFT_CUFFT_PROVIDER_ABI_VERSION};
        bool double_precision{};
        direction transform_direction{direction::forward};
        placement transform_placement{};
        std::uint32_t rank{};
        std::array<std::size_t, 3> lengths{};
        std::size_t batch_count{};
    };

    struct cufft_provider_v2
    {
        std::uint32_t abi_version;
        const char *name;
        bool (*supports)(const sycl::queue &, char *, std::size_t);
        void *(*create)(sycl::queue &, const cufft_plan_config_v2 &, char *, std::size_t);
        void (*destroy)(void *);
        sycl::event (*execute)(
            void *, const void *, void *, syclfft::span<const sycl::event>, char *, std::size_t);
    };

    using get_cufft_provider_v2_fn = const cufft_provider_v2 *(*)();

} // namespace syclfft::detail

#if defined(_WIN32)
#    define SYCLFFT_CUFFT_PROVIDER_EXPORT extern "C" __declspec(dllexport)
#else
#    define SYCLFFT_CUFFT_PROVIDER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

SYCLFFT_CUFFT_PROVIDER_EXPORT const syclfft::detail::cufft_provider_v2 *
    syclfft_get_cufft_provider_v2();
