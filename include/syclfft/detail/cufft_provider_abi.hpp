#pragma once

#include <sycl/sycl.hpp>

#include <cstddef>
#include <cstdint>
#include <span>

#include <syclfft/common.hpp>

#define SYCLFFT_CUFFT_PROVIDER_ABI_VERSION 1u

namespace syclfft::detail {

struct cufft_plan_config_v1 {
  std::uint32_t abi_version{SYCLFFT_CUFFT_PROVIDER_ABI_VERSION};
  bool double_precision{};
  direction transform_direction{};
  placement transform_placement{};
  std::uint32_t rank{};
  std::size_t lengths[3]{};
  std::size_t batch_count{};
};

struct cufft_provider_v1 {
  std::uint32_t abi_version;
  const char *name;
  bool (*supports)(const sycl::queue &, char *, std::size_t);
  void *(*create)(sycl::queue &, const cufft_plan_config_v1 &, char *,
                  std::size_t);
  void (*destroy)(void *);
  sycl::event (*execute)(void *, const void *, void *,
                         std::span<const sycl::event>, char *, std::size_t);
};

using get_cufft_provider_v1_fn = const cufft_provider_v1 *(*)();

} // namespace syclfft::detail

#if defined(_WIN32)
#define SYCLFFT_CUFFT_PROVIDER_EXPORT extern "C" __declspec(dllexport)
#else
#define SYCLFFT_CUFFT_PROVIDER_EXPORT                                          \
  extern "C" __attribute__((visibility("default")))
#endif

SYCLFFT_CUFFT_PROVIDER_EXPORT const syclfft::detail::cufft_provider_v1 *
syclfft_get_cufft_provider_v1();
