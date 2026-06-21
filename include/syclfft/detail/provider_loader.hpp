#pragma once

#include <memory>
#include <string>
#include <vector>

#include <syclfft/common.hpp>
#include <syclfft/detail/provider_abi.h>

namespace syclfft::detail {

struct loaded_host_provider {
  const syclfft_host_provider_v1 *api{};
  std::shared_ptr<void> module;
};

struct loaded_provider_symbol {
  void *symbol{};
  std::shared_ptr<void> module;
};

SYCLFFT_CORE_EXPORT loaded_host_provider
load_host_provider(const std::string &provider_file);
SYCLFFT_CORE_EXPORT loaded_provider_symbol load_provider_symbol(
    const std::string &provider_file, const std::string &symbol_name);
SYCLFFT_CORE_EXPORT std::vector<std::string> provider_search_paths();

} // namespace syclfft::detail
