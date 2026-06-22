#include <syclfft/detail/provider_abi.h>

namespace
{
    const syclfft_host_provider_v1 bad_provider{
        .abi_version = SYCLFFT_PROVIDER_ABI_VERSION + 1,
        .name = "bad-abi",
        .create = nullptr,
        .destroy = nullptr,
        .execute = nullptr,
    };
} // namespace

SYCLFFT_PROVIDER_EXPORT const syclfft_host_provider_v1 *syclfft_get_host_provider_v1()
{
    return &bad_provider;
}
