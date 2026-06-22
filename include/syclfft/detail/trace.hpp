#pragma once

#include <cstdio>
#include <cstdlib>

namespace syclfft::detail
{
    inline void trace(const char *message) noexcept
    {
        const char *enabled = std::getenv("SYCLFFT_TRACE");
        if (!enabled || enabled[0] == '\0' || enabled[0] == '0') {
            return;
        }
        std::fputs("[syclfft] ", stderr);
        std::fputs(message, stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);
    }
} // namespace syclfft::detail
