#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>

namespace syclfft::detail
{
    inline void set_error_buffer(char *output, std::size_t capacity, const std::string &message)
    {
        if (!output || capacity == 0) {
            return;
        }
        const auto count = std::min(capacity - 1, message.size());
        std::memcpy(output, message.data(), count);
        output[count] = '\0';
    }
} // namespace syclfft::detail
