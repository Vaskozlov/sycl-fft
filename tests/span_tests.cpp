#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <syclfft/span.hpp>
#include <vector>

namespace
{

    void check(bool condition, const char *message)
    {
        if (!condition) {
            throw std::runtime_error(message);
        }
    }

    int sum(syclfft::span<const int> values)
    {
        int result = 0;
        for (const auto value : values) {
            result += value;
        }
        return result;
    }

} // namespace

int main()
try {
    int values[]{1, 2, 3, 4}; // NOLINT(modernize-avoid-c-arrays)
    syclfft::span<int> mutable_values{values};
    syclfft::span<const int> const_values{mutable_values};
    check(const_values.data() == values, "converting span changed the data pointer");
    check(const_values.size() == 4, "array span has the wrong size");
    check(sum(syclfft::span<int>{values}) == 10, "temporary converting span failed");

    const std::array<int, 3> array_values{
        {5, 6, 7}
    };
    syclfft::span<const int> array_span{array_values};
    check(array_span.size() == array_values.size(), "std::array span has the wrong size");

    const std::vector<int> vector_values{8, 9, 10};
    syclfft::span<const int> vector_span{vector_values};
    check(vector_span.first(2)[1] == 9, "span::first returned the wrong view");
    check(vector_span.last(2)[0] == 9, "span::last returned the wrong view");
    check(vector_span.subspan(1)[1] == 10, "span::subspan returned the wrong view");

    const syclfft::span<const int> empty;
    // Check every observer independently, including size(), rather than only truthiness.
    // NOLINTBEGIN(readability-container-size-empty)
    check(
        empty.empty() && empty.size() == 0 && empty.data() == nullptr, "default span is not empty");
    // NOLINTEND(readability-container-size-empty)
    check(empty.begin() == empty.end(), "empty span iterators differ");
    check(empty.last(0).empty(), "empty span::last is not empty");
    check(empty.subspan(0).empty(), "empty span::subspan is not empty");

    std::cout << "span tests passed\n";
    return 0;
} catch (const std::exception &ex) {
    std::cerr << "span tests failed: " << ex.what() << '\n';
    return 1;
}
