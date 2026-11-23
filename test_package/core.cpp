#include <syclfft/common.hpp>

int main()
{
    return syclfft::detail::checked_element_count({4, 8}, 3) == 96 ? 0 : 1;
}
