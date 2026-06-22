#pragma once

#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>

namespace syclfft
{

    template <class T>
    class span
    {
    public:
        using element_type = T;
        using value_type = std::remove_cv_t<T>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using pointer = T *;
        using reference = T &;
        using iterator = pointer;
        using reverse_iterator = std::reverse_iterator<iterator>;

        constexpr span() noexcept
          : data_(nullptr)
          , size_(0)
        {}

        constexpr span(pointer data, size_type size) noexcept
          : data_(data)
          , size_(size)
        {}

        constexpr span(pointer first, pointer last) noexcept
          : data_(first)
          , size_(first == last ? 0 : static_cast<size_type>(last - first))
        {}

        template <class U, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
        constexpr span(const span<U> &other) noexcept
          : data_(other.data())
          , size_(other.size())
        {}

        template <
            class U, std::size_t Size, std::enable_if_t<std::is_convertible_v<U *, T *>, int> = 0>
        constexpr span(U (&array)[Size]) noexcept // NOLINT(modernize-avoid-c-arrays)
          : data_(array)
          , size_(Size)
        {}

        template <
            class Container,
            std::enable_if_t<
                !std::is_same_v<std::remove_cv_t<Container>, span>
                    && std::is_convertible_v<decltype(std::declval<Container &>().data()), T *>,
                int> = 0>
        constexpr span(Container &container) noexcept
          : data_(container.data())
          , size_(static_cast<size_type>(container.size()))
        {}

        constexpr pointer data() const noexcept
        {
            return data_;
        }

        constexpr size_type size() const noexcept
        {
            return size_;
        }

        constexpr bool empty() const noexcept
        {
            return size_ == 0;
        }

        constexpr reference operator[](size_type index) const noexcept
        {
            return data_[index];
        }

        constexpr iterator begin() const noexcept
        {
            return data_;
        }

        constexpr iterator end() const noexcept
        {
            return size_ == 0 ? data_ : data_ + size_;
        }

        constexpr reverse_iterator rbegin() const noexcept
        {
            return reverse_iterator{end()};
        }

        constexpr reverse_iterator rend() const noexcept
        {
            return reverse_iterator{begin()};
        }

        constexpr span first(size_type count) const noexcept
        {
            return span{data_, count};
        }

        constexpr span last(size_type count) const noexcept
        {
            return count == 0 ? span{end(), size_type{0}} : span{data_ + size_ - count, count};
        }

        constexpr span subspan(size_type offset, size_type count) const noexcept
        {
            return span{offset == 0 ? data_ : data_ + offset, count};
        }

        constexpr span subspan(size_type offset) const noexcept
        {
            return span{offset == 0 ? data_ : data_ + offset, size_ - offset};
        }

    private:
        pointer data_;
        size_type size_;
    };

    template <class T, std::size_t Size>
    span(T (&)[Size]) -> span<T>; // NOLINT(modernize-avoid-c-arrays)

} // namespace syclfft
