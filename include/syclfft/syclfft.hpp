#pragma once

#include <array>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <sycl/sycl.hpp>
#include <syclfft/common.hpp>
#include <type_traits>
#include <vector>

// SyclCPLX 32684c7 uses three DPC++ detail utilities in group algorithms.
// AdaptiveCpp does not provide those non-standard names, so supply the minimal
// compatibility surface before including the unchanged upstream header.
#if defined(SYCLFFT_USE_ADAPTIVECPP)
namespace hipsycl::sycl::detail
{
    template <class T>
    using is_pointer = std::is_pointer<T>;
    template <class T>
    using remove_pointer = std::remove_pointer<T>;
    template <class T>
    using remove_pointer_t = std::remove_pointer_t<T>;

    template <class Group, class Ptr, class Function>
    inline void for_each(Group group, Ptr first, Ptr last, Function function)
    {
        auto offset = static_cast<std::ptrdiff_t>(group.get_local_linear_id());
        const auto stride = static_cast<std::ptrdiff_t>(group.get_local_linear_range());
        for (auto index = offset; index < last - first; index += stride) {
            function(first[index]);
        }
    }
} // namespace hipsycl::sycl::detail
#endif

#include <syclfft/detail/sycl_ext_complex.hpp>

namespace syclfft
{

    template <class Scalar>
    using complex = sycl::ext::cplx::complex<Scalar>;

    template <class Scalar>
    class plan
    {
        static_assert(
            std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
            "syclfft supports float and double precision");

    public:
        using scalar_type = Scalar;
        using complex_type = complex<Scalar>;

        plan(const plan &) = delete;
        plan &operator=(const plan &) = delete;
        plan(plan &&) noexcept;
        plan &operator=(plan &&) noexcept;
        ~plan();

        sycl::event execute(complex_type *inout);
        sycl::event execute(complex_type *inout, const sycl::event &dependency);
        sycl::event execute(complex_type *inout, syclfft::span<const sycl::event> dependencies);

        sycl::event execute(const complex_type *input, complex_type *output);
        sycl::event
            execute(const complex_type *input, complex_type *output, const sycl::event &dependency);
        sycl::event execute(
            const complex_type *input, complex_type *output,
            syclfft::span<const sycl::event> dependencies);

        [[nodiscard]] syclfft::span<const std::size_t> shape() const noexcept;
        [[nodiscard]] std::size_t batch_count() const noexcept;
        [[nodiscard]] syclfft::direction direction() const noexcept;
        [[nodiscard]] syclfft::provider selected_provider() const noexcept;
        [[nodiscard]] std::size_t scratch_size_bytes() const noexcept;

        plan(
            sycl::queue queue, std::vector<std::size_t> lengths, std::size_t batch_count,
            syclfft::direction direction, plan_options options);

    private:
        class impl;
        std::unique_ptr<impl> impl_;
    };

    SYCLFFT_SYCL_EXPORT std::vector<provider_status> query_providers(const sycl::queue &queue);

    template <class Scalar>
    plan<Scalar> plan_many_dft(
        sycl::queue queue, syclfft::span<const std::size_t> lengths, std::size_t batch_count,
        syclfft::direction direction, plan_options options = {})
    {
        return plan<Scalar>(
            std::move(queue), {lengths.begin(), lengths.end()}, batch_count, direction, options);
    }

    template <class Scalar>
    plan<Scalar> plan_many_dft(
        sycl::queue queue, std::initializer_list<std::size_t> lengths, std::size_t batch_count,
        syclfft::direction direction, plan_options options = {})
    {
        return plan<Scalar>(
            std::move(queue), {lengths.begin(), lengths.end()}, batch_count, direction, options);
    }

    template <class Scalar>
    plan<Scalar> plan_dft_1d(
        sycl::queue queue, std::size_t n, syclfft::direction direction, plan_options options = {})
    {
        return plan_many_dft<Scalar>(std::move(queue), {n}, 1, direction, options);
    }

    template <class Scalar>
    plan<Scalar> plan_dft_2d(
        sycl::queue queue, std::size_t n0, std::size_t n1, syclfft::direction direction,
        plan_options options = {})
    {
        return plan_many_dft<Scalar>(std::move(queue), {n0, n1}, 1, direction, options);
    }

    template <class Scalar>
    plan<Scalar> plan_dft_3d(
        sycl::queue queue, std::size_t n0, std::size_t n1, std::size_t n2,
        syclfft::direction direction, plan_options options = {})
    {
        return plan_many_dft<Scalar>(std::move(queue), {n0, n1, n2}, 1, direction, options);
    }

#if !defined(SYCLFFT_BUILDING_SYCL)
#    if defined(_WIN32)
    extern template class SYCLFFT_SYCL_EXPORT plan<float>;
    extern template class SYCLFFT_SYCL_EXPORT plan<double>;
#    else
    extern template class plan<float>;
    extern template class plan<double>;
#    endif
#endif

} // namespace syclfft
