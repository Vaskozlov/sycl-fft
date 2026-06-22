#pragma once

#include <complex>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <span>
#include <syclfft/common.hpp>
#include <type_traits>
#include <vector>

namespace syclfft::host
{

    template <class Scalar>
    class plan
    {
        static_assert(
            std::is_same_v<Scalar, float> || std::is_same_v<Scalar, double>,
            "syclfft supports float and double precision");

    public:
        using scalar_type = Scalar;
        using complex_type = std::complex<Scalar>;

        plan(const plan &) = delete;
        plan &operator=(const plan &) = delete;
        plan(plan &&) noexcept;
        plan &operator=(plan &&) noexcept;
        ~plan();

        void execute(complex_type *inout);
        void execute(const complex_type *input, complex_type *output);

        [[nodiscard]] std::span<const std::size_t> shape() const noexcept;
        [[nodiscard]] std::size_t batch_count() const noexcept;
        [[nodiscard]] syclfft::direction direction() const noexcept;
        [[nodiscard]] syclfft::provider selected_provider() const noexcept;
        [[nodiscard]] std::size_t scratch_size_bytes() const noexcept;

        plan(
            std::vector<std::size_t> lengths, std::size_t batch_count, syclfft::direction direction,
            plan_options options);

    private:
        class impl;
        std::unique_ptr<impl> impl_;
    };

    template <class Scalar>
    plan<Scalar> plan_many_dft(
        std::span<const std::size_t> lengths, std::size_t batch_count, syclfft::direction direction,
        plan_options options = {})
    {
        return plan<Scalar>({lengths.begin(), lengths.end()}, batch_count, direction, options);
    }

    template <class Scalar>
    plan<Scalar> plan_many_dft(
        std::initializer_list<std::size_t> lengths,
        std::size_t batch_count,
        syclfft::direction direction,
        plan_options options = {})
    {
        return plan<Scalar>({lengths.begin(), lengths.end()}, batch_count, direction, options);
    }

    template <class Scalar>
    plan<Scalar> plan_dft_1d(std::size_t n, syclfft::direction direction, plan_options options = {})
    {
        return plan_many_dft<Scalar>({n}, 1, direction, options);
    }

    template <class Scalar>
    plan<Scalar> plan_dft_2d(
        std::size_t n0, std::size_t n1, syclfft::direction direction, plan_options options = {})
    {
        return plan_many_dft<Scalar>({n0, n1}, 1, direction, options);
    }

    template <class Scalar>
    plan<Scalar> plan_dft_3d(
        std::size_t n0, std::size_t n1, std::size_t n2, syclfft::direction direction,
        plan_options options = {})
    {
        return plan_many_dft<Scalar>({n0, n1, n2}, 1, direction, options);
    }

#if !defined(SYCLFFT_BUILDING_HOST)
#    if defined(_WIN32)
    extern template class SYCLFFT_HOST_EXPORT plan<float>;
    extern template class SYCLFFT_HOST_EXPORT plan<double>;
#    else
    extern template class plan<float>;
    extern template class plan<double>;
#    endif
#endif

} // namespace syclfft::host
