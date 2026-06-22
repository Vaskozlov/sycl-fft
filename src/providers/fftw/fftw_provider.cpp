#include <algorithm>
#include <array>
#include <complex>
#include <cstring>
#include <fftw3.h>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <syclfft/detail/provider_abi.h>
#include <syclfft/detail/provider_error_buffer.hpp>
#include <type_traits>
#include <vector>

namespace
{

    std::mutex &fftw_planner_mutex()
    {
        static std::mutex mutex;
        return mutex;
    }

    template <class Scalar>
    struct fftw_traits;

    template <>
    struct fftw_traits<double>
    {
        using plan_type = fftw_plan;
        using complex_type = fftw_complex;

        static void *allocate(std::size_t bytes)
        {
            return fftw_malloc(bytes);
        }

        static void deallocate(void *ptr)
        {
            fftw_free(ptr);
        }

        static plan_type plan_many(
            int rank, const int *lengths, int batch, int distance, complex_type *input,
            complex_type *output, int sign, unsigned flags)
        {
            return fftw_plan_many_dft(
                rank,
                lengths,
                batch,
                input,
                nullptr,
                1,
                distance,
                output,
                nullptr,
                1,
                distance,
                sign,
                flags);
        }

        static void execute(plan_type plan, complex_type *input, complex_type *output)
        {
            fftw_execute_dft(plan, input, output);
        }

        static void destroy(plan_type plan)
        {
            fftw_destroy_plan(plan);
        }

        static int alignment(double *ptr)
        {
            return fftw_alignment_of(ptr);
        }
    };

    template <>
    struct fftw_traits<float>
    {
        using plan_type = fftwf_plan;
        using complex_type = fftwf_complex;

        static void *allocate(std::size_t bytes)
        {
            return fftwf_malloc(bytes);
        }

        static void deallocate(void *ptr)
        {
            fftwf_free(ptr);
        }

        static plan_type plan_many(
            int rank, const int *lengths, int batch, int distance, complex_type *input,
            complex_type *output, int sign, unsigned flags)
        {
            return fftwf_plan_many_dft(
                rank,
                lengths,
                batch,
                input,
                nullptr,
                1,
                distance,
                output,
                nullptr,
                1,
                distance,
                sign,
                flags);
        }

        static void execute(plan_type plan, complex_type *input, complex_type *output)
        {
            fftwf_execute_dft(plan, input, output);
        }

        static void destroy(plan_type plan)
        {
            fftwf_destroy_plan(plan);
        }

        static int alignment(float *ptr)
        {
            return fftwf_alignment_of(ptr);
        }
    };

    template <class Scalar>
    class fftw_plan_holder
    {
        using traits = fftw_traits<Scalar>;
        using native_plan = typename traits::plan_type;
        using native_complex = typename traits::complex_type;

    public:
        explicit fftw_plan_holder(const syclfft_host_plan_config_v1 &config)
          : in_place_(config.in_place != 0)
        {
            static_assert(sizeof(std::complex<Scalar>) == sizeof(native_complex));
            if (config.rank == 0 || config.rank > 3 || config.batch_count == 0) {
                throw std::runtime_error("Invalid FFTW rank or batch count");
            }
            auto elements = static_cast<std::size_t>(config.batch_count);
            for (std::uint32_t i = 0; i < config.rank; ++i) {
                if (config.lengths[i] == 0
                    || config.lengths[i]
                           > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("FFTW dimensions must fit in int and be non-zero");
                }
                if (elements > std::numeric_limits<std::size_t>::max()
                                   / static_cast<std::size_t>(config.lengths[i])) {
                    throw std::runtime_error("FFTW element count overflow");
                }
                elements *= static_cast<std::size_t>(config.lengths[i]);
                lengths_[i] = static_cast<int>(config.lengths[i]);
            }
            if (config.batch_count > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("FFTW batch count must fit in int");
            }
            rank_ = static_cast<int>(config.rank);
            batch_ = static_cast<int>(config.batch_count);
            const auto distance = elements / static_cast<std::size_t>(batch_);
            if (distance > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("FFTW transform size must fit in int");
            }
            distance_ = static_cast<int>(distance);
            if (config.direction != SYCLFFT_DIRECTION_FORWARD_V1
                && config.direction != SYCLFFT_DIRECTION_BACKWARD_V1) {
                throw std::runtime_error("Invalid FFTW direction");
            }
            sign_ = config.direction == SYCLFFT_DIRECTION_FORWARD_V1 ? FFTW_FORWARD : FFTW_BACKWARD;
            flags_ = config.measure ? FFTW_MEASURE : FFTW_ESTIMATE;

            const auto bytes = elements * sizeof(native_complex);
            planning_input_.reset(traits::allocate(bytes));
            if (!planning_input_) {
                throw std::bad_alloc{};
            }
            if (in_place_) {
                planning_output_alias_ = planning_input_.get();
            } else {
                planning_output_.reset(traits::allocate(bytes));
                if (!planning_output_) {
                    throw std::bad_alloc{};
                }
                planning_output_alias_ = planning_output_.get();
            }

            {
                std::scoped_lock lock(fftw_planner_mutex());
                aligned_ = make_plan(flags_);
            }
            if (!aligned_) {
                throw std::runtime_error("fftw_plan_many_dft returned null");
            }
            input_alignment_ = traits::alignment(static_cast<Scalar *>(planning_input_.get()));
            output_alignment_ = traits::alignment(static_cast<Scalar *>(planning_output_alias_));
        }

        fftw_plan_holder(const fftw_plan_holder &) = delete;
        fftw_plan_holder &operator=(const fftw_plan_holder &) = delete;
        fftw_plan_holder(fftw_plan_holder &&) = delete;
        fftw_plan_holder &operator=(fftw_plan_holder &&) = delete;

        ~fftw_plan_holder()
        {
            std::scoped_lock lock(fftw_planner_mutex());
            if (aligned_) {
                traits::destroy(aligned_);
            }
            if (unaligned_) {
                traits::destroy(unaligned_);
            }
        }

        void execute(void *input, void *output)
        {
            auto *native_input = reinterpret_cast<native_complex *>(input);
            auto *native_output = reinterpret_cast<native_complex *>(output);
            const bool aligned =
                traits::alignment(reinterpret_cast<Scalar *>(input)) == input_alignment_
                && traits::alignment(reinterpret_cast<Scalar *>(output)) == output_alignment_;
            native_plan selected = aligned_;
            if (!aligned) {
                if (!unaligned_) {
                    std::scoped_lock lock(fftw_planner_mutex());
                    unaligned_ = make_plan(flags_ | FFTW_UNALIGNED);
                    if (!unaligned_) {
                        throw std::runtime_error("Unable to create unaligned FFTW plan");
                    }
                }
                selected = unaligned_;
            }
            traits::execute(selected, native_input, native_output);
        }

    private:
        struct deleter
        {
            void operator()(void *ptr) const
            {
                traits::deallocate(ptr);
            }
        };

        native_plan make_plan(unsigned flags)
        {
            return traits::plan_many(
                rank_,
                lengths_.data(),
                batch_,
                distance_,
                static_cast<native_complex *>(planning_input_.get()),
                static_cast<native_complex *>(planning_output_alias_),
                sign_,
                flags);
        }

        std::array<int, 3> lengths_{};
        int rank_{};
        int batch_{};
        int distance_{};
        int sign_{};
        unsigned flags_{};
        bool in_place_{};
        int input_alignment_{};
        int output_alignment_{};
        std::unique_ptr<void, deleter> planning_input_;
        std::unique_ptr<void, deleter> planning_output_;
        void *planning_output_alias_{};
        native_plan aligned_{};
        native_plan unaligned_{};
    };

    struct erased_plan
    {
        erased_plan() = default;
        erased_plan(const erased_plan &) = delete;
        erased_plan &operator=(const erased_plan &) = delete;
        erased_plan(erased_plan &&) = delete;
        erased_plan &operator=(erased_plan &&) = delete;
        virtual ~erased_plan() = default;
        virtual void execute(void *, void *) = 0;
    };

    template <class Scalar>
    struct erased_plan_impl final : erased_plan
    {
        explicit erased_plan_impl(const syclfft_host_plan_config_v1 &config)
          : plan(config)
        {}

        void execute(void *input, void *output) override
        {
            plan.execute(input, output);
        }

        fftw_plan_holder<Scalar> plan;
    };

    void *create_plan(const syclfft_host_plan_config_v1 *config, char *error, size_t capacity)
    {
        try {
            if (!config || config->abi_version != SYCLFFT_PROVIDER_ABI_VERSION) {
                throw std::runtime_error("Invalid FFTW provider configuration ABI");
            }
            if (config->scalar == SYCLFFT_SCALAR_FLOAT_V1) {
                return new erased_plan_impl<float>(*config);
            }
            if (config->scalar == SYCLFFT_SCALAR_DOUBLE_V1) {
                return new erased_plan_impl<double>(*config);
            }
            throw std::runtime_error("Unsupported FFTW scalar type");
        } catch (const std::exception &ex) {
            syclfft::detail::set_error_buffer(error, capacity, ex.what());
            return nullptr;
        }
    }

    void destroy_plan(void *plan)
    {
        delete static_cast<erased_plan *>(plan);
    }

    int execute_plan(void *plan, void *input, void *output, char *error, size_t capacity)
    {
        try {
            if (!plan || !input || !output) {
                throw std::runtime_error("Invalid FFTW execution argument");
            }
            static_cast<erased_plan *>(plan)->execute(input, output);
            return 0;
        } catch (const std::exception &ex) {
            syclfft::detail::set_error_buffer(error, capacity, ex.what());
            return 1;
        }
    }

    const syclfft_host_provider_v1 provider_api{
        SYCLFFT_PROVIDER_ABI_VERSION,
        "fftw",
        &create_plan,
        &destroy_plan,
        &execute_plan,
    };

} // namespace

SYCLFFT_PROVIDER_EXPORT const syclfft_host_provider_v1 *syclfft_get_host_provider_v1()
{
    return &provider_api;
}
