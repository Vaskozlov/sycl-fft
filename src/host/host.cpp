#include <syclfft/host.hpp>

#include <syclfft/detail/provider_loader.hpp>

#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <type_traits>

namespace syclfft::host {
namespace {

template <class Scalar> constexpr syclfft_scalar_v1 scalar_id() {
  if constexpr (std::is_same_v<Scalar, float>)
    return SYCLFFT_SCALAR_FLOAT_V1;
  return SYCLFFT_SCALAR_DOUBLE_V1;
}

std::size_t transform_size(const std::vector<std::size_t> &shape) {
  std::size_t result = 1;
  for (const auto length : shape)
    result *= length;
  return result;
}

} // namespace

template <class Scalar> class plan<Scalar>::impl {
public:
  impl(std::vector<std::size_t> lengths, std::size_t batch_count,
       syclfft::direction dir, plan_options opts)
      : lengths_(std::move(lengths)), batch_count_(batch_count),
        direction_(dir), options_(opts) {
    element_count_ = detail::checked_element_count(lengths_, batch_count_);
    if (options_.preferred_provider != provider::automatic &&
        options_.preferred_provider != provider::fftw) {
      throw exception(error_code::provider_unavailable,
                      "Host plans only support the FFTW provider");
    }
    auto loaded = detail::load_host_provider("syclfft_provider_fftw");
    api_ = loaded.api;
    module_ = std::move(loaded.module);

    syclfft_host_plan_config_v1 config{};
    config.abi_version = SYCLFFT_PROVIDER_ABI_VERSION;
    config.scalar = scalar_id<Scalar>();
    config.direction = static_cast<std::int32_t>(direction_);
    config.rank = static_cast<std::uint32_t>(lengths_.size());
    for (std::size_t i = 0; i < lengths_.size(); ++i) {
      config.lengths[i] = static_cast<std::uint64_t>(lengths_[i]);
    }
    config.batch_count = static_cast<std::uint64_t>(batch_count_);
    config.in_place = options_.placement == placement::in_place;
    config.measure = options_.planning == planning_mode::measure;

    std::array<char, 1024> error{};
    handle_ = api_->create(&config, error.data(), error.size());
    if (!handle_) {
      throw exception(error_code::planning_failed,
                      error[0] ? error.data() : "FFTW plan creation failed");
    }
  }

  ~impl() {
    if (handle_)
      api_->destroy(handle_);
  }

  void execute(const complex_type *input, complex_type *output,
               bool one_pointer) {
    if (!input || !output) {
      throw exception(error_code::invalid_pointer,
                      "FFT input and output must be non-null");
    }
    if (options_.placement == placement::in_place && !one_pointer) {
      throw exception(error_code::invalid_argument,
                      "In-place plan must be executed with execute(inout)");
    }
    if (options_.placement == placement::out_of_place && one_pointer) {
      throw exception(
          error_code::invalid_argument,
          "Out-of-place plan requires distinct input and output arguments");
    }

    std::lock_guard lock(mutex_);
    std::array<char, 1024> error{};
    if (api_->execute(handle_, const_cast<complex_type *>(input), output,
                      error.data(), error.size()) != 0) {
      throw exception(error_code::execution_failed,
                      error[0] ? error.data() : "FFTW execution failed");
    }

    const auto scale = static_cast<Scalar>(detail::normalization_scale(
        options_.normalization, direction_, transform_size(lengths_)));
    if (scale != Scalar{1}) {
      for (std::size_t i = 0; i < element_count_; ++i)
        output[i] *= scale;
    }
  }

  std::vector<std::size_t> lengths_;
  std::size_t batch_count_{};
  std::size_t element_count_{};
  syclfft::direction direction_{};
  plan_options options_{};
  const syclfft_host_provider_v1 *api_{};
  std::shared_ptr<void> module_;
  void *handle_{};
  std::mutex mutex_;
};

template <class Scalar>
plan<Scalar>::plan(std::vector<std::size_t> lengths, std::size_t batch_count,
                   syclfft::direction direction, plan_options options)
    : impl_(std::make_unique<impl>(std::move(lengths), batch_count, direction,
                                   options)) {}

template <class Scalar> plan<Scalar>::plan(plan &&) noexcept = default;
template <class Scalar>
plan<Scalar> &plan<Scalar>::operator=(plan &&) noexcept = default;
template <class Scalar> plan<Scalar>::~plan() = default;

template <class Scalar> void plan<Scalar>::execute(complex_type *inout) {
  if (!impl_)
    throw exception(error_code::invalid_state,
                    "Cannot execute a moved-from plan");
  impl_->execute(inout, inout, true);
}

template <class Scalar>
void plan<Scalar>::execute(const complex_type *input, complex_type *output) {
  if (!impl_)
    throw exception(error_code::invalid_state,
                    "Cannot execute a moved-from plan");
  impl_->execute(input, output, false);
}

template <class Scalar>
std::span<const std::size_t> plan<Scalar>::shape() const noexcept {
  return impl_ ? std::span<const std::size_t>{impl_->lengths_}
               : std::span<const std::size_t>{};
}

template <class Scalar> std::size_t plan<Scalar>::batch_count() const noexcept {
  return impl_ ? impl_->batch_count_ : 0;
}

template <class Scalar>
syclfft::direction plan<Scalar>::direction() const noexcept {
  return impl_ ? impl_->direction_ : syclfft::direction::forward;
}

template <class Scalar>
syclfft::provider plan<Scalar>::selected_provider() const noexcept {
  return impl_ ? provider::fftw : provider::automatic;
}

template <class Scalar>
std::size_t plan<Scalar>::scratch_size_bytes() const noexcept {
  if (!impl_)
    return 0;
  const auto buffers =
      impl_->options_.placement == placement::in_place ? 1u : 2u;
  return buffers * impl_->element_count_ * sizeof(complex_type);
}

template class plan<float>;
template class plan<double>;

} // namespace syclfft::host
