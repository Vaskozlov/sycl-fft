#include <syclfft/host.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Scalar>
std::vector<std::complex<Scalar>>
reference_axis(const std::vector<std::complex<Scalar>> &input,
               std::span<const std::size_t> shape, std::size_t batch_count,
               std::size_t axis, syclfft::direction direction) {
  auto output = input;
  std::size_t transform_size = 1;
  for (auto n : shape)
    transform_size *= n;
  std::size_t stride = 1;
  for (std::size_t i = axis + 1; i < shape.size(); ++i)
    stride *= shape[i];
  const auto n = shape[axis];
  const auto sign =
      direction == syclfft::direction::forward ? Scalar{-1} : Scalar{1};
  const auto pi = std::acos(Scalar{-1});
  for (std::size_t batch = 0; batch < batch_count; ++batch) {
    const auto offset = batch * transform_size;
    for (std::size_t linear = 0; linear < transform_size; ++linear) {
      const auto coordinate = (linear / stride) % n;
      const auto base = linear - coordinate * stride;
      std::complex<Scalar> sum{};
      for (std::size_t k = 0; k < n; ++k) {
        const auto angle = sign * Scalar{2} * pi *
                           static_cast<Scalar>(coordinate * k) /
                           static_cast<Scalar>(n);
        sum += input[offset + base + k * stride] *
               std::complex<Scalar>{std::cos(angle), std::sin(angle)};
      }
      output[offset + linear] = sum;
    }
  }
  return output;
}

template <class Scalar>
std::vector<std::complex<Scalar>>
reference(std::vector<std::complex<Scalar>> values,
          std::span<const std::size_t> shape, std::size_t batch_count,
          syclfft::direction direction, syclfft::normalization normalization) {
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    values = reference_axis(values, shape, batch_count, axis, direction);
  }
  std::size_t n = 1;
  for (auto length : shape)
    n *= length;
  const auto scale =
      syclfft::detail::normalization_scale(normalization, direction, n);
  for (auto &value : values)
    value *= static_cast<Scalar>(scale);
  return values;
}

template <class Scalar>
void compare(const std::vector<std::complex<Scalar>> &actual,
             const std::vector<std::complex<Scalar>> &expected,
             Scalar tolerance, const std::string &label) {
  check(actual.size() == expected.size(), label + ": size mismatch");
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const auto error = std::abs(actual[i] - expected[i]);
    const auto bound = tolerance * (Scalar{1} + std::abs(expected[i]));
    if (error > bound) {
      throw std::runtime_error(label + ": mismatch at element " +
                               std::to_string(i));
    }
  }
}

template <class Scalar>
void run_case(std::vector<std::size_t> shape, std::size_t batches,
              syclfft::direction direction,
              syclfft::normalization normalization,
              syclfft::placement placement) {
  std::size_t count = batches;
  for (auto n : shape)
    count *= n;
  std::vector<std::complex<Scalar>> input(count);
  for (std::size_t i = 0; i < count; ++i) {
    input[i] = {static_cast<Scalar>((i * 7) % 13) / Scalar{7},
                static_cast<Scalar>((i * 5 + 3) % 11) / Scalar{9}};
  }
  const auto expected =
      reference(input, shape, batches, direction, normalization);
  auto fft = syclfft::host::plan_many_dft<Scalar>(
      shape, batches, direction,
      {.placement = placement,
       .normalization = normalization,
       .preferred_provider = syclfft::provider::fftw});
  check(fft.selected_provider() == syclfft::provider::fftw,
        "wrong host provider");
  check(std::equal(fft.shape().begin(), fft.shape().end(), shape.begin()),
        "wrong shape");
  check(fft.batch_count() == batches, "wrong batch count");
  if (placement == syclfft::placement::in_place) {
    auto actual = input;
    fft.execute(actual.data());
    compare(actual, expected,
            std::is_same_v<Scalar, float> ? Scalar{2e-4} : Scalar{1e-11},
            "in-place");
  } else {
    std::vector<std::complex<Scalar>> actual(count);
    fft.execute(input.data(), actual.data());
    compare(actual, expected,
            std::is_same_v<Scalar, float> ? Scalar{2e-4} : Scalar{1e-11},
            "out-of-place");
  }
}

template <class Function>
void expect_error(syclfft::error_code code, Function &&function) {
  try {
    function();
  } catch (const syclfft::exception &ex) {
    check(ex.code() == code, "wrong exception code");
    return;
  }
  throw std::runtime_error("expected syclfft::exception");
}

} // namespace

int main() try {
  for (auto normalization :
       {syclfft::normalization::none, syclfft::normalization::forward,
        syclfft::normalization::backward, syclfft::normalization::orthogonal}) {
    run_case<float>({8}, 2, syclfft::direction::forward, normalization,
                    syclfft::placement::out_of_place);
    run_case<double>({3, 5}, 1, syclfft::direction::backward, normalization,
                     syclfft::placement::in_place);
  }
  run_case<float>({2, 3, 4}, 2, syclfft::direction::backward,
                  syclfft::normalization::none,
                  syclfft::placement::out_of_place);

  {
    std::vector<std::complex<double>> input(13), spectrum(13), restored(13);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = {static_cast<double>(i) / 7.0,
                  static_cast<double>(i % 3) / 5.0};
    auto forward = syclfft::host::plan_dft_1d<double>(
        input.size(), syclfft::direction::forward);
    auto backward = syclfft::host::plan_dft_1d<double>(
        input.size(), syclfft::direction::backward,
        {.normalization = syclfft::normalization::backward});
    forward.execute(input.data(), spectrum.data());
    backward.execute(spectrum.data(), restored.data());
    compare(restored, input, 2e-11, "host round trip");
  }

  expect_error(syclfft::error_code::invalid_argument, [] {
    (void)syclfft::host::plan_dft_1d<float>(0, syclfft::direction::forward);
  });
  expect_error(syclfft::error_code::invalid_argument, [] {
    (void)syclfft::host::plan_many_dft<float>(
        {std::numeric_limits<std::size_t>::max(), 2}, 1,
        syclfft::direction::forward);
  });
  expect_error(syclfft::error_code::invalid_argument, [] {
    (void)syclfft::host::plan_many_dft<float>({2, 2, 2, 2}, 1,
                                              syclfft::direction::forward);
  });
  expect_error(syclfft::error_code::invalid_pointer, [] {
    auto fft =
        syclfft::host::plan_dft_1d<float>(4, syclfft::direction::forward);
    fft.execute(nullptr, nullptr);
  });
  expect_error(syclfft::error_code::invalid_argument, [] {
    auto fft = syclfft::host::plan_dft_1d<float>(
        4, syclfft::direction::forward,
        {.placement = syclfft::placement::in_place});
    std::vector<std::complex<float>> input(4), output(4);
    fft.execute(input.data(), output.data());
  });
  expect_error(syclfft::error_code::invalid_state, [] {
    auto original =
        syclfft::host::plan_dft_1d<float>(4, syclfft::direction::forward);
    auto moved = std::move(original);
    (void)moved;
    original.execute(nullptr, nullptr);
  });

  std::cout << "host tests passed\n";
  return 0;
} catch (const std::exception &ex) {
  std::cerr << "host tests failed: " << ex.what() << '\n';
  return 1;
}
