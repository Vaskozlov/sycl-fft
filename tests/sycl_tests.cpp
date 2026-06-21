#include <syclfft/syclfft.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

void check(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Scalar>
std::vector<std::complex<Scalar>>
direct_dft(const std::vector<std::complex<Scalar>> &input,
           syclfft::direction direction, syclfft::normalization normalization) {
  const auto count = input.size();
  std::vector<std::complex<Scalar>> output(count);
  const auto sign =
      direction == syclfft::direction::forward ? Scalar{-1} : Scalar{1};
  const auto pi = std::acos(Scalar{-1});
  for (std::size_t k = 0; k < count; ++k) {
    for (std::size_t n = 0; n < count; ++n) {
      const auto angle = sign * Scalar{2} * pi * static_cast<Scalar>(k * n) /
                         static_cast<Scalar>(count);
      output[k] +=
          input[n] * std::complex<Scalar>{std::cos(angle), std::sin(angle)};
    }
  }
  const auto scale = static_cast<Scalar>(
      syclfft::detail::normalization_scale(normalization, direction, count));
  for (auto &value : output)
    value *= scale;
  return output;
}

template <class Scalar>
std::vector<std::complex<Scalar>>
reference_many(std::vector<std::complex<Scalar>> values,
               std::span<const std::size_t> shape, std::size_t batch_count,
               syclfft::direction direction) {
  std::size_t transform_size = 1;
  for (const auto n : shape)
    transform_size *= n;
  const auto sign =
      direction == syclfft::direction::forward ? Scalar{-1} : Scalar{1};
  const auto pi = std::acos(Scalar{-1});
  for (std::size_t axis = 0; axis < shape.size(); ++axis) {
    auto output = values;
    std::size_t stride = 1;
    for (std::size_t i = axis + 1; i < shape.size(); ++i)
      stride *= shape[i];
    const auto length = shape[axis];
    for (std::size_t batch = 0; batch < batch_count; ++batch) {
      const auto offset = batch * transform_size;
      for (std::size_t linear = 0; linear < transform_size; ++linear) {
        const auto coordinate = (linear / stride) % length;
        const auto base = linear - coordinate * stride;
        std::complex<Scalar> sum{};
        for (std::size_t k = 0; k < length; ++k) {
          const auto angle = sign * Scalar{2} * pi *
                             static_cast<Scalar>(coordinate * k) /
                             static_cast<Scalar>(length);
          sum += values[offset + base + k * stride] *
                 std::complex<Scalar>{std::cos(angle), std::sin(angle)};
        }
        output[offset + linear] = sum;
      }
    }
    values = std::move(output);
  }
  return values;
}

template <class Scalar>
void run_many(sycl::queue &queue, std::vector<std::size_t> shape,
              std::size_t batch_count) {
  std::size_t count = batch_count;
  for (const auto n : shape)
    count *= n;
  std::vector<std::complex<Scalar>> source(count);
  for (std::size_t i = 0; i < count; ++i) {
    source[i] = {static_cast<Scalar>((i * 11 + 2) % 17) / Scalar{8},
                 static_cast<Scalar>((i * 3 + 1) % 13) / Scalar{7}};
  }
  const auto expected =
      reference_many(source, shape, batch_count, syclfft::direction::forward);
  auto *input = sycl::malloc_shared<syclfft::complex<Scalar>>(count, queue);
  auto *output = sycl::malloc_shared<syclfft::complex<Scalar>>(count, queue);
  for (std::size_t i = 0; i < count; ++i)
    input[i] = {source[i].real(), source[i].imag()};
  auto fft = syclfft::plan_many_dft<Scalar>(
      queue, shape, batch_count, syclfft::direction::forward,
      {.preferred_provider = syclfft::provider::portable_sycl});
  fft.execute(input, output).wait_and_throw();
  const auto tolerance =
      std::is_same_v<Scalar, float> ? Scalar{8e-4} : Scalar{5e-11};
  for (std::size_t i = 0; i < count; ++i) {
    const std::complex<Scalar> actual{output[i].real(), output[i].imag()};
    check(std::abs(actual - expected[i]) <=
              tolerance * (Scalar{1} + std::abs(expected[i])),
          "multidimensional or batch mismatch at element " + std::to_string(i));
  }
  sycl::free(input, queue);
  sycl::free(output, queue);
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

template <class Scalar>
void run_1d(sycl::queue &queue, std::size_t count, syclfft::direction direction,
            syclfft::normalization normalization,
            syclfft::placement placement) {
  std::vector<std::complex<Scalar>> source(count);
  for (std::size_t i = 0; i < count; ++i) {
    source[i] = {static_cast<Scalar>((i * 5 + 1) % 9) / Scalar{4},
                 static_cast<Scalar>((i * 3 + 2) % 7) / Scalar{5}};
  }
  const auto expected = direct_dft(source, direction, normalization);
  auto *input = sycl::malloc_shared<syclfft::complex<Scalar>>(count, queue);
  auto *output =
      placement == syclfft::placement::in_place
          ? input
          : sycl::malloc_shared<syclfft::complex<Scalar>>(count, queue);
  auto *repeated_output =
      placement == syclfft::placement::out_of_place
          ? sycl::malloc_shared<syclfft::complex<Scalar>>(count, queue)
          : nullptr;
  check(input && output &&
            (placement == syclfft::placement::in_place || repeated_output),
        "USM allocation failed");
  for (std::size_t i = 0; i < count; ++i)
    input[i] = {source[i].real(), source[i].imag()};
  auto fft = syclfft::plan_dft_1d<Scalar>(
      queue, count, direction,
      {.placement = placement,
       .normalization = normalization,
       .preferred_provider = syclfft::provider::portable_sycl});
  auto dependency =
      queue.submit([&](sycl::handler &handler) { handler.single_task([] {}); });
  auto event = placement == syclfft::placement::in_place
                   ? fft.execute(input, dependency)
                   : fft.execute(input, output, dependency);
  sycl::event repeated;
  if (repeated_output)
    repeated = fft.execute(input, repeated_output);
  event.wait_and_throw();
  if (repeated_output)
    repeated.wait_and_throw();
  const auto tolerance =
      std::is_same_v<Scalar, float> ? Scalar{4e-4} : Scalar{2e-11};
  for (std::size_t i = 0; i < count; ++i) {
    const std::complex<Scalar> actual{output[i].real(), output[i].imag()};
    if (std::abs(actual - expected[i]) >
        tolerance * (Scalar{1} + std::abs(expected[i]))) {
      throw std::runtime_error("portable mismatch at element " +
                               std::to_string(i));
    }
    if (repeated_output) {
      const std::complex<Scalar> repeated_actual{repeated_output[i].real(),
                                                 repeated_output[i].imag()};
      check(std::abs(repeated_actual - expected[i]) <=
                tolerance * (Scalar{1} + std::abs(expected[i])),
            "serialized repeated execution mismatch");
    }
  }
  check(fft.selected_provider() == syclfft::provider::portable_sycl,
        "wrong portable provider");
  check(fft.scratch_size_bytes() >=
            2 * count * sizeof(syclfft::complex<Scalar>),
        "portable scratch size is too small");
  if (output != input)
    sycl::free(output, queue);
  if (repeated_output)
    sycl::free(repeated_output, queue);
  sycl::free(input, queue);
}

} // namespace

int main() try {
  static_assert(sizeof(syclfft::complex<float>) == 2 * sizeof(float));
  static_assert(alignof(syclfft::complex<float>) == alignof(float));
  static_assert(std::is_trivially_copyable_v<syclfft::complex<float>>);
  struct native_complex_float {
    float real;
    float imag;
  };
  static_assert(sizeof(native_complex_float) ==
                sizeof(syclfft::complex<float>));
  static_assert(alignof(native_complex_float) ==
                alignof(syclfft::complex<float>));
  const std::complex<float> host_value{2.5f, -1.25f};
  const syclfft::complex<float> sycl_value{host_value};
  const std::complex<float> converted = sycl_value;
  check(converted == host_value, "SyclCPLX std::complex conversion failed");
  const auto native_value = std::bit_cast<native_complex_float>(sycl_value);
  check(native_value.real == host_value.real() &&
            native_value.imag == host_value.imag(),
        "SyclCPLX vendor complex layout mismatch");

  sycl::queue queue;
  for (auto normalization :
       {syclfft::normalization::none, syclfft::normalization::forward,
        syclfft::normalization::backward, syclfft::normalization::orthogonal}) {
    run_1d<float>(queue, 8, syclfft::direction::forward, normalization,
                  syclfft::placement::out_of_place);
  }
  run_1d<float>(queue, 7, syclfft::direction::backward,
                syclfft::normalization::none, syclfft::placement::in_place);
  run_1d<float>(queue, 6, syclfft::direction::forward,
                syclfft::normalization::none, syclfft::placement::out_of_place);
  run_1d<float>(queue, 15, syclfft::direction::backward,
                syclfft::normalization::none, syclfft::placement::out_of_place);
  run_1d<float>(queue, 35, syclfft::direction::forward,
                syclfft::normalization::none, syclfft::placement::out_of_place);
  run_1d<float>(queue, 11, syclfft::direction::backward,
                syclfft::normalization::none, syclfft::placement::in_place);
  run_many<float>(queue, {3, 4}, 2);
  run_many<float>(queue, {2, 3, 5}, 1);
  run_many<float>(queue, {11, 3}, 1);
  if (queue.get_device().has(sycl::aspect::fp64)) {
    run_1d<double>(queue, 11, syclfft::direction::forward,
                   syclfft::normalization::orthogonal,
                   syclfft::placement::out_of_place);
  } else {
    expect_error(syclfft::error_code::unsupported_precision, [&] {
      (void)syclfft::plan_dft_1d<double>(queue, 8, syclfft::direction::forward);
    });
  }

  {
    auto *input = sycl::malloc_shared<syclfft::complex<float>>(13, queue);
    auto *spectrum = sycl::malloc_shared<syclfft::complex<float>>(13, queue);
    auto *restored = sycl::malloc_shared<syclfft::complex<float>>(13, queue);
    for (std::size_t i = 0; i < 13; ++i)
      input[i] = {static_cast<float>(i) / 7.0f,
                  static_cast<float>(i % 3) / 5.0f};
    auto forward =
        syclfft::plan_dft_1d<float>(queue, 13, syclfft::direction::forward);
    auto backward = syclfft::plan_dft_1d<float>(
        queue, 13, syclfft::direction::backward,
        {.normalization = syclfft::normalization::backward});
    const auto forward_done = forward.execute(input, spectrum);
    backward.execute(spectrum, restored, forward_done).wait_and_throw();
    for (std::size_t i = 0; i < 13; ++i) {
      check(std::abs(static_cast<std::complex<float>>(restored[i]) -
                     static_cast<std::complex<float>>(input[i])) < 2e-3f,
            "portable round trip mismatch");
    }
    sycl::free(input, queue);
    sycl::free(spectrum, queue);
    sycl::free(restored, queue);
  }

  const auto providers = syclfft::query_providers(queue);
  check(!providers.empty() && providers.front().available,
        "portable provider unavailable");
  {
    auto automatic =
        syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
    check(automatic.selected_provider() == syclfft::provider::portable_sycl,
          "automatic CPU selection did not safely fall back to portable SYCL");
  }
  {
    auto *input_a = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    auto *input_b = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    auto *output_a = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    auto *output_b = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    for (std::size_t i = 0; i < 8; ++i) {
      input_a[i] = {i == 0 ? 1.0f : 0.0f, 0.0f};
      input_b[i] = {i == 1 ? 1.0f : 0.0f, 0.0f};
    }
    auto first =
        syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
    auto second =
        syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
    auto first_done = first.execute(input_a, output_a);
    auto second_done = second.execute(input_b, output_b);
    first_done.wait_and_throw();
    second_done.wait_and_throw();
    check(std::abs(output_a[0].real() - 1.0f) < 1e-5f &&
              std::abs(output_b[0].real() - 1.0f) < 1e-5f,
          "concurrent plan execution failed");
    sycl::free(input_a, queue);
    sycl::free(input_b, queue);
    sycl::free(output_a, queue);
    sycl::free(output_b, queue);
  }

  const sycl::context foreign_context{queue.get_device()};
  if (foreign_context != queue.get_context()) {
    sycl::queue foreign_queue{foreign_context, queue.get_device()};
    auto *foreign_input =
        sycl::malloc_shared<syclfft::complex<float>>(8, foreign_queue);
    auto *output = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    if (sycl::get_pointer_type(foreign_input, queue.get_context()) ==
        sycl::usm::alloc::unknown) {
      expect_error(syclfft::error_code::invalid_pointer, [&] {
        auto fft =
            syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
        fft.execute(foreign_input, output);
      });
    }
    sycl::free(foreign_input, foreign_queue);
    sycl::free(output, queue);
  }
  const auto fftw_status =
      std::find_if(providers.begin(), providers.end(),
                   [](const syclfft::provider_status &status) {
                     return status.id == syclfft::provider::fftw;
                   });
  if (fftw_status != providers.end() && fftw_status->available) {
    std::vector<std::complex<float>> source(8);
    source[1] = {1.0f, -0.5f};
    const auto expected = direct_dft(source, syclfft::direction::forward,
                                     syclfft::normalization::none);
    auto *input = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    auto *output = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    for (std::size_t i = 0; i < 8; ++i)
      input[i] = source[i];
    auto fft = syclfft::plan_dft_1d<float>(
        queue, 8, syclfft::direction::forward,
        {.preferred_provider = syclfft::provider::fftw});
    fft.execute(input, output).wait_and_throw();
    for (std::size_t i = 0; i < 8; ++i) {
      check(std::abs(static_cast<std::complex<float>>(output[i]) -
                     expected[i]) < 1e-4f,
            "explicit FFTW SYCL path mismatch");
    }
    sycl::free(input, queue);
    sycl::free(output, queue);
  }

  {
    auto *input = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    auto *output = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    auto first_half = queue.submit([&](sycl::handler &handler) {
      handler.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> i) {
        input[i[0]] = {i[0] == 0 ? 1.0f : 0.0f, 0.0f};
      });
    });
    auto second_half = queue.submit([&](sycl::handler &handler) {
      handler.parallel_for(sycl::range<1>{4}, [=](sycl::id<1> i) {
        input[i[0] + 4] = {0.0f, 0.0f};
      });
    });
    const std::array dependencies{first_half, second_half};
    auto fft =
        syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
    fft.execute(input, output, std::span<const sycl::event>{dependencies})
        .wait_and_throw();
    for (std::size_t i = 0; i < 8; ++i) {
      check(std::abs(output[i].real() - 1.0f) < 1e-5f &&
                std::abs(output[i].imag()) < 1e-5f,
            "event-span dependency ordering failed");
    }
    sycl::free(input, queue);
    sycl::free(output, queue);
  }

  expect_error(syclfft::error_code::provider_unavailable, [&] {
    (void)syclfft::plan_dft_1d<float>(
        queue, 8, syclfft::direction::forward,
        {.preferred_provider = syclfft::provider::cufft});
  });
  expect_error(syclfft::error_code::invalid_pointer, [&] {
    auto fft =
        syclfft::plan_dft_1d<float>(queue, 8, syclfft::direction::forward);
    fft.execute(nullptr, nullptr);
  });
  expect_error(syclfft::error_code::invalid_argument, [&] {
    auto fft = syclfft::plan_dft_1d<float>(
        queue, 8, syclfft::direction::forward,
        {.placement = syclfft::placement::in_place});
    auto *input = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    auto *output = sycl::malloc_shared<syclfft::complex<float>>(8, queue);
    try {
      fft.execute(input, output);
    } catch (...) {
      sycl::free(input, queue);
      sycl::free(output, queue);
      throw;
    }
  });
  expect_error(syclfft::error_code::invalid_state, [&] {
    auto original =
        syclfft::plan_dft_1d<float>(queue, 4, syclfft::direction::forward);
    auto moved = std::move(original);
    (void)moved;
    original.execute(nullptr, nullptr);
  });

  std::cout << "SYCL tests passed on "
            << queue.get_device().get_info<sycl::info::device::name>() << '\n';
  return 0;
} catch (const std::exception &ex) {
  std::cerr << "SYCL tests failed: " << ex.what() << '\n';
  return 1;
}
