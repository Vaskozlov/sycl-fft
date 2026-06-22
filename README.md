# syclfft

`syclfft` is a C++20 C2C FFT library with FFTW-style reusable RAII plans and a
unified USM interface for SYCL. Plans are direction-specific and
pointer-independent; execution returns native `sycl::event` objects.

Version 0.1 supports:

- `float` and `double` C2C transforms;
- contiguous 1D, 2D, and 3D transforms and contiguous batches;
- in-place and out-of-place execution;
- no normalization, forward/backward normalization, and orthogonal normalization;
- AdaptiveCpp and Open DPC++ on Linux and Windows;
- host-only FFTW on macOS;
- a portable pure-SYCL provider for CPU and OpenCL queues;
- an optional runtime-loaded cuFFT provider for CUDA queues.

R2C/C2R, arbitrary strides, split-complex storage, wisdom, and graph-capture
guarantees are outside the v0.1 scope. Dimensions use C order, so the last
dimension is contiguous.

## SYCL API

```cpp
#include <syclfft/syclfft.hpp>

sycl::queue queue;
auto* input = sycl::malloc_shared<syclfft::complex<float>>(1024, queue);
auto* output = sycl::malloc_shared<syclfft::complex<float>>(1024, queue);

auto fft = syclfft::plan_dft_1d<float>(
    queue, 1024, syclfft::direction::forward,
    {
        .placement = syclfft::placement::out_of_place,
        .normalization = syclfft::normalization::none,
        .preferred_provider = syclfft::provider::automatic,
    });

sycl::event done = fft.execute(input, output);
done.wait_and_throw();
```

`execute` also accepts one dependency event or a `std::span` of events. A plan
serializes executions which share its internal scratch. Create multiple plans
when transforms must run concurrently.

All device pointers must be USM allocations belonging to the plan queue's
context. The portable provider never copies transform data to the host. This is
also the OpenCL path in AdaptiveCpp: kernels operate directly on USM pointers,
without exposing an OpenCL queue or using `cl_mem` buffers.

The portable provider requires the `usm_device_allocations` aspect. PoCL does
not currently expose device USM to Open DPC++, so the PoCL CI jobs verify that
provider discovery reports a clear unavailable status and that plan creation
fails safely. The Intel CPU OpenCL CI job executes the complete FFT test suite.
AdaptiveCpp's OpenMP backend provides the portable CPU execution coverage on
Linux, Windows, and macOS.

The automatic selection order is:

- CUDA queue: cuFFT, then portable SYCL;
- OpenCL or CPU queue: portable SYCL.

An explicit provider request never falls back. Explicit FFTW selection is
available for SYCL CPU queues with host/shared USM and is synchronous. Inspect
runtime state with `syclfft::query_providers(queue)`.

## Host API

```cpp
#include <syclfft/host.hpp>

std::vector<std::complex<double>> input(256), output(256);
auto fft = syclfft::host::plan_dft_1d<double>(
    256, syclfft::direction::forward,
    {.placement = syclfft::placement::out_of_place});
fft.execute(input.data(), output.data());
```

The host library is isolated from SYCL: linking `syclfft::host` does not load or
link a SYCL runtime.

## Providers

The portable provider uses global-memory radix-2 and mixed-radix 2/3/5/7
stages. Other lengths use Bluestein convolution with plan-owned chirp spectra
and radix-2 convolution FFTs. Multidimensional transforms execute one axis at a
time using plan-owned USM scratch.

FFTW and cuFFT live in loadable provider modules. Modules are searched only in:

1. directories listed by `SYCLFFT_PLUGIN_PATH`; and
2. the installed `syclfft/providers` directory beside the core library.

The working directory is never searched. Missing modules and ABI/dependency
failures are reported through provider diagnostics. `rocfft` and `onemkl` are
reserved provider identifiers for future plugins.

VkFFT is not used in v0.1. AdaptiveCpp does not expose the native OpenCL queue
needed by VkFFT, and VkFFT's OpenCL interface uses `cl_mem` rather than a
zero-copy USM path.

## Conan 2

Use a profile with C++20 enabled:

```sh
conan install . -of build/conan -s build_type=Release \
  -s compiler.cppstd=20 --build=missing
cmake --preset conan-release
cmake --build --preset conan-release
ctest --test-dir build/conan/build/Release --output-on-failure
```

Conan options are:

- `with_host=True|False`;
- `with_sycl=True|False` (forced off on macOS);
- `sycl_implementation=adaptivecpp|dpcpp`;
- `use_system_fftw=True|False`;
- `with_cufft=auto|True|False`.

The default dependency is `fftw/3.3.10`, built static with single and double
precision only. CUDA is never fetched by Conan. `with_cufft=auto` builds the
plugin only when CMake finds an installed CUDA Toolkit; `True` requires it.

For AdaptiveCpp, make its CMake package discoverable, for example:

```sh
cmake -S . -B build -G Ninja \
  -DAdaptiveCpp_DIR=/opt/homebrew/opt/adaptivecpp/lib/cmake/AdaptiveCpp \
  -DSYCLFFT_BUILD_SYCL=ON -DSYCLFFT_SYCL_IMPLEMENTATION=AdaptiveCpp
```

For Open DPC++, configure with `icpx`/`dpcpp` and
`-DSYCLFFT_SYCL_IMPLEMENTATION=DPCPP`.

System FFTW mode requires both `fftw3` and `fftw3f`:

```sh
cmake -S . -B build -DSYCLFFT_USE_SYSTEM_FFTW=ON
```

Set `FFTW3_ROOT` when the installation is outside normal CMake or pkg-config
search paths.

Installed CMake targets and matching Conan components are `syclfft::host` and
`syclfft::sycl` (plus the internal shared `syclfft::core`).

## Continuous integration

The GitHub Actions workflow uses GitHub-hosted runners only. It covers macOS
FFTW; AdaptiveCpp OpenMP CPU execution on macOS, Linux, and Windows; Open DPC++
execution through PoCL and Intel CPU OpenCL runtimes on Linux; Open DPC++
execution through PoCL on Windows; install-tree consumers; and a compile-only
cuFFT plugin build in an NVIDIA CUDA development container. The Windows
AdaptiveCpp job uses the official LLVM 20 nightly binary from the project's
`develop` workflow. The Windows PoCL job uses the official 7.0 installer because
PoCL 7.1 does not publish a Windows installer. No GPU or self-hosted runner is
required.

## License

The core and portable provider are Apache-2.0. The vendored SyclCPLX header is
pinned to commit `32684c725eace3ee49146dbec2737b6bc2cc942d` and retains its
Apache-2.0 WITH LLVM-exception attribution.

The optional FFTW provider links FFTW, which is GPL-2.0-or-later. A distributed
binary containing that module may therefore carry GPL redistribution
obligations. See [NOTICE](NOTICE).
