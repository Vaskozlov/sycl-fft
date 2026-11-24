# syclfft

`syclfft` is a C++17 C2C FFT library with FFTW-style reusable RAII plans and a
unified USM interface for SYCL. Plans are direction-specific and
pointer-independent; execution returns native `sycl::event` objects.

Version 0.1 supports:

- `float` and `double` C2C transforms;
- contiguous 1D, 2D, and 3D transforms and contiguous batches;
- in-place and out-of-place execution;
- no normalization, forward/backward normalization, and orthogonal normalization;
- AdaptiveCpp on Linux, Windows, and macOS;
- Open DPC++ on Linux and Windows;
- a portable pure-SYCL provider for CPU, CUDA, and OpenCL queues;
- an optional runtime-loaded FFTW provider for SYCL CPU queues;
- optional runtime-loaded cuFFT and VkFFT providers for CUDA queues.

R2C/C2R, arbitrary strides, split-complex storage, wisdom, and graph-capture
guarantees are outside the v0.1 scope. Dimensions use C order, so the last
dimension is contiguous.

## SYCL API

```cpp
#include <syclfft/syclfft.hpp>

sycl::queue queue;
auto* input = sycl::malloc_shared<syclfft::complex<float>>(1024, queue);
auto* output = sycl::malloc_shared<syclfft::complex<float>>(1024, queue);

syclfft::plan_options options;
options.placement = syclfft::placement::out_of_place;
options.normalization = syclfft::normalization::none;
options.preferred_provider = syclfft::provider::automatic;

auto fft = syclfft::plan_dft_1d<float>(
    queue, 1024, syclfft::direction::forward, options);

sycl::event done = fft.execute(input, output);
done.wait_and_throw();
```

`execute` also accepts one dependency event or a C++17-compatible
`syclfft::span<const sycl::event>` of events. Arrays and contiguous containers
convert to `syclfft::span` without copying. A plan serializes executions which
share its internal scratch. Create multiple plans when transforms must run
concurrently.

All device pointers must be USM allocations belonging to the plan queue's
context. Out-of-place input and output must cover disjoint ranges of complex
elements. Adjacent ranges in one allocation are supported. The portable
provider never copies transform data to the host. This is
also the OpenCL path in AdaptiveCpp: kernels operate directly on USM pointers,
without exposing an OpenCL queue or using `cl_mem` buffers.

The portable provider requires the `usm_device_allocations` aspect. PoCL does
not consistently expose usable device USM to Open DPC++, so the Linux PoCL CI
job verifies that provider discovery reports a clear unavailable status and
that plan creation fails safely. The Intel CPU OpenCL CI job executes the
complete FFT test suite.

AdaptiveCpp's OpenMP backend provides the portable CPU execution coverage on
Linux, Windows, and macOS.

A plan defaults to `provider::automatic`. On a CUDA queue it tries cuFFT,
then VkFFT, then portable SYCL. With driver-only AdaptiveCpp and VkFFT
enabled, an ordinary CUDA plan therefore selects **VkFFT**. If its module
or compiler helper is unavailable, or VkFFT rejects the plan, automatic
selection proceeds to portable SYCL. CPU and OpenCL queues select portable
SYCL. Explicit provider requests report a failure if that provider cannot
serve the plan. Select FFTW explicitly for synchronous CPU execution with
host/shared USM.

Use `plan.selected_provider()` to inspect the choice for a particular plan.
`syclfft::query_providers(queue)` reports discovery and queue compatibility.
Shape support and runtime compiler dependencies are checked during planning.

`syclfft` intentionally does not wrap a standalone host-only FFT API. If an
application does not need SYCL queues, events, or USM, use FFTW directly.

## Providers

The portable provider has dedicated 1D radix-2 kernels for 1,024, 2,048 and
4,096 points. On CPUs each work-item transforms a contiguous batch member.
On CUDA GPUs with sufficient work-group resources one group transforms each batch
member in local memory. These kernels write directly to the destination and
need no global scratch buffers. Other radix-2 transforms and mixed-radix
2/3/5/7 transforms use global-memory stages. Other lengths use Bluestein
convolution with plan-owned chirp spectra. Multidimensional transforms execute
one axis at a time using plan-owned USM scratch.

FFTW, cuFFT, and VkFFT live in loadable provider modules. Modules are searched only in:

1. directories listed by `SYCLFFT_PLUGIN_PATH`; and
2. the installed `syclfft/providers` directory beside the core library.

The working directory is never searched. Missing modules and ABI/dependency
failures are reported through provider diagnostics. `rocfft` and `onemkl` are
reserved provider identifiers for future plugins.

### Toolkit-free CUDA with VkFFT

On Linux with driver-only AdaptiveCpp, the optional VkFFT provider creates
C2C plans for dimensions and batch counts supplied at runtime. During plan
creation, a small helper beside the plugin translates VkFFT's generated CUDA
source to PTX through the distribution's shared Clang frontend. The NVIDIA
driver loads the PTX and executes it on the SYCL queue's CUDA stream. Reuse
plans to amortize compilation time.

The Linux provider supports LLVM/Clang 15 through 21. AdaptiveCpp's Conan
package supplies its LLVM release and the CUDA driver headers. Its system
recipe references the matching Clang and LLVM development packages. Deployment
requires their matching shared runtimes and the NVIDIA driver. If the selected
LLVM cannot emit code for the GPU architecture, automatic selection falls
back to portable SYCL. For shared USM on WSL, the driver-only AdaptiveCpp
fork includes an allocator fix in commit `a6a8919d`.

For a manual CMake build with an older AdaptiveCpp SDK that omits the LLVM
release, set `SYCLFFT_EXPECTED_LLVM_VERSION` to the release used to build
that SDK. If it also omits the driver header path, set
`SYCLFFT_CUDA_DRIVER_INCLUDE_DIR` to the directory containing its `cuda.h`.

For VkFFT transforms with more than one point, input and output addresses
require 8-byte alignment for FP32 and 16-byte alignment for FP64. Normal
USM allocations and whole-complex offsets from them satisfy this requirement.

The [C2C benchmark](benchmarks/README.md) measures reusable-plan latency and
throughput. The [CPU and CUDA review](benchmarks/wsl-review.md) covers batched
4,096-point accuracy, prime-length transforms and provider timings.

## Conan 2

syclfft resolves the AdaptiveCpp SDK through Conan. The Linux and macOS
package is named `adaptivecpp`. Windows uses `adaptivecpp-llvm`. The SDK
supplies its compiler, runtime, and CMake integration. Its resolved version,
recipe revision, and binary configuration participate in syclfft's package ID.

### Local AdaptiveCpp and VkFFT

From the sycl-fft checkout, export the recipes to your local cache. This
Linux example uses the sibling AdaptiveCpp checkout on
`feature/cuda-driver-only-packaging` and distro LLVM 20. Use the
`llvm-config` for your installed LLVM release (15–21). On a fresh Conan
installation, run `conan profile detect` once.

```sh
conan export ../AdaptiveCpp/recipes/clang-cpp-system --version="$(llvm-config-20 --version)"
conan export ../AdaptiveCpp
conan export recipes/vkfft

conan create . --no-remote --build=missing -s compiler.cppstd=17 \
  -o 'syclfft/*:with_vkfft=True' \
  -o 'syclfft/*:with_fftw=False' \
  -o 'syclfft/*:build_tests=True'
```

With `--no-remote`, Conan selects the newest locally exported AdaptiveCpp
release. AdaptiveCpp resolves the newest supported `clang-cpp-system`
release in that cache and checks it against the installed LLVM. To repeat
the same resolution on a later build, add `--lockfile-out=conan.lock` to
the first create command and `--lockfile=conan.lock` to the next. The host
compiler comes from your ordinary Conan profile.

The VkFFT recipe downloads its upstream archive on first use and verifies
its SHA-256. The example disables FFTW so all Conan dependencies resolve
locally. For CPU FFTW, use the system recipe below or resolve the default
`fftw/3.3.10` from ConanCenter. Only the separate cuFFT provider requires
an NVIDIA Toolkit.

For development, use `conan build . --build=missing` with the same profile
and options. A downstream Conan recipe can require `syclfft/0.1.0`. Its
CMake project can then use `find_package(syclfft CONFIG REQUIRED)` and
link `syclfft::sycl`. AdaptiveCpp consumers apply `add_sycl_to_target` to
their own SYCL sources. The [consumer test](test_package/CMakeLists.txt)
shows the complete target setup. Core-only builds expose `syclfft::core`.

On macOS, export the AdaptiveCpp recipe and set
`adaptivecpp/*:llvm_version` to your installed upstream LLVM release in
the host profile. Set `adaptivecpp/*:with_cuda=False` for a CPU build.
On Windows, DPC++ is the default. To select AdaptiveCpp, export the fork's
`recipes/adaptivecpp-llvm` recipe and set
`syclfft/*:sycl_implementation=adaptivecpp` in the host profile.

### Open DPC++

Open DPC++ supplies its own SYCL-enabled Clang, headers, and `libsycl`.
Select that SDK through a Conan host profile. On Linux x86-64, the supplied
profile targets the open-source 6.3.0 release. Set `DPCPP_ROOT` to its
extracted directory and run from the repository root:

```sh
export DPCPP_ROOT=/path/to/dpcpp
conan create . --no-remote --build=missing -pr:a ./profiles/linux-dpcpp \
  -o 'syclfft/*:with_fftw=False'
```

Use the same host profile for applications that link this binary. Execution
also requires a compatible SYCL device runtime. The portable FFT uses USM.

### Options

| Conan option | What it controls |
| --- | --- |
| `build_sycl`, `build_tests`, `build_examples`, `build_benchmarks` | Build the public SYCL library, tests, examples, and benchmark. Defaults are `True`, `False`, `True`, and `False`. |
| `sycl_implementation` | `adaptivecpp` or `dpcpp`. Select the host compiler through the normal Conan profile. |
| `acpp_targets` | Target used to compile syclfft's own kernels, such as `generic` (default) or `omp`. |
| `with_vkfft` | Enable VkFFT for Linux with a CUDA-enabled AdaptiveCpp package. LLVM follows the AdaptiveCpp dependency. |
| `with_fftw`, `fftw_version`, `use_system_fftw`, `fftw_root` | FFTW provider, exact release, system-library selection, and optional installation prefix. Both float and double libraries are required. |
| `with_cufft`, `cuda_toolkit_version`, `cuda_toolkit_root` | cuFFT plugin and its exact toolkit version and optional prefix. `auto` enables cuFFT when a toolkit version is specified and VkFFT is disabled. `True` explicitly requests cuFFT. |

For system FFTW, export `recipes/fftw-system` with the installed version,
then set `use_system_fftw=True` and `fftw_version` to that version. For
the cuFFT plugin, export `recipes/cuda-toolkit-system` with the exact installed
toolkit version and set `cuda_toolkit_version` accordingly. For example:

```sh
conan export recipes/fftw-system --version=3.3.10
conan export recipes/cuda-toolkit-system --version=13.5.37
```

## Continuous integration

The GitHub Actions workflow uses GitHub-hosted runners only. It covers
AdaptiveCpp OpenMP CPU execution on macOS, Linux, and Windows; the optional FFTW
provider through the macOS AdaptiveCpp job; Open DPC++ capability handling with
PoCL and execution through Intel CPU OpenCL on Linux; Open DPC++ compilation and
linking on Windows; install-tree SYCL consumers; and a compile-only cuFFT plugin
build in an NVIDIA CUDA development container. The Windows AdaptiveCpp job uses
the official LLVM 20 nightly binary from the project's `develop` workflow. Open
DPC++ does not provide an OpenMP backend, and its native CPU target is currently
disabled on Windows, so DPC++ execution is not claimed by that job. No GPU or
self-hosted runner is required.

## License

The core and portable provider are Apache-2.0. The vendored SyclCPLX header is
pinned to commit `32684c725eace3ee49146dbec2737b6bc2cc942d` and retains its
Apache-2.0 WITH LLVM-exception attribution.

The optional FFTW provider links FFTW, which is GPL-2.0-or-later. A distributed
binary containing that module may therefore carry GPL redistribution
obligations. See [NOTICE](NOTICE).
