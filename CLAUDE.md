# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`syclfft` is a C++17 complex-to-complex FFT library with FFTW-style reusable RAII plans. It provides:

- **SYCL API** (`syclfft::plan`) — async execution returning `sycl::event`, USM-based, works with AdaptiveCpp and Open DPC++
- **Host API** (`syclfft::host::plan`) — synchronous, uses `std::complex`, isolated from SYCL (no SYCL runtime needed)

The provider architecture is plugin-based: the portable SYCL provider is built-in, while FFTW and cuFFT live in loadable modules discovered via `SYCLFFT_PLUGIN_PATH`.

## Build System

CMake 3.22+, C++17, **shared libraries only** (static builds are a fatal error).

### Quick Build (Host/FFTW only — works on macOS)

```sh
# macOS (system FFTW via brew)
brew install cmake ninja fftw
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSYCLFFT_BUILD_SYCL=OFF -DSYCLFFT_USE_SYSTEM_FFTW=ON -DSYCLFFT_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

### Build with AdaptiveCpp SYCL

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DAdaptiveCpp_DIR=<path>/lib/cmake/AdaptiveCpp \
  -DSYCLFFT_BUILD_SYCL=ON -DSYCLFFT_SYCL_IMPLEMENTATION=AdaptiveCpp \
  -DSYCLFFT_BUILD_TESTS=ON
cmake --build build
```

### Build with Open DPC++

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=<dpcpp>/bin/clang -DCMAKE_CXX_COMPILER=<dpcpp>/bin/clang++ \
  -DSYCLFFT_BUILD_SYCL=ON -DSYCLFFT_SYCL_IMPLEMENTATION=DPCPP \
  -DSYCLFFT_BUILD_TESTS=ON
cmake --build build
```

### Key CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SYCLFFT_BUILD_HOST` | ON | Host API + FFTW provider module |
| `SYCLFFT_BUILD_SYCL` | ON (OFF on macOS) | SYCL API |
| `SYCLFFT_SYCL_IMPLEMENTATION` | AdaptiveCpp | `AdaptiveCpp` or `DPCPP` |
| `SYCLFFT_CUFFT` | AUTO | Build cuFFT provider: `AUTO`/`ON`/`OFF` |
| `SYCLFFT_USE_SYSTEM_FFTW` | OFF | Use system FFTW instead of Conan |
| `SYCLFFT_BUILD_TESTS` | OFF | Enable tests |
| `SYCLFFT_BUILD_EXAMPLES` | ON | Build examples |

### CMake Targets

| Target | Description |
|--------|-------------|
| `syclfft::core` | Provider loader (dlopen/LoadLibrary, ABI checking) |
| `syclfft::host` | Host API (links core, delegates to FFTW provider) |
| `syclfft::sycl` | SYCL API (links core, includes portable SYCL provider) |
| `syclfft_provider_fftw` | MODULE lib — FFTW provider plugin |
| `syclfft_provider_cufft` | MODULE lib — cuFFT provider plugin (optional) |

### Conan 2

```sh
conan install . -of build/conan -s build_type=Release -s compiler.cppstd=17 --build=missing
cmake --preset conan-release
cmake --build --preset conan-release
```

## Testing

**No external test framework** — custom `check(bool, string)` macro throws `std::runtime_error`; `main()` wrapped in try/catch.

Enable with `-DSYCLFFT_BUILD_TESTS=ON`. Three test executables:

| Test | CTest name | What it covers |
|------|------------|----------------|
| `syclfft_host_tests` | `syclfft.host` | Host API via FFTW: 1D/2D/3D, normalizations, in-place/out-of-place, round-trip accuracy, error handling |
| `syclfft_sycl_tests` | `syclfft.sycl` | SYCL API + portable provider: radix-2, mixed-radix, Bluestein, multidimensional, batching, events, concurrent plans, provider querying |
| `syclfft_plugin_tests` | `syclfft.plugin-abi` | Provider ABI version rejection |

All tests set `SYCLFFT_PLUGIN_PATH` to the build providers directory automatically via CTest properties.

### Running Tests

```sh
# All tests
ctest --test-dir build --output-on-failure

# Single test
ctest --test-dir build --output-on-failure --test-case syclfft.host

# Run SYCL tests directly (e.g., for debugging)
./build/tests/syclfft_sycl_tests

# PoCL graceful-rejection mode (when portable provider can't use USM)
./build/tests/syclfft_sycl_tests --expect-portable-unavailable
```

## Architecture

### Provider Model

The core library (`syclfft_core`) contains a provider loader that uses `dlopen`/`LoadLibrary` to discover provider modules. Search order:
1. Directories listed in `SYCLFFT_PLUGIN_PATH` environment variable
2. Installed `syclfft/providers` directory beside the core library

The working directory is **never** searched. Missing modules and ABI failures are reported through `query_providers()`.

Provider ABI is defined in `include/syclfft/detail/provider_abi.h` (C ABI for host/FFTW) and `include/syclfft/detail/cufft_provider_abi.hpp` (C++ ABI for cuFFT).

### Portable SYCL Provider (Built-in)

Located inline in `src/sycl/sycl.cpp` (~700 lines). Pure-SYCL FFT with:
- **Radix-2**: bit-reversal + Cooley-Tukey butterfly stages
- **Mixed-radix 2/3/5/7**: multi-stage decomposition
- **Bluestein convolution**: arbitrary sizes via chirp-z → radix-2 convolution
- **Multi-dimensional**: axis-by-axis, alternating between two scratch buffers

Twiddle tables precomputed at plan creation, uploaded to USM. Requires `usm_device_allocations` aspect.

### Provider Selection

With `provider::automatic`:
- CUDA queue → cuFFT, then portable SYCL
- OpenCL/CPU queue → portable SYCL

Explicit provider requests never fall back. FFTW on SYCL CPU queues is synchronous (waits, runs on host, returns event).

### Public API

**`include/syclfft/syclfft.hpp`** — SYCL API: `plan_dft_1d/2d/3d`, `plan_many_dft`, `query_providers(queue)`, `plan<Scalar>` with `execute()` overloads accepting optional `sycl::event` or `syclfft::span<const sycl::event>` dependencies.

**`include/syclfft/host.hpp`** — Host API (in `syclfft::host` namespace): same `plan<Scalar>` interface but synchronous, no events, uses `std::complex`.

**`include/syclfft/common.hpp`** — Shared enums (`direction`, `placement`, `normalization`, `planning_mode`, `provider`), `plan_options`, `provider_status`, `error_code`, `exception`.

### Third-Party

SyclCPLX (`third_party/syclcplx/`) — vendored SYCL complex number header, pinned to commit `32684c7`, Apache-2.0 WITH LLVM-exception. Copied to build tree as `syclfft/detail/sycl_ext_complex.hpp`.

## CI Coverage

All GitHub-hosted runners, no GPU or self-hosted required:
- **macOS**: Host FFTW + install consumer; AdaptiveCpp OpenMP CPU
- **Linux**: AdaptiveCpp OpenMP CPU + install consumer; DPC++ with PoCL (graceful rejection) and Intel CPU OpenCL (full execution)
- **Windows**: Host FFTW via Conan + install consumer; DPC++ compile/link only; AdaptiveCpp OpenMP CPU
- **CUDA**: Compile-only cuFFT plugin in NVIDIA container (DPC++)

## Code Style

`.clang-format`: LLVM-based, 100-col limit, 4-space indent, `AlwaysBreakTemplateDeclarations: Yes`, `SortIncludes: CaseSensitive`, custom brace wrapping (AfterClass/Enum/Function/Namespace/Struct all true).
