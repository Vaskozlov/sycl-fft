# C2C correctness and performance on CPU and CUDA

Tested on 2026-11-23 with a Ryzen 9 5950X and a GeForce GTX 1660 SUPER
(6 GiB, SM 7.5) on the same WSL2 host. The starting syclfft revision was
`6a3a3d1`. Both devices used a Release build with Clang 20.1.8 and the
same AdaptiveCpp generic binary. AdaptiveCpp was built from the driver-only
branch at `66a65d0`. The optional comparison providers used FFTW 3.3.10 and
cuFFT 12.5.0.14 from CUDA Toolkit 13.5.37. The Windows driver was 610.88.
[AdaptiveCpp build configuration](https://github.com/Vaskozlov/AdaptiveCpp/blob/66a65d00bff200c68ccc0f541e978e1289538e80/doc/install-cuda-driver.md)

## The reported 4,096-point transform

Every frequency bin was compared with a double-precision FFTW reference.
The table gives relative L2 errors over whole batches. Each revision ran
128 configurations covering batch counts
1, 3, 32 and 256, FP32 and FP64, both directions, both placements and all four
normalizations. The input contained independent real and imaginary components
from a seeded generator, with distinct values across batches. All three
providers received the same input. Out-of-place input preservation was checked.

| Maximum relative L2 error across configurations | Original FP32 | Fixed FP32 | Original and fixed FP64 |
| --- | ---: | ---: | ---: |
| CPU portable vs FFTW reference | 2.77e-7 | 1.25e-7 | 5.21e-16 |
| CUDA portable vs FFTW reference | 2.73e-7 | 1.10e-7 | 5.18e-16 |
| CUDA cuFFT vs FFTW reference | 1.80e-7 | 1.80e-7 | 5.39e-16 |
| CPU portable vs CUDA cuFFT | 3.24e-7 | 2.03e-7 | 6.61e-16 |

These are relative errors over the spectrum. The largest **absolute**
FP32 difference in any bin between CPU portable and cuFFT was `1.10e-4` on
the original build and `5.19e-5` after the changes. Comparisons that require
identical bins or apply a small fixed absolute tolerance can flag these
differences. The reported batch count, exact inputs, error magnitude and
consumer code were unavailable, so the cause of that specific report remains
open. Batched lengths here follow the public contiguous layout, with the next
transform starting 4,096 complex elements after the previous one.

The regression suite also tests three different complex tones in adjacent
4,096-point batches and validates every bin against its analytical value.
It uses a second queue in the same context to consume the returned event,
including after plan destruction.

## Defects reproduced

**Loss of precision at prime lengths.** The portable Bluestein plan evaluated
an angle proportional to the squared index in the transform's precision.
At 65,537 points, the original FP32 angle lost enough precision to spread
energy into the wrong bins. Planning now computes the squared index modulo
`2N` using integer arithmetic, generates angles in double precision on the
host and reuses the convolution twiddles. The 4,096-point transform uses a
power-of-two algorithm and does not evaluate a Bluestein chirp.

| Dense FP32 forward C2C, relative L2 vs FP64 FFTW | Original CPU | Fixed CPU | Original CUDA portable | Fixed CUDA portable | cuFFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,093 | 3.98e-4 | 2.70e-7 | 3.98e-4 | 2.60e-7 | 3.38e-7 |
| 65,536 | 3.66e-7 | 1.49e-7 | 3.63e-7 | 1.40e-7 | 2.42e-7 |
| 65,537 | 7.19e-3 | 2.36e-7 | 7.19e-3 | 2.29e-7 | 3.57e-7 |

**cuFFT module lifetime.** With the original module-lifetime behavior a CUDA test process could
segfault after the transform completed. GDB showed AdaptiveCpp destroying a
kernel operation on its DAG worker and calling into unmapped code. Keeping
the provider mapped with `LD_PRELOAD` let the original code finish. That
workaround was used for the original revision's numerical comparison above.
The fixed library retains each selected cuFFT module for the process lifetime,
so a returned event or deferred operation can outlive the plan. Repeated CUDA
suite runs then passed with ordinary provider loading.

**Oversized plans.** Some point counts fit in `size_t` while their allocation
sizes in bytes overflow. Planning now rejects these cases before allocating
portable scratch or initializing transform tables. Plan-only tests
exercise the boundaries.

**Overlapping batches.** An out-of-place transform requires disjoint input
and output ranges. The common execution path now rejects partial overlap
before a provider can overwrite input belonging to a later batch member.
Tests cover both pointer orders and adjacent disjoint views of one allocation.

## Performance

Median warmed latency in **microseconds per whole batch**, FP32 forward,
out-of-place, without normalization. Measurements include host submission
and completion on resident data. The CPU used 16 OpenMP threads. The FFTW
plugin uses serial FFTW. Each plan ran once, warmed up five more times, then
ran 21 measured trials. The CSV includes p10/p90, plan creation, first use
and a separate eight-submit burst measurement. First-use times depend on
the state of compiler and driver caches. In the CSV, `staged` is the portable
implementation after the numerical fixes and before the batched kernel changes.
`final` is the optimized build, `plugin` uses the library's FFTW or cuFFT
provider, and `direct` calls the FFT library on a CUDA stream. The scratch
column records the two global scratch buffers for staged native plans, the
reported workspace for direct cuFFT, or the temporary buffer for direct
VkFFT. Table storage and vendor-plugin internals are excluded.
[Timing data](wsl-gtx1660-super.csv)

| 1D length or shape | Batch | CPU native before → after | CPU FFTW | CUDA native before → after | CUDA cuFFT | Direct VkFFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 32 | 1,651.5 → 322.6 | 261.5 | 621.4 → 65.9 | 35.5 | 37.6 |
| 4,096 | 256 | 4,795.6 → 1,710.8 | 1,372.2 | 1,423.6 → 314.2 | 87.2 | 90.2 |
| 2,048 | 32 | 391.3 → 213.1 | 91.0 | 535.3 → 50.5 | 32.2 | 32.7 |
| 2,048 | 256 | 2,276.3 → 597.9 | 469.4 | 882.2 → 187.1 | 55.5 | 53.2 |
| 1,024 | 256 | 977.3 → 294.8 | 279.4 | 629.0 → 75.4 | 46.5 | 41.2 |
| 4,096 | 1 | 137.3 → 77.6 | 54.2 | 541.0 → 46.4 | 28.7 | 28.3 |
| 1,024 | 1 | 144.2 → 61.1 | 55.3 | 453.0 → 31.2 | 28.8 | 25.6 |
| 1,000 | 256 | 2,523.4 → 2,344.5 | 484.4 | 1,382.0 → 1,321.5 | 45.5 | 41.7 |
| 4,093 | 32 | 2,616.8 → 2,647.2 | 1,834.8 | 1,623.1 → 1,609.2 | 94.9 | 80.7 |
| 65,536 | 1 | 1,612.0 → 1,534.2 | 313.0 | 723.0 → 724.0 | 42.4 | 45.3 |
| 256 × 256 | 1 | 686.2 → 619.5 | 1,288.8 | 873.4 → 857.2 | 39.8 | 33.2 |

For repeated resident batches, the median amortized time below includes eight
consecutive submissions and one completion wait, divided by eight. The native
and cuFFT plugin measurements include SYCL execution and dependency tracking.

| Length × batch | Native CPU µs | Native CUDA µs | cuFFT plugin µs | Direct cuFFT µs | Direct VkFFT µs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4,096 × 1 | 38.4 | 50.5 | 37.9 | 10.3 | 10.7 |
| 4,096 × 32 | 297.5 | 71.9 | 44.1 | 16.4 | 18.4 |
| 4,096 × 256 | 999.6 | 313.2 | 96.6 | 68.0 | 71.9 |

At 4,096 × 32, native CUDA reached 65.9 µs from 621.4 µs (9.4× faster).
At batch 256 it reached 314.2 µs from 1,423.6 µs (4.5× faster). That latter
shape also measured 250–251 µs during prototype comparisons. These
measurements characterize this machine and launch configuration. At batch
256, the final native GPU burst sustained about 0.82 million 4,096-point
transforms per second, compared with 3.56 million for direct VkFFT.

The native fast paths cover contiguous 1D lengths 1,024, 2,048 and 4,096.
On the CPU, one work-item computes a whole transform in its output buffer.
On this AdaptiveCpp runtime, using smaller work-groups helps distribute small
batches across CPU threads. On CUDA, 256 lanes process one transform in local
memory. Pairs of radix-2 stages form register-based radix-4 butterflies.
Length 2,048 has one final radix-2 stage. The kernel reads its entire input
before writing results, folds normalization into the final store and uses
no global scratch buffers. Plans retain their twiddle tables. Shapes or devices
outside these fast paths use the staged implementation.

Before optimization, Nsight Systems counted 168 native CUDA kernels for 12
executions of 4,096 × 32. Each execution used one permutation, 12 butterfly
stages and one copy. The optimized native path and cuFFT each launched one
kernel per execution. Summed GPU kernel time per batch averaged 126 µs for
the staged implementation, 37.8 µs for the optimized native kernel and 12.1 µs
for cuFFT. Kernel time excludes host submission and gaps between kernels.

The 1,000-point mixed-radix and 4,093-point Bluestein batches still use
global-memory stages. At 1,000 × 256, native CUDA took 1,322 µs against
41.7 µs for direct VkFFT, a 32× difference. This makes a local-memory
mixed-radix kernel a candidate for the next batched optimization.

VkFFT v1.3.4 and direct cuFFT used the same seeded dense input, device
allocations, warmup schedule, serial waits, eight-submit bursts and sampled
double-precision DFT checks as the SYCL harness. Direct calls synchronized the
CUDA stream and bypassed the SYCL submission layer. A separate analytical
two-impulse probe checked all VkFFT output bins and input preservation.
Hardware clock rates and CPU thread affinity were uncontrolled. Each fresh
benchmark process could reuse compiler or driver caches.

## Toolkit-free VkFFT provider

Upstream VkFFT v1.3.4 uses CUDA Runtime calls for allocation, copies and
events. It also emits CUDA C++ when a plan is created and calls NVRTC to
compile it. The NVIDIA driver accepts PTX as input. Replacing the runtime
calls alone left a dependency on NVRTC in the original standalone probe.
[VkFFT compiler source](https://github.com/DTolm/VkFFT/blob/v1.3.4/vkFFT/vkFFT/vkFFT_PlanManagement/vkFFT_API_handles/vkFFT_CompileKernel.h#L300-L359)
and [CUDA Driver module formats](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html).

The optional provider patches the pinned VkFFT headers to call the driver
API and to request PTX through a callback. A private helper compiles fresh
plans with the same distribution Clang/LLVM release used by AdaptiveCpp.
Using a separate process lets applications load their own Clang frontend
without conflicting LLVM command-line option registrations. The installed
provider and helper dynamically load or link their dependencies. The CUDA
driver's own PTX JIT is supplied by the NVIDIA driver.

The driver-only AdaptiveCpp branch and syclfft were rebuilt on WSL with both
the NVIDIA Toolkit and the driver library directories hidden. The build used
Clang 20.1.8 and the driver's declarations from the AdaptiveCpp source tree.
On the deployment side, the Toolkit, Clang executables and Clang resource
headers were hidden. The installed helper links the Ubuntu `libclang-cpp20`
and matching LLVM runtime libraries. The stripped helper is 35,432 bytes and
the provider is 704,304 bytes on this x86-64 build.

Batched forward FP32 timings below use the same SYCL public API, resident
device buffers, five warmups and 21 timed samples for each provider. Each
latency is the median host submission-through-completion time for a whole
batch. The last column records construction of the VkFFT plan in that process.
The `driver_only` rows include the burst measurements and the native planning
cost. GPU and driver cache state can affect planning time.
[Raw timings](wsl-gtx1660-super.csv)

| Length | Batch | Native SYCL (µs) | VkFFT via SYCL (µs) | VkFFT plan (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 4,096 | 1 | 57.4 | 34.3 | 183.6 |
| 4,096 | 32 | 69.2 | 45.0 | 73.2 |
| 4,096 | 256 | 314.0 | 105.0 | 72.9 |
| 4,093 | 32 | 1,574.9 | 96.7 | 309.4 |
| 1,000 | 256 | 1,334.7 | 49.8 | 154.7 |
| 2,048 | 256 | 194.1 | 61.6 | 172.8 |
| 1,024 | 256 | 76.9 | 44.3 | 224.6 |

Full-spectrum comparisons against double-precision FFTW used both directions,
placements and precisions, four normalizations and batch counts 1, 3, 32
and 256. All 256 configurations passed, including out-of-place input
preservation. A separate public-API test consumed the returned event on
another queue. The installed provider also compiled a new 1,009 × 7
configuration with the system Clang executables hidden.

| Length | Largest relative L2 error, VkFFT FP32 | Largest relative L2 error, VkFFT FP64 |
| ---: | ---: | ---: |
| 4,096 | 1.34e-7 | 3.26e-16 |
| 4,093 | 2.80e-7 | 6.58e-16 |

The toolkit-free build, test script, build and loader logs, per-configuration
accuracy CSVs and raw latencies are in
`~/syclfft-review/toolkit-free.eEC8sB` on the WSL host. The optional build
uses the [AdaptiveCpp driver-only branch](https://github.com/Vaskozlov/AdaptiveCpp/blob/66a65d00bff200c68ccc0f541e978e1289538e80/doc/install-cuda-driver.md).

## Conan build verification

The `driver_only` timing rows were collected before the plan and dependency
refactor. On November 24, the Conan build repeated the full 256-case
comparison and preserved the reported worst-case relative errors. With the
Toolkit hidden, automatic selection used VkFFT. At 4,096 × 32 and 4,096 × 256,
its median latencies were 40.7 and 102.9 µs. Explicit portable SYCL took
66.4 and 312.3 µs. The `conan_review` rows record these four measurements.
The dependency graph used `vkfft-driver/1.3.4`, `clang-cpp-system/20.1.8`
and `fftw-system/3.3.10`, plus the local driver-only AdaptiveCpp installation.

Build logs and per-configuration results are in
`~/syclfft-review/conan.1jvUfI` on the WSL machine.

## AdaptiveCpp shared USM on WSL

During the Conan consumer test, the driver-only AdaptiveCpp revision
`66a65d0` reported CUDA error 1 while freeing shared USM. This standalone
reproducer produced the same error on the GTX 1660 SUPER:

```cpp
#include <sycl/sycl.hpp>
int main() {
    sycl::queue q{sycl::gpu_selector_v};
    float *p = sycl::malloc_shared<float>(8, q);
    p[0] = 1.0f;
    sycl::free(p, q);
    q.wait_and_throw();
}
```

A driver probe returned `CU_MEMORYTYPE_HOST` and `IS_MANAGED=1` for a
`cuMemAllocManaged` allocation. `cuMemFree_v2` released it successfully.
The original fork selected `cuMemFreeHost` from the memory type alone.
AdaptiveCpp commit `a6a8919d` gives the managed flag precedence. The new
allocator regression failed before the fix, then all four driver tests and
the standalone WSL reproducer passed. The installed FFT consumer exercises
shared USM and waits for asynchronous errors after freeing its buffers.
[Original AdaptiveCpp CUDA allocator](https://github.com/Vaskozlov/AdaptiveCpp/blob/66a65d00bff200c68ccc0f541e978e1289538e80/src/runtime/cuda/cuda_allocator.cpp#L67-L83)

## Reproduction

The source and build directories, standalone probes, GDB log, per-configuration
accuracy CSVs and Nsight traces are in `~/syclfft-review` on the WSL host.
The benchmark can also be built in a fresh checkout. [Benchmark instructions](README.md)

```sh
cd ~/syclfft-review
export LD_LIBRARY_PATH="$PWD/fftw/usr/lib/x86_64-linux-gnu:/usr/lib/wsl/lib:/usr/local/cuda/lib64"
export SYCLFFT_PLUGIN_PATH="$PWD/fixed/syclfft/providers" OMP_NUM_THREADS=16
ACPP_VISIBILITY_MASK=omp ctest --test-dir fixed --output-on-failure
ACPP_VISIBILITY_MASK=cuda fixed/tests/syclfft_sycl_tests --require-cufft
# Both CPU and CUDA are needed for the pairwise comparison.
env -u ACPP_VISIBILITY_MASK ./compare4096
env -u ACPP_VISIBILITY_MASK SYCLFFT_PLUGIN_PATH="$PWD/original/syclfft/providers" \
  LD_LIBRARY_PATH="$PWD/original:$LD_LIBRARY_PATH" \
  LD_PRELOAD="$PWD/original/syclfft/providers/syclfft_provider_cufft.so" ./compare4096
bash run-matrix.sh
bash run-4096-bench.sh
bash compare-perf.sh portable_sycl cuda staged
bash compare-perf.sh portable_sycl cuda fused
ACPP_VISIBILITY_MASK=cuda bench/syclfft_benchmark cuda portable_sycl float 2048 32 21 8
cd vkfft-reference
bash run-dense.sh
./cuda-dense-reference vkfft_stock 2048 32 21 8
```

The VkFFT probe, its no-runtime link log and raw timings are in
`~/syclfft-review/vkfft-reference`. The archive SHA-256 is
`b61055393adb3adc79009fe12401cbfbbdfba584e665e9c35fcbf4b32fb31f30`.
The historical stock comparison below requires the NVIDIA Toolkit. Use the
[driver-only build instructions](../README.md#toolkit-free-cuda-with-vkfft)
to build the optional provider. To repeat the historical comparison, run the
following commands in the reference directory. Removing `-lcudart` from the link reproduces the failed
driver-only build.

```sh
g++ -std=c++20 -O3 -DNDEBUG -isystem VkFFT-1.3.4/vkFFT \
  -isystem /usr/local/cuda/include -c syclfft_vkfft_reference.cpp -o reference.o
g++ reference.o -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 \
  -Wl,-rpath,/usr/lib/wsl/lib:/usr/local/cuda/lib64 -lcuda -lnvrtc -lcudart -o vkfft-reference
./vkfft-reference
```
