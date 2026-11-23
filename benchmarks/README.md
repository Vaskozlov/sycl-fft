# C2C benchmarks

Build the benchmark with Conan. Export the optional provider recipes once as
described in the [build instructions](../README.md#conan-2). For the
driver-only AdaptiveCpp CUDA configuration:

```sh
conan build . --build=missing -s build_type=Release -s compiler.cppstd=17 \
  -o 'syclfft/*:build_benchmarks=True' -o 'syclfft/*:with_vkfft=True' \
  -o 'syclfft/*:adaptivecpp_root=/path/to/acpp' -o 'syclfft/*:acpp_targets=generic' \
  -o 'syclfft/*:cuda_driver_include_dir=/path/to/AdaptiveCpp/third-party/nvidia/include'
export SYCLFFT_PLUGIN_PATH="$PWD/build/Release/syclfft/providers"
build/Release/benchmarks/syclfft_benchmark cuda automatic float 4096 256 21 8
build/Release/benchmarks/syclfft_benchmark cuda portable_sycl float 4096 256 21 8
build/Release/benchmarks/syclfft_benchmark cuda vkfft float 4096 256 21 8
```

These paths use a single-configuration Release build. Arguments are device,
provider, precision, shape, batch count, sample count, and burst size. The
provider column records the choice made by the plan, including in automatic
mode. Use `x` between dimensions, as in `256x256`. Batch count, sample count,
and burst size default to 1, 15, and 8.

| Shape | Batch | Purpose |
| --- | ---: | --- |
| 1024 | 1 | Small radix-2 latency |
| 1024 | 256 | Batched radix-2 throughput |
| 4096 | 1, 32, 256 | Reported 1D workload and batch scaling |
| 1000 | 256 | Batched mixed-radix throughput |
| 4093 | 32 | Batched prime-length Bluestein |
| 256x256 | 1 | Multidimensional with contiguous and strided axes |
| 65536 | 1 | Large 1D radix-2 |

Each invocation writes one CSV row plus a header to stdout and the device
information to stderr. It reuses an out-of-place, unnormalized forward plan
and deterministic dense complex input. Input and output use device USM on
CUDA and shared USM on the CPU. Allocation and upload happen before timing.
The planning column is an estimate-mode construction measurement for one
plan. The first-execution column can include SYCL JIT and launch startup.
Five additional warm-ups precede the measured samples.

Latency samples time one submission and its completion wait. Burst samples
time multiple submissions and the last completion wait, then divide by the
burst size. Both are host wall-clock measurements in microseconds per plan
execution, including dispatch overhead. To get transforms per second from a
burst measurement, multiply batch count by 1,000,000 and divide by the
per-execution time in microseconds.

After timing, the benchmark checks three frequency bins in the first and
last transforms against a direct DFT, then reports the largest absolute
error divided by the square root of the transform size. Exit code 2 marks a
row whose output exceeded the error limit of 0.0005 for float or 1e-10 for
double. The scratch column reports the library's public
`scratch_size_bytes()` value, which covers the portable provider's scratch
buffers. Provider-owned workspace and precomputed tables are omitted.
