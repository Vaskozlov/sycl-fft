#pragma once

#include <stddef.h>
#include <stdint.h>

#define SYCLFFT_PROVIDER_ABI_VERSION 1u

#if defined(_WIN32)
#  define SYCLFFT_PROVIDER_EXPORT extern "C" __declspec(dllexport)
#else
#  define SYCLFFT_PROVIDER_EXPORT extern "C" __attribute__((visibility("default")))
#endif

typedef enum syclfft_scalar_v1 {
  SYCLFFT_SCALAR_FLOAT_V1 = 1,
  SYCLFFT_SCALAR_DOUBLE_V1 = 2,
} syclfft_scalar_v1;

typedef struct syclfft_host_plan_config_v1 {
  uint32_t abi_version;
  syclfft_scalar_v1 scalar;
  int32_t direction;
  uint32_t rank;
  uint64_t lengths[3];
  uint64_t batch_count;
  uint32_t in_place;
  uint32_t measure;
} syclfft_host_plan_config_v1;

typedef struct syclfft_host_provider_v1 {
  uint32_t abi_version;
  const char* name;
  void* (*create)(const syclfft_host_plan_config_v1*, char*, size_t);
  void (*destroy)(void*);
  int (*execute)(void*, void*, void*, char*, size_t);
} syclfft_host_provider_v1;

typedef const syclfft_host_provider_v1* (*syclfft_get_host_provider_v1_fn)();

SYCLFFT_PROVIDER_EXPORT const syclfft_host_provider_v1* syclfft_get_host_provider_v1();
