#pragma once

#include <stddef.h>
#include <stdint.h>

/* This header is a C ABI. C constructs and fixed-size fields are intentional. */
/* NOLINTBEGIN */

#define SYCLFFT_PROVIDER_ABI_VERSION 1u

#if defined(__cplusplus)
#    define SYCLFFT_PROVIDER_EXTERN_C extern "C"
#else
#    define SYCLFFT_PROVIDER_EXTERN_C
#endif

#if defined(_WIN32)
#    define SYCLFFT_PROVIDER_EXPORT SYCLFFT_PROVIDER_EXTERN_C __declspec(dllexport)
#else
#    define SYCLFFT_PROVIDER_EXPORT SYCLFFT_PROVIDER_EXTERN_C __attribute__((visibility("default")))
#endif

typedef uint32_t syclfft_scalar_v1;

#define SYCLFFT_SCALAR_UNKNOWN_V1 0u
#define SYCLFFT_SCALAR_FLOAT_V1 1u
#define SYCLFFT_SCALAR_DOUBLE_V1 2u
#define SYCLFFT_DIRECTION_FORWARD_V1 (-1)
#define SYCLFFT_DIRECTION_BACKWARD_V1 1

typedef struct syclfft_host_plan_config_v1
{
    uint32_t abi_version;
    syclfft_scalar_v1 scalar;
    int32_t direction;
    uint32_t rank;
    uint64_t lengths[3];
    uint64_t batch_count;
    uint32_t in_place;
    uint32_t measure;
} syclfft_host_plan_config_v1;

typedef struct syclfft_host_provider_v1
{
    uint32_t abi_version;
    const char *name;
    void *(*create)(const syclfft_host_plan_config_v1 *, char *, size_t);
    void (*destroy)(void *);
    int (*execute)(void *, void *, void *, char *, size_t);
} syclfft_host_provider_v1;

typedef const syclfft_host_provider_v1 *(*syclfft_get_host_provider_v1_fn)(void);

SYCLFFT_PROVIDER_EXPORT const syclfft_host_provider_v1 *syclfft_get_host_provider_v1(void);

/* NOLINTEND */
