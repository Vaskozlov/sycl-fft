if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.22)
  project(syclfft_benchmarks LANGUAGES C CXX)

  set(SYCLFFT_SYCL_IMPLEMENTATION "AdaptiveCpp" CACHE STRING "AdaptiveCpp or DPCPP")
  if(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
    find_package(AdaptiveCpp CONFIG REQUIRED)
  endif()

  # Build the library from the same revision as the benchmark by default.
  set(SYCLFFT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/.." CACHE PATH "syclfft source tree")
  set(SYCLFFT_BUILD_SYCL ON)
  set(SYCLFFT_BUILD_EXAMPLES OFF)
  set(SYCLFFT_BUILD_TESTS OFF)
  set(SYCLFFT_BUILD_BENCHMARKS OFF)
  add_subdirectory("${SYCLFFT_SOURCE_DIR}" syclfft-source)
endif()

add_executable(syclfft_benchmark sycl.cpp)
target_link_libraries(syclfft_benchmark PRIVATE syclfft::sycl)
if(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
  add_sycl_to_target(TARGET syclfft_benchmark SOURCES sycl.cpp)
endif()
