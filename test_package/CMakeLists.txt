cmake_minimum_required(VERSION 3.24)
project(syclfft_test_package LANGUAGES CXX)

find_package(syclfft CONFIG REQUIRED)

if(SYCLFFT_TEST_HOST)
  add_executable(test_host host.cpp)
  target_link_libraries(test_host PRIVATE syclfft::host)
endif()

if(SYCLFFT_TEST_SYCL)
  add_executable(test_sycl sycl.cpp)
  target_link_libraries(test_sycl PRIVATE syclfft::sycl)
  if(SYCLFFT_TEST_IMPLEMENTATION STREQUAL "AdaptiveCpp")
    find_package(AdaptiveCpp CONFIG REQUIRED)
    add_sycl_to_target(TARGET test_sycl SOURCES sycl.cpp)
  endif()
endif()
