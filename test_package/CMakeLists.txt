cmake_minimum_required(VERSION 3.24)
project(syclfft_test_package LANGUAGES CXX)

find_package(syclfft CONFIG REQUIRED)

add_executable(test_sycl sycl.cpp)
target_link_libraries(test_sycl PRIVATE syclfft::sycl)
if(SYCLFFT_TEST_IMPLEMENTATION STREQUAL "AdaptiveCpp")
  find_package(AdaptiveCpp CONFIG REQUIRED)
  add_sycl_to_target(TARGET test_sycl SOURCES sycl.cpp)
endif()
