cmake_minimum_required(VERSION 3.22)

project(syclfft VERSION 0.1.0 LANGUAGES C CXX)

if(WIN32)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BUILD_SHARED_LIBS "Build shared syclfft libraries" ON)
option(SYCLFFT_BUILD_SYCL "Build the SYCL API" ON)
set(SYCLFFT_SYCL_IMPLEMENTATION "AdaptiveCpp" CACHE STRING "AdaptiveCpp or DPCPP")
set_property(CACHE SYCLFFT_SYCL_IMPLEMENTATION PROPERTY STRINGS AdaptiveCpp DPCPP)
set(SYCLFFT_FFTW "AUTO" CACHE STRING "Build FFTW provider: AUTO, ON, or OFF")
set_property(CACHE SYCLFFT_FFTW PROPERTY STRINGS AUTO ON OFF)
set(SYCLFFT_CUFFT "AUTO" CACHE STRING "Build cuFFT provider: AUTO, ON, or OFF")
set_property(CACHE SYCLFFT_CUFFT PROPERTY STRINGS AUTO ON OFF)
option(SYCLFFT_VKFFT "Build the optional toolkit-free VkFFT CUDA provider" OFF)
if(SYCLFFT_VKFFT AND
   (NOT SYCLFFT_BUILD_SYCL OR NOT SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp"
    OR NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"))
  message(FATAL_ERROR "SYCLFFT_VKFFT requires the AdaptiveCpp SYCL build on Linux")
endif()

option(SYCLFFT_USE_SYSTEM_FFTW "Find FFTW outside Conan-generated package files" OFF)
option(SYCLFFT_BUILD_TESTS "Build tests" OFF)
option(SYCLFFT_BUILD_EXAMPLES "Build examples" ON)
option(SYCLFFT_BUILD_BENCHMARKS "Build the C2C benchmark" OFF)

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "syclfft v0.1 supports shared core libraries and provider modules only")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(SyclfftProviders)

add_library(syclfft_core SHARED src/core/provider_loader.cpp)
add_library(syclfft::core ALIAS syclfft_core)
target_include_directories(syclfft_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(syclfft_core PRIVATE SYCLFFT_BUILDING_CORE)
target_compile_features(syclfft_core PUBLIC cxx_std_17)
set_target_properties(syclfft_core PROPERTIES EXPORT_NAME core)
target_link_libraries(syclfft_core PRIVATE ${CMAKE_DL_LIBS})
set(_syclfft_libraries syclfft_core)
set(SYCLFFT_PACKAGE_ADAPTIVECPP_VERSION "")

if(WIN32)
  set(SYCLFFT_INSTALL_PROVIDER_DIR "${CMAKE_INSTALL_BINDIR}/syclfft/providers")
else()
  set(SYCLFFT_INSTALL_PROVIDER_DIR "${CMAKE_INSTALL_LIBDIR}/syclfft/providers")
endif()
set(SYCLFFT_BUILD_PROVIDER_DIR "${CMAKE_BINARY_DIR}/syclfft/providers")

syclfft_configure_fftw()

if(SYCLFFT_BUILD_SYCL)
  add_library(syclfft_sycl SHARED src/sycl/sycl.cpp)
  add_library(syclfft::sycl ALIAS syclfft_sycl)
  target_link_libraries(syclfft_sycl PUBLIC syclfft_core)
  target_compile_features(syclfft_sycl PUBLIC cxx_std_17)
  target_compile_definitions(syclfft_sycl PRIVATE SYCLFFT_BUILDING_SYCL)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/syclfft/detail")
  configure_file(third_party/syclcplx/sycl_ext_complex.hpp
    "${CMAKE_CURRENT_BINARY_DIR}/include/syclfft/detail/sycl_ext_complex.hpp" COPYONLY)
  target_include_directories(syclfft_sycl
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
  set_target_properties(syclfft_sycl PROPERTIES EXPORT_NAME sycl)
  list(APPEND _syclfft_libraries syclfft_sycl)

  if(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
    if(SYCLFFT_EXPECTED_ACPP_VERSION)
      find_package(AdaptiveCpp "${SYCLFFT_EXPECTED_ACPP_VERSION}" EXACT CONFIG REQUIRED)
    else()
      find_package(AdaptiveCpp CONFIG REQUIRED)
    endif()
    set(SYCLFFT_PACKAGE_ADAPTIVECPP_VERSION "${AdaptiveCpp_VERSION}")
    target_compile_definitions(syclfft_sycl PUBLIC SYCLFFT_USE_ADAPTIVECPP=1)
    target_link_libraries(syclfft_sycl INTERFACE AdaptiveCpp::acpp-rt)
    add_sycl_to_target(TARGET syclfft_sycl SOURCES src/sycl/sycl.cpp)
  elseif(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "DPCPP")
    target_compile_options(syclfft_sycl PUBLIC -fsycl)
    target_link_options(syclfft_sycl PUBLIC -fsycl)
    target_compile_definitions(syclfft_sycl PUBLIC SYCLFFT_USE_DPCPP=1)
  else()
    message(FATAL_ERROR "SYCLFFT_SYCL_IMPLEMENTATION must be AdaptiveCpp or DPCPP")
  endif()

  syclfft_configure_cufft()

  if(SYCLFFT_VKFFT)
    include(SyclfftVkFFT)
    syclfft_add_vkfft_provider()
  endif()

  if(TARGET syclfft_provider_fftw)
    target_compile_definitions(syclfft_sycl PRIVATE SYCLFFT_HAS_FFTW_PROVIDER=1)
  endif()
endif()

set_target_properties(${_syclfft_libraries} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0)
install(TARGETS ${_syclfft_libraries}
  EXPORT syclfftTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES third_party/syclcplx/sycl_ext_complex.hpp
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/syclfft/detail)
install(FILES LICENSE NOTICE README.md
  DESTINATION ${CMAKE_INSTALL_DATADIR}/syclfft)
install(FILES third_party/syclcplx/LLVM-EXCEPTION.txt
  DESTINATION ${CMAKE_INSTALL_DATADIR}/syclfft/licenses)
install(EXPORT syclfftTargets
  FILE syclfftTargets.cmake
  NAMESPACE syclfft::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/syclfft)

configure_package_config_file(cmake/syclfftConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/syclfftConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/syclfft)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/syclfftConfigVersion.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/syclfftConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/syclfftConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/syclfft)

if(SYCLFFT_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
if(SYCLFFT_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(SYCLFFT_BUILD_BENCHMARKS)
  if(NOT TARGET syclfft_sycl)
    message(FATAL_ERROR "Building the C2C benchmark requires the SYCL library")
  endif()
  add_subdirectory(benchmarks)
endif()
