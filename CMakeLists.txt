cmake_minimum_required(VERSION 3.24)

project(syclfft VERSION 0.1.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(BUILD_SHARED_LIBS "Build shared syclfft libraries" ON)
option(SYCLFFT_BUILD_HOST "Build the FFTW host API and provider" ON)
if(APPLE)
  option(SYCLFFT_BUILD_SYCL "Build the SYCL API" OFF)
else()
  option(SYCLFFT_BUILD_SYCL "Build the SYCL API" ON)
endif()
set(SYCLFFT_SYCL_IMPLEMENTATION "AdaptiveCpp" CACHE STRING "AdaptiveCpp or DPCPP")
set_property(CACHE SYCLFFT_SYCL_IMPLEMENTATION PROPERTY STRINGS AdaptiveCpp DPCPP)
set(SYCLFFT_CUFFT "AUTO" CACHE STRING "Build cuFFT provider: AUTO, ON, or OFF")
set_property(CACHE SYCLFFT_CUFFT PROPERTY STRINGS AUTO ON OFF)
option(SYCLFFT_USE_SYSTEM_FFTW "Find FFTW outside Conan-generated package files" OFF)
option(SYCLFFT_BUILD_TESTS "Build tests" OFF)
option(SYCLFFT_BUILD_EXAMPLES "Build examples" ON)

if(NOT BUILD_SHARED_LIBS)
  message(FATAL_ERROR "syclfft v0.1 supports shared core libraries and provider modules only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

add_library(syclfft_core SHARED src/core/provider_loader.cpp)
add_library(syclfft::core ALIAS syclfft_core)
target_include_directories(syclfft_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_definitions(syclfft_core PRIVATE SYCLFFT_BUILDING_CORE)
target_compile_features(syclfft_core PUBLIC cxx_std_20)
set_target_properties(syclfft_core PROPERTIES
  EXPORT_NAME core
  VERSION ${PROJECT_VERSION}
  SOVERSION 0
)
if(CMAKE_DL_LIBS)
  target_link_libraries(syclfft_core PRIVATE ${CMAKE_DL_LIBS})
endif()

if(WIN32)
  set(SYCLFFT_INSTALL_PROVIDER_DIR "${CMAKE_INSTALL_BINDIR}/syclfft/providers")
else()
  set(SYCLFFT_INSTALL_PROVIDER_DIR "${CMAKE_INSTALL_LIBDIR}/syclfft/providers")
endif()
set(SYCLFFT_BUILD_PROVIDER_DIR "${CMAKE_BINARY_DIR}/syclfft/providers")

if(SYCLFFT_BUILD_HOST)
  if(SYCLFFT_USE_SYSTEM_FFTW)
    find_package(FFTW3 MODULE REQUIRED COMPONENTS FLOAT DOUBLE)
  else()
    find_package(FFTW3 CONFIG REQUIRED)
    if(NOT TARGET FFTW3::fftw3 OR NOT TARGET FFTW3::fftw3f)
      message(FATAL_ERROR "FFTW3 package must provide FFTW3::fftw3 and FFTW3::fftw3f")
    endif()
  endif()

  add_library(syclfft_host SHARED src/host/host.cpp)
  add_library(syclfft::host ALIAS syclfft_host)
  target_link_libraries(syclfft_host PUBLIC syclfft_core)
  target_compile_features(syclfft_host PUBLIC cxx_std_20)
  target_compile_definitions(syclfft_host PRIVATE SYCLFFT_BUILDING_HOST)
  set_target_properties(syclfft_host PROPERTIES
    EXPORT_NAME host
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
  )

  add_library(syclfft_provider_fftw MODULE src/providers/fftw/fftw_provider.cpp)
  target_link_libraries(syclfft_provider_fftw PRIVATE FFTW3::fftw3 FFTW3::fftw3f)
  target_include_directories(syclfft_provider_fftw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  set_target_properties(syclfft_provider_fftw PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${SYCLFFT_BUILD_PROVIDER_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${SYCLFFT_BUILD_PROVIDER_DIR}"
  )
  install(TARGETS syclfft_provider_fftw
    LIBRARY DESTINATION "${SYCLFFT_INSTALL_PROVIDER_DIR}"
    RUNTIME DESTINATION "${SYCLFFT_INSTALL_PROVIDER_DIR}"
  )
endif()

if(SYCLFFT_BUILD_SYCL)
  add_library(syclfft_sycl SHARED src/sycl/sycl.cpp)
  add_library(syclfft::sycl ALIAS syclfft_sycl)
  target_link_libraries(syclfft_sycl PUBLIC syclfft_core)
  target_compile_features(syclfft_sycl PUBLIC cxx_std_20)
  target_compile_definitions(syclfft_sycl PRIVATE SYCLFFT_BUILDING_SYCL)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include/syclfft/detail")
  configure_file(third_party/syclcplx/sycl_ext_complex.hpp
    "${CMAKE_CURRENT_BINARY_DIR}/include/syclfft/detail/sycl_ext_complex.hpp" COPYONLY)
  target_include_directories(syclfft_sycl
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>)
  set_target_properties(syclfft_sycl PROPERTIES
    EXPORT_NAME sycl
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
  )

  if(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
    find_package(AdaptiveCpp CONFIG REQUIRED)
    target_compile_definitions(syclfft_sycl PUBLIC SYCLFFT_USE_ADAPTIVECPP=1)
    add_sycl_to_target(TARGET syclfft_sycl SOURCES src/sycl/sycl.cpp)
  elseif(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "DPCPP")
    target_compile_options(syclfft_sycl PUBLIC -fsycl)
    target_link_options(syclfft_sycl PUBLIC -fsycl)
    target_compile_definitions(syclfft_sycl PUBLIC SYCLFFT_USE_DPCPP=1)
  else()
    message(FATAL_ERROR "SYCLFFT_SYCL_IMPLEMENTATION must be AdaptiveCpp or DPCPP")
  endif()

  string(TOUPPER "${SYCLFFT_CUFFT}" _syclfft_cufft)
  if(_syclfft_cufft STREQUAL "TRUE")
    set(_syclfft_cufft "ON")
  elseif(_syclfft_cufft STREQUAL "FALSE")
    set(_syclfft_cufft "OFF")
  elseif(NOT _syclfft_cufft MATCHES "^(AUTO|ON|OFF)$")
    message(FATAL_ERROR "SYCLFFT_CUFFT must be AUTO, ON, or OFF")
  endif()
  if(NOT _syclfft_cufft STREQUAL "OFF")
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      add_library(syclfft_provider_cufft MODULE src/providers/cufft/cufft_provider.cpp)
      target_link_libraries(syclfft_provider_cufft PRIVATE syclfft_sycl CUDA::cufft CUDA::cudart)
      target_include_directories(syclfft_provider_cufft PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
      set_target_properties(syclfft_provider_cufft PROPERTIES
        PREFIX ""
        LIBRARY_OUTPUT_DIRECTORY "${SYCLFFT_BUILD_PROVIDER_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${SYCLFFT_BUILD_PROVIDER_DIR}"
      )
      if(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
        add_sycl_to_target(TARGET syclfft_provider_cufft SOURCES src/providers/cufft/cufft_provider.cpp)
      else()
        target_compile_options(syclfft_provider_cufft PRIVATE -fsycl)
        target_link_options(syclfft_provider_cufft PRIVATE -fsycl)
      endif()
      install(TARGETS syclfft_provider_cufft
        LIBRARY DESTINATION "${SYCLFFT_INSTALL_PROVIDER_DIR}"
        RUNTIME DESTINATION "${SYCLFFT_INSTALL_PROVIDER_DIR}"
      )
      target_compile_definitions(syclfft_sycl PRIVATE SYCLFFT_HAS_CUFFT_PROVIDER=1)
    elseif(_syclfft_cufft STREQUAL "ON")
      message(FATAL_ERROR "SYCLFFT_CUFFT=ON, but CUDAToolkit was not found")
    else()
      message(STATUS "CUDAToolkit not found; cuFFT provider will not be built")
    endif()
  endif()
endif()

if(TARGET syclfft_sycl AND TARGET syclfft_host)
  target_link_libraries(syclfft_sycl PRIVATE syclfft_host)
  target_compile_definitions(syclfft_sycl PRIVATE SYCLFFT_HAS_FFTW_PROVIDER=1)
endif()

install(TARGETS syclfft_core
  EXPORT syclfftTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
if(TARGET syclfft_host)
  install(TARGETS syclfft_host EXPORT syclfftTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
if(TARGET syclfft_sycl)
  install(TARGETS syclfft_sycl EXPORT syclfftTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

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
