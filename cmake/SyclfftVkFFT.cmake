include_guard(DIRECTORY)

function(syclfft_add_vkfft_compiler)
  set(llvm_version "${ACPP_LLVM_VERSION}")
  if(SYCLFFT_EXPECTED_LLVM_VERSION)
    if(llvm_version AND NOT llvm_version VERSION_EQUAL SYCLFFT_EXPECTED_LLVM_VERSION)
      message(FATAL_ERROR
        "VkFFT must use the LLVM release reported by AdaptiveCpp (${llvm_version})")
    endif()
    set(llvm_version "${SYCLFFT_EXPECTED_LLVM_VERSION}")
  endif()
  if(NOT llvm_version)
    message(FATAL_ERROR
      "VkFFT requires AdaptiveCpp's LLVM release. For an older SDK, set "
      "SYCLFFT_EXPECTED_LLVM_VERSION to the exact release used to build AdaptiveCpp.")
  endif()

  find_package(LLVM "${llvm_version}" EXACT CONFIG REQUIRED HINTS "${ACPP_LLVM_CMAKE_DIR}")
  if(LLVM_VERSION_MAJOR LESS 15 OR LLVM_VERSION_MAJOR GREATER 21)
    message(FATAL_ERROR "The VkFFT compiler requires LLVM 15 through 21, found ${LLVM_PACKAGE_VERSION}")
  endif()

  file(REAL_PATH "${LLVM_DIR}" llvm_config_dir)
  cmake_path(GET llvm_config_dir PARENT_PATH llvm_cmake_dir)
  set(clang_hint "${llvm_cmake_dir}/clang")
  set(llvm_version "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}.${LLVM_VERSION_PATCH}")
  if(LLVM_VERSION_MAJOR EQUAL 15)
    # LLVM 15 installs ClangConfig.cmake without a package version file.
    find_package(Clang CONFIG REQUIRED PATHS "${clang_hint}" NO_DEFAULT_PATH)
    find_file(clang_version_header NAMES clang/Basic/Version.inc
      PATHS ${CLANG_INCLUDE_DIRS} NO_DEFAULT_PATH NO_CACHE REQUIRED)
    file(STRINGS "${clang_version_header}" clang_version_line REGEX "^#define CLANG_VERSION_STRING ")
    if(NOT clang_version_line MATCHES "\"([0-9]+\\.[0-9]+\\.[0-9]+)"
       OR NOT CMAKE_MATCH_1 VERSION_EQUAL llvm_version)
      message(FATAL_ERROR "${Clang_DIR} must use Clang ${llvm_version} to match LLVM")
    endif()
  else()
    find_package(Clang "${llvm_version}" EXACT CONFIG REQUIRED HINTS "${clang_hint}")
  endif()
  file(REAL_PATH "${LLVM_DIR}" clang_llvm_config_dir)
  if(NOT clang_llvm_config_dir STREQUAL llvm_config_dir OR NOT LLVM_PACKAGE_VERSION VERSION_EQUAL llvm_version)
    message(FATAL_ERROR "${Clang_DIR} must use the LLVM installation at ${llvm_config_dir}")
  endif()

  if(NOT TARGET LLVM OR NOT TARGET clang-cpp)
    message(FATAL_ERROR "The VkFFT compiler requires the shared LLVM and clang-cpp libraries")
  endif()
  foreach(library IN ITEMS LLVM clang-cpp)
    get_target_property(library_type ${library} TYPE)
    if(NOT library_type STREQUAL "SHARED_LIBRARY")
      message(FATAL_ERROR "The VkFFT compiler requires ${library} as a shared library")
    endif()
  endforeach()

  set(_provider_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/providers/vkfft")
  add_executable(syclfft_vkfft_compiler "${_provider_dir}/vkfft_compiler_main.cpp")
  target_compile_features(syclfft_vkfft_compiler PRIVATE cxx_std_17)
  target_include_directories(syclfft_vkfft_compiler SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${CLANG_INCLUDE_DIRS})
  target_link_libraries(syclfft_vkfft_compiler PRIVATE clang-cpp LLVM)
  set_target_properties(syclfft_vkfft_compiler PROPERTIES
    OUTPUT_NAME syclfft-vkfft-compiler
    RUNTIME_OUTPUT_DIRECTORY "${SYCLFFT_BUILD_PROVIDER_DIR}"
    INSTALL_RPATH_USE_LINK_PATH TRUE
  )
endfunction()

function(syclfft_add_vkfft_provider)
  if(NOT SYCLFFT_CUDA_DRIVER_INCLUDE_DIR)
    set(SYCLFFT_CUDA_DRIVER_INCLUDE_DIR "${ACPP_CUDA_DRIVER_INCLUDE_DIR}")
  endif()
  if(NOT EXISTS "${SYCLFFT_CUDA_DRIVER_INCLUDE_DIR}/cuda.h")
    message(FATAL_ERROR
      "VkFFT needs CUDA Driver API headers from a driver-enabled AdaptiveCpp package. "
      "For a manual build, set SYCLFFT_CUDA_DRIVER_INCLUDE_DIR to the directory with cuda.h.")
  endif()
  if(NOT EXISTS "${SYCLFFT_VKFFT_INCLUDE_DIR}/vkFFT.h")
    message(FATAL_ERROR "Patched VkFFT headers are required. Install with Conan option with_vkfft=True.")
  endif()
  syclfft_add_vkfft_compiler()

  set(_provider_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/providers/vkfft")
  set(_source "${_provider_dir}/vkfft_provider.cpp")
  set(_compiler_client "${_provider_dir}/vkfft_compiler.cpp")
  syclfft_add_provider(vkfft "${_source}" "${_compiler_client}")
  add_dependencies(syclfft_provider_vkfft syclfft_vkfft_compiler)
  target_compile_definitions(syclfft_provider_vkfft PRIVATE VKFFT_BACKEND=1 VKFFT_EXTERNAL_CUDA_PTX=1)
  target_include_directories(syclfft_provider_vkfft SYSTEM PRIVATE
    "${SYCLFFT_VKFFT_INCLUDE_DIR}" "${SYCLFFT_CUDA_DRIVER_INCLUDE_DIR}")
  target_compile_definitions(syclfft_provider_vkfft PRIVATE _GNU_SOURCE)
  target_link_libraries(syclfft_provider_vkfft PRIVATE syclfft_sycl ${CMAKE_DL_LIBS})
  target_link_options(syclfft_provider_vkfft PRIVATE "LINKER:--no-undefined")
  set_target_properties(syclfft_provider_vkfft PROPERTIES INSTALL_RPATH "$ORIGIN/../..")
  add_sycl_to_target(TARGET syclfft_provider_vkfft SOURCES "${_source}")
  target_compile_definitions(syclfft_sycl PRIVATE SYCLFFT_HAS_VKFFT_PROVIDER=1)
  install(TARGETS syclfft_vkfft_compiler RUNTIME DESTINATION "${SYCLFFT_INSTALL_PROVIDER_DIR}")
  install(FILES
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../third_party/vkfft/VkFFT-MIT.txt"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../third_party/vkfft/Clang-LLVM-LICENSE.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/syclfft/licenses")
endfunction()
