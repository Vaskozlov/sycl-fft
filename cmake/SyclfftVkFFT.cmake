include_guard(DIRECTORY)

function(syclfft_add_vkfft_compiler)
  get_filename_component(_acpp_compiler "${ACPP_COMPILER}" REALPATH)
  get_filename_component(_acpp_bin_dir "${_acpp_compiler}" DIRECTORY)
  set(_acpp_core "${_acpp_bin_dir}/../etc/AdaptiveCpp/acpp-core.json")
  if(NOT EXISTS "${_acpp_core}")
    set(_acpp_core "/etc/AdaptiveCpp/acpp-core.json")
  endif()
  if(EXISTS "${_acpp_core}")
    file(READ "${_acpp_core}" _acpp_metadata)
    string(JSON _acpp_llvm_major ERROR_VARIABLE _acpp_llvm_error
      GET "${_acpp_metadata}" plugin-llvm-version-major)
    string(JSON _acpp_clang ERROR_VARIABLE _acpp_clang_error
      GET "${_acpp_metadata}" default-clang)
    if(NOT _acpp_clang_error AND IS_ABSOLUTE "${_acpp_clang}")
      get_filename_component(_acpp_clang_path "${_acpp_clang}" REALPATH)
      get_filename_component(_acpp_clang_bin_dir "${_acpp_clang_path}" DIRECTORY)
      set(_llvm_hint "${_acpp_clang_bin_dir}/../lib/cmake/llvm")
      if(EXISTS "${_acpp_clang_path}")
        execute_process(COMMAND "${_acpp_clang_path}" --version
          OUTPUT_VARIABLE _acpp_clang_version_text ERROR_QUIET)
        if(_acpp_clang_version_text MATCHES "clang version ([0-9]+\\.[0-9]+\\.[0-9]+)")
          set(_acpp_clang_version "${CMAKE_MATCH_1}")
        endif()
      endif()
    endif()
  endif()

  if(SYCLFFT_EXPECTED_LLVM_VERSION)
    find_package(LLVM "${SYCLFFT_EXPECTED_LLVM_VERSION}" EXACT CONFIG REQUIRED HINTS "${_llvm_hint}")
  else()
    find_package(LLVM 20.1 CONFIG REQUIRED HINTS "${_llvm_hint}")
  endif()
  if(NOT LLVM_VERSION_MAJOR EQUAL 20)
    message(FATAL_ERROR "This VkFFT frontend uses the LLVM/Clang 20 API")
  endif()
  if(_acpp_llvm_major MATCHES "^[0-9]+$" AND NOT LLVM_VERSION_MAJOR EQUAL _acpp_llvm_major)
    message(FATAL_ERROR "${LLVM_DIR} must use the same LLVM major as AdaptiveCpp (${_acpp_llvm_major})")
  endif()
  if(_acpp_clang_version AND NOT LLVM_PACKAGE_VERSION VERSION_EQUAL _acpp_clang_version)
    message(FATAL_ERROR "Select LLVM_DIR matching AdaptiveCpp's Clang ${_acpp_clang_version}")
  endif()
  get_filename_component(_llvm_cmake_dir "${LLVM_DIR}" DIRECTORY)
  find_package(Clang ${LLVM_PACKAGE_VERSION} EXACT CONFIG REQUIRED HINTS "${_llvm_cmake_dir}/clang")
  if(NOT TARGET LLVM OR NOT TARGET clang-cpp)
    message(FATAL_ERROR "The VkFFT compiler requires the shared LLVM and clang-cpp libraries")
  endif()
  foreach(_library IN ITEMS LLVM clang-cpp)
    get_target_property(_kind ${_library} TYPE)
    if(NOT _kind STREQUAL "SHARED_LIBRARY")
      message(FATAL_ERROR "The VkFFT compiler requires ${_library} as a shared library")
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
  find_path(SYCLFFT_CUDA_DRIVER_INCLUDE_DIR NAMES cuda.h
    HINTS "${ACPP_CUDA_DRIVER_INCLUDE_DIR}"
    DOC "Directory containing the CUDA Driver API cuda.h header" NO_DEFAULT_PATH)
  if(NOT EXISTS "${SYCLFFT_CUDA_DRIVER_INCLUDE_DIR}/cuda.h")
    message(FATAL_ERROR
      "Set the Conan option cuda_driver_include_dir to the AdaptiveCpp checkout's third-party/nvidia/include directory")
  endif()
  if(NOT EXISTS "${SYCLFFT_VKFFT_INCLUDE_DIR}/vkFFT.h")
    message(FATAL_ERROR "Patched VkFFT headers are required. Install with Conan option with_vkfft=True.")
  endif()
  syclfft_add_vkfft_compiler()

  set(_provider_dir "${CMAKE_CURRENT_SOURCE_DIR}/src/providers/vkfft")
  set(_source "${_provider_dir}/vkfft_provider.cpp")
  set(_compiler_client "${_provider_dir}/vkfft_compiler.cpp")
  add_library(syclfft_provider_vkfft MODULE "${_source}" "${_compiler_client}")
  add_dependencies(syclfft_provider_vkfft syclfft_vkfft_compiler)
  target_compile_definitions(syclfft_provider_vkfft PRIVATE VKFFT_BACKEND=1 VKFFT_EXTERNAL_CUDA_PTX=1)
  target_include_directories(syclfft_provider_vkfft SYSTEM PRIVATE
    "${SYCLFFT_VKFFT_INCLUDE_DIR}" "${SYCLFFT_CUDA_DRIVER_INCLUDE_DIR}")
  target_compile_definitions(syclfft_provider_vkfft PRIVATE _GNU_SOURCE)
  target_link_libraries(syclfft_provider_vkfft PRIVATE syclfft_sycl ${CMAKE_DL_LIBS})
  target_link_options(syclfft_provider_vkfft PRIVATE "LINKER:--no-undefined")
  set_target_properties(syclfft_provider_vkfft PROPERTIES
    PREFIX ""
    LIBRARY_OUTPUT_DIRECTORY "${SYCLFFT_BUILD_PROVIDER_DIR}"
    INSTALL_RPATH "$ORIGIN/../.."
  )
  add_sycl_to_target(TARGET syclfft_provider_vkfft SOURCES "${_source}")
  target_compile_definitions(syclfft_sycl PRIVATE SYCLFFT_HAS_VKFFT_PROVIDER=1)
  install(TARGETS syclfft_provider_vkfft LIBRARY DESTINATION "${SYCLFFT_INSTALL_PROVIDER_DIR}")
  install(TARGETS syclfft_vkfft_compiler RUNTIME DESTINATION "${SYCLFFT_INSTALL_PROVIDER_DIR}")
  install(FILES
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../third_party/vkfft/VkFFT-MIT.txt"
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../third_party/vkfft/Clang-LLVM-LICENSE.txt"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/syclfft/licenses")
endfunction()
