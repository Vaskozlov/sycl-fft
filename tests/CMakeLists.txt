if(TARGET syclfft_host)
  add_executable(syclfft_host_tests host_tests.cpp)
  target_link_libraries(syclfft_host_tests PRIVATE syclfft::host)
  add_test(NAME syclfft.host COMMAND syclfft_host_tests)
  set_tests_properties(syclfft.host PROPERTIES
    ENVIRONMENT "SYCLFFT_PLUGIN_PATH=${SYCLFFT_BUILD_PROVIDER_DIR}")

  add_library(syclfft_bad_fftw MODULE bad_fftw_provider.cpp)
  target_include_directories(syclfft_bad_fftw PRIVATE ${PROJECT_SOURCE_DIR}/include)
  set_target_properties(syclfft_bad_fftw PROPERTIES
    PREFIX ""
    OUTPUT_NAME "syclfft_provider_fftw"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bad-provider"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bad-provider")
  add_executable(syclfft_plugin_tests plugin_tests.cpp)
  target_link_libraries(syclfft_plugin_tests PRIVATE syclfft::host)
  add_dependencies(syclfft_plugin_tests syclfft_bad_fftw)
  add_test(NAME syclfft.plugin-abi COMMAND syclfft_plugin_tests)
  set_tests_properties(syclfft.plugin-abi PROPERTIES
    ENVIRONMENT "SYCLFFT_PLUGIN_PATH=${CMAKE_CURRENT_BINARY_DIR}/bad-provider")
endif()

if(TARGET syclfft_sycl)
  add_executable(syclfft_sycl_tests sycl_tests.cpp)
  target_link_libraries(syclfft_sycl_tests PRIVATE syclfft::sycl)
  if(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
    add_sycl_to_target(TARGET syclfft_sycl_tests SOURCES sycl_tests.cpp)
  endif()
  add_test(NAME syclfft.sycl COMMAND syclfft_sycl_tests)
  set_tests_properties(syclfft.sycl PROPERTIES
    ENVIRONMENT "SYCLFFT_PLUGIN_PATH=${SYCLFFT_BUILD_PROVIDER_DIR}")
endif()
