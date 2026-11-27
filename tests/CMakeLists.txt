add_executable(syclfft_span_tests span_tests.cpp)
target_link_libraries(syclfft_span_tests PRIVATE syclfft::core)
add_test(NAME syclfft.span COMMAND syclfft_span_tests)

add_library(syclfft_bad_fftw MODULE bad_fftw_provider.cpp)
target_include_directories(syclfft_bad_fftw PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(syclfft_bad_fftw PROPERTIES
  PREFIX ""
  OUTPUT_NAME "syclfft_provider_fftw"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bad-provider"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bad-provider")
add_executable(syclfft_plugin_tests plugin_tests.cpp)
target_link_libraries(syclfft_plugin_tests PRIVATE syclfft::core)
add_dependencies(syclfft_plugin_tests syclfft_bad_fftw)
add_test(NAME syclfft.plugin-abi COMMAND syclfft_plugin_tests)
set_tests_properties(syclfft.plugin-abi PROPERTIES
  ENVIRONMENT "SYCLFFT_PLUGIN_PATH=${CMAKE_CURRENT_BINARY_DIR}/bad-provider")

if(WIN32)
  set_property(TEST syclfft.span syclfft.plugin-abi PROPERTY ENVIRONMENT_MODIFICATION
    "PATH=path_list_prepend:$<TARGET_FILE_DIR:syclfft_core>")
endif()

if(TARGET syclfft_sycl)
  add_executable(syclfft_sycl_tests sycl_tests.cpp)
  target_include_directories(syclfft_sycl_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(syclfft_sycl_tests PRIVATE syclfft::sycl)
  if(SYCLFFT_SYCL_IMPLEMENTATION STREQUAL "AdaptiveCpp")
    add_sycl_to_target(TARGET syclfft_sycl_tests SOURCES sycl_tests.cpp)
  endif()
  add_test(NAME syclfft.sycl COMMAND syclfft_sycl_tests)
  set_tests_properties(syclfft.sycl PROPERTIES
    ENVIRONMENT "SYCLFFT_PLUGIN_PATH=${SYCLFFT_BUILD_PROVIDER_DIR}")
  if(WIN32)
    set_property(TEST syclfft.sycl PROPERTY ENVIRONMENT_MODIFICATION
      "PATH=path_list_prepend:$<TARGET_FILE_DIR:syclfft_core>"
      "PATH=path_list_prepend:$<TARGET_FILE_DIR:syclfft_sycl>")
    if(TARGET AdaptiveCpp::acpp-rt)
      set_property(TEST syclfft.sycl APPEND PROPERTY ENVIRONMENT_MODIFICATION
        "PATH=path_list_prepend:$<TARGET_FILE_DIR:AdaptiveCpp::acpp-rt>")
    endif()
  endif()
endif()
