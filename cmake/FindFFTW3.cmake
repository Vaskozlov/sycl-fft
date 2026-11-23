include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_FFTW3 QUIET fftw3)
  pkg_check_modules(PC_FFTW3F QUIET fftw3f)
endif()

find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h
  HINTS ${FFTW3_ROOT} ENV FFTW3_ROOT ${PC_FFTW3_INCLUDE_DIRS}
  PATH_SUFFIXES include)
find_library(FFTW3_DOUBLE_LIBRARY NAMES fftw3
  HINTS ${FFTW3_ROOT} ENV FFTW3_ROOT ${PC_FFTW3_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)
find_library(FFTW3_FLOAT_LIBRARY NAMES fftw3f
  HINTS ${FFTW3_ROOT} ENV FFTW3_ROOT ${PC_FFTW3F_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64)

if(FFTW3_FIND_VERSION)
  unset(FFTW3_VERSION)
  function(_fftw_version_from_library _library _module _output)
    get_filename_component(_libdir "${_library}" DIRECTORY)
    set(_metadata "${_libdir}/pkgconfig/${_module}.pc")
    if(EXISTS "${_metadata}")
      set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_metadata}")
      file(STRINGS "${_metadata}" _version_line REGEX "^Version:[ 	]*[^ 	]+")
      if(_version_line MATCHES "^Version:[ 	]*([^ 	]+)")
        set(${_output} "${CMAKE_MATCH_1}" PARENT_SCOPE)
      endif()
    endif()
  endfunction()

  _fftw_version_from_library("${FFTW3_DOUBLE_LIBRARY}" fftw3 _fftw_double_version)
  _fftw_version_from_library("${FFTW3_FLOAT_LIBRARY}" fftw3f _fftw_float_version)
  if(_fftw_double_version AND _fftw_double_version VERSION_EQUAL _fftw_float_version)
    set(FFTW3_VERSION "${_fftw_double_version}")
  endif()
  find_package_handle_standard_args(FFTW3
    REQUIRED_VARS FFTW3_INCLUDE_DIR FFTW3_DOUBLE_LIBRARY FFTW3_FLOAT_LIBRARY FFTW3_VERSION
    VERSION_VAR FFTW3_VERSION
    REASON_FAILURE_MESSAGE "Both FFTW libraries require matching version metadata in libdir/pkgconfig")
else()
  find_package_handle_standard_args(FFTW3
    REQUIRED_VARS FFTW3_INCLUDE_DIR FFTW3_DOUBLE_LIBRARY FFTW3_FLOAT_LIBRARY)
endif()

if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3)
  add_library(FFTW3::fftw3 UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw3 PROPERTIES
    IMPORTED_LOCATION "${FFTW3_DOUBLE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
endif()
if(FFTW3_FOUND AND NOT TARGET FFTW3::fftw3f)
  add_library(FFTW3::fftw3f UNKNOWN IMPORTED)
  set_target_properties(FFTW3::fftw3f PROPERTIES
    IMPORTED_LOCATION "${FFTW3_FLOAT_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${FFTW3_INCLUDE_DIR}")
endif()

mark_as_advanced(FFTW3_INCLUDE_DIR FFTW3_DOUBLE_LIBRARY FFTW3_FLOAT_LIBRARY)
