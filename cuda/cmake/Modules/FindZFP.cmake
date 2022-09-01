#
# FindZFP
# -----------
#
# Try to find the ZFP library
#
# This module defines the following variables:
#
#   ZFP_FOUND        - System has ZFP
#   ZFP_INCLUDE_DIRS - The ZFP include directory
#   ZFP_LIBRARIES    - Link these to use ZFP
#
# and the following imported targets:
#   ZFP::zfp - The ZFP library target
#
# You can also set the following variable to help guide the search:
#   ZFP_ROOT - The install prefix for ZFP containing the
#              include and lib folders
#              Note: this can be set as a CMake variable or an
#                    environment variable.  If specified as a CMake
#                    variable, it will override any setting specified
#                    as an environment variable.

if(NOT ZFP_FOUND)
    if(NOT ZFP_ROOT)
        if(NOT ("$ENV{ZFP_ROOT}" STREQUAL ""))
            set(ZFP_ROOT "$ENV{ZFP_ROOT}")
        endif()
    endif()
    if(PKG_CONFIG_FOUND)
        set(_CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH})
        if(ZFP_ROOT)
            list(INSERT CMAKE_PREFIX_PATH 0 "${ZFP_ROOT}")
        elseif(NOT ENV{ZFP_ROOT} STREQUAL "")
            list(INSERT CMAKE_PREFIX_PATH 0 "$ENV{ZFP_ROOT}")
        endif()
        set(PKG_CONFIG_USE_CMAKE_PREFIX_PATH ON)
        pkg_check_modules(PC_ZFP zfp)
        set(CMAKE_PREFIX_PATH ${_CMAKE_PREFIX_PATH})
        unset(_CMAKE_PREFIX_PATH)
        if(PC_ZFP_FOUND)
            if(BUILD_SHARED_LIBS)
                set(_PC_TYPE)
            else()
                set(_PC_TYPE _STATIC)
            endif()
            set(ZFP_LIBRARIES ${PC_ZFP${_PC_TYPE}_LINK_LIBRARIES})
            set(ZFP_LIBRARY_HINT ${PC_ZFP${_PC_TYPE}_LIBRARY_DIRS})
            set(ZFP_INCLUDE_DIR ${PC_ZFP${_PC_TYPE}_INCLUDE_DIRS})
            set(ZFP_VERSION ${PC_ZFP_VERSION})
            find_library(ZFP_LIBRARY ZFP HINTS ${ZFP_LIBRARY_HINT})
            set(HAVE_ZFP TRUE)
        endif()
    endif()
    if(ZFP_ROOT AND NOT PC_ZFP_FOUND)
        set(ZFP_INCLUDE_OPTS HINTS ${ZFP_ROOT}/include)
        find_path(ZFP_INCLUDE_DIR zfp.h ${ZFP_INCLUDE_OPTS})
        find_library(ZFP_LIBRARY zfp HINTS ${ZFP_LIBRARY_HINT}) 
    endif()
    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(ZFP
        FOUND_VAR ZFP_FOUND
        REQUIRED_VARS ZFP_INCLUDE_DIR 
    )
    if(ZFP_FOUND)
        if(ZFP_FOUND AND NOT TARGET ZFP::zfp)
        add_library(ZFP::zfp UNKNOWN IMPORTED)
        set_target_properties(ZFP::zfp PROPERTIES
       	    IMPORTED_LOCATION             "${ZFP_LIBRARY}"
            INTERFACE_LINK_LIBRARIES      "${ZFP_LIBRARIES}"
            INTERFACE_INCLUDE_DIRECTORIES "${ZFP_INCLUDE_DIR}"
        )
        endif()
    endif()
endif()