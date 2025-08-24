cmake_minimum_required(VERSION 3.23)

project(diffvg VERSION 0.2.0 DESCRIPTION "Differentiable Vector Graphics")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options first (used below)
# Make CUDA optional and default to ON for source installs
option(DIFFVG_CUDA "Build diffvg with GPU code path?" ON)

# Use modern Python discovery (works with Python 3.10 - 3.12 and beyond)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
# Prefer system pybind11, fall back to vendored copy
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    # Allow older pybind11 CMake to run with modern CMake via policy minimum
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    add_subdirectory(pybind11)
endif()

if(DIFFVG_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
        message(WARNING "CUDA compiler not found. Building without CUDA support.")
        set(DIFFVG_CUDA OFF)
    else()
        message(STATUS "Build with CUDA support")
        enable_language(CUDA)
        # Prefer native CMake CUDA; require toolkit for cudart linkage
        find_package(CUDAToolkit QUIET)
        if(NOT CUDAToolkit_FOUND)
            message(WARNING "CUDAToolkit not found. Building without CUDA support.")
            set(DIFFVG_CUDA OFF)
        else()
            set(CMAKE_CUDA_STANDARD 14)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)
            # Determine CUDA architectures robustly
            # Priority: CMAKE_CUDA_ARCHITECTURES (user) > TORCH_CUDA_ARCH_LIST env > DIFFVG_CUDA_ARCHS env > default
            if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
                set(_ENV_ARCH_LIST "")
                if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
                    set(_ENV_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
                elseif(DEFINED ENV{DIFFVG_CUDA_ARCHS})
                    set(_ENV_ARCH_LIST "$ENV{DIFFVG_CUDA_ARCHS}")
                endif()
                if(NOT _ENV_ARCH_LIST STREQUAL "")
                    string(REPLACE "," ";" _ENV_ARCH_LIST "${_ENV_ARCH_LIST}")
                    string(REPLACE " " ";" _ENV_ARCH_LIST "${_ENV_ARCH_LIST}")
                    set(_ARCH_LIST "")
                    foreach(a IN LISTS _ENV_ARCH_LIST)
                        string(REGEX REPLACE "^sm_" "" a "${a}")
                        string(REGEX REPLACE "^compute_" "" a "${a}")
                        list(APPEND _ARCH_LIST "${a}")
                    endforeach()
                    set(CMAKE_CUDA_ARCHITECTURES "${_ARCH_LIST}" CACHE STRING "CUDA architectures" FORCE)
                else()
                    # Default to Ada (RTX 40-series) if not specified
                    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
                endif()
            endif()
            # Safety override: CUDA 12+ has issues with very old arch (e.g., 52). Avoid < 70 by default.
            if(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.0)
                set(_ARCHS_FILTERED "")
                foreach(a IN LISTS CMAKE_CUDA_ARCHITECTURES)
                    if(a GREATER_EQUAL 70)
                        list(APPEND _ARCHS_FILTERED "${a}")
                    endif()
                endforeach()
                if(_ARCHS_FILTERED STREQUAL "")
                    message(WARNING "No supported GPU arch >= 70 found in '${CMAKE_CUDA_ARCHITECTURES}' for CUDA ${CUDAToolkit_VERSION}. Using 89. Set -DCMAKE_CUDA_ARCHITECTURES=75;86;89 to customize.")
                    set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures" FORCE)
                else()
                    if(NOT _ARCHS_FILTERED STREQUAL "${CMAKE_CUDA_ARCHITECTURES}")
                        message(STATUS "Filtering unsupported GPU arch < 70 from '${CMAKE_CUDA_ARCHITECTURES}' for CUDA ${CUDAToolkit_VERSION}. Using '${_ARCHS_FILTERED}'.")
                        set(CMAKE_CUDA_ARCHITECTURES "${_ARCHS_FILTERED}" CACHE STRING "CUDA architectures" FORCE)
                    endif()
                endif()
            endif()
        endif()
    endif()
else()
    message(STATUS "Build without CUDA support")
    # Use Thrust headers from CUDA Toolkit (preferred) or system install
endif()

# include_directories(${CMAKE_SOURCE_DIR}/pybind11/include)
include_directories(${Python_INCLUDE_DIRS})

# Use CCCL/Thrust from the CUDA Toolkit when available; this applies to both
# CPU and CUDA builds (headers are usable on host-only). As a fallback for
# CPU-only builds without the Toolkit, try to locate a system-installed Thrust.
# No vendored submodule is required.
if(NOT CUDAToolkit_FOUND)
    find_package(CUDAToolkit QUIET)
endif()

set(_DIFFVG_ADDED_THRUST_INCLUDE FALSE)
if(CUDAToolkit_FOUND)
    message(STATUS "Using Thrust/CCCL from CUDA Toolkit at: ${CUDAToolkit_INCLUDE_DIRS}")
    include_directories(${CUDAToolkit_INCLUDE_DIRS})
    set(_DIFFVG_ADDED_THRUST_INCLUDE TRUE)
endif()

if(NOT _DIFFVG_ADDED_THRUST_INCLUDE)
    find_package(Thrust QUIET)
    if(THRUST_FOUND)
        message(STATUS "Using system Thrust at: ${THRUST_INCLUDE_DIR}")
        include_directories(${THRUST_INCLUDE_DIR})
        set(_DIFFVG_ADDED_THRUST_INCLUDE TRUE)
    endif()
endif()

if(NOT _DIFFVG_ADDED_THRUST_INCLUDE AND NOT DIFFVG_CUDA)
    message(FATAL_ERROR "No Thrust/CCCL headers found. Install CUDA Toolkit (preferred) or provide a system Thrust install.")
endif()

if(DIFFVG_CUDA)
    # CUDAToolkit::cudart will be linked below
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>)
endif()

if(NOT MSVC)
  # These compile definitions are not meaningful for MSVC
  add_compile_options(-Wall -g -O3 -fvisibility=hidden -Wno-unknown-pragmas)
else()
  add_compile_options(/Wall /Zi)
  add_link_options(/DEBUG)
endif()

if(NOT DIFFVG_CUDA)
    add_compile_options("-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")
endif()

set(SRCS atomic.h
         color.h
         cdf.h
         cuda_utils.h
         diffvg.h
         edge_query.h
         filter.h
         matrix.h
         parallel.h
         pcg.h
         ptr.h
         sample_boundary.h
         scene.h
         shape.h
         solve.h
         vector.h
         within_distance.h
         winding_number.h
         atomic.cpp
         color.cpp
         diffvg.cpp
         parallel.cpp
         scene.cpp
         shape.cpp)

add_library(diffvg MODULE ${SRCS})
if(DIFFVG_CUDA)
    target_compile_definitions(diffvg PRIVATE COMPILE_WITH_CUDA)
    # Compile select sources with CUDA language to enable device code
    set_source_files_properties(
        diffvg.cpp
        scene.cpp
        PROPERTIES LANGUAGE CUDA)
endif()

if(APPLE)
    # The "-undefined dynamic_lookup" is a hack for systems with
    # multiple Python installed. If we link a particular Python version
    # here, and we import it with a different Python version later.
    # likely a segmentation fault.
    # The solution for Linux Mac OS machines, as mentioned in 
    # https://github.com/pybind/pybind11/blob/master/tools/pybind11Tools.cmake
    # is to not link against Python library at all and resolve the symbols
    # at compile time.
    set(DYNAMIC_LOOKUP "-undefined dynamic_lookup")
endif()

target_link_libraries(diffvg ${DYNAMIC_LOOKUP} pybind11::module)
if(DIFFVG_CUDA)
    # Link against shared cudart to avoid duplication with PyTorch CUDA runtime
    target_link_libraries(diffvg CUDA::cudart)
endif()

if(WIN32)
    # See: https://pybind11.readthedocs.io/en/master/compiling.html#advanced-interface-library-target
    target_link_libraries(diffvg pybind11::module)
    set_target_properties(diffvg PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                            SUFFIX "${PYTHON_MODULE_EXTENSION}")
endif()

set_target_properties(diffvg PROPERTIES SKIP_BUILD_RPATH FALSE)
set_target_properties(diffvg PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
if(UNIX AND NOT APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    set_target_properties(diffvg PROPERTIES INSTALL_RPATH "@loader_path")
endif()

set_property(TARGET diffvg PROPERTY CXX_STANDARD 14)
set_target_properties(diffvg PROPERTIES PREFIX "")
# Still enable assertion in release mode
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
string( REPLACE "/DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
string( REPLACE "-DNDEBUG" "" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")



# Install the extension module into the top-level of the wheel/site-packages
install(TARGETS diffvg
        LIBRARY DESTINATION .
        RUNTIME DESTINATION .)
