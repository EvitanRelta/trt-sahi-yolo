cmake_minimum_required(VERSION 3.16)
project(trt-sahi-yolo LANGUAGES CXX CUDA)

# ── Options ──────────────────────────────────────────────────────────────────
option(TRT8 "Use TensorRT 8 API (default)" ON)

# ── CUDA architecture ───────────────────────────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES 75)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 17 via flags (CMake 3.16 doesn't know the CUDA17 dialect)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --std=c++17")

# ── Dependencies ─────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# TensorRT — not a standard CMake package; locate manually
find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)
find_library(NVINFER_LIB nvinfer
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib)

if(NOT TRT_INCLUDE_DIR OR NOT NVINFER_LIB)
    message(FATAL_ERROR "TensorRT not found. Set TRT_INCLUDE_DIR and NVINFER_LIB.")
endif()

# CUDA runtime library
find_library(CUDART_LIB cudart HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDA_DRIVER_LIB cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# ── Collect source files ────────────────────────────────────────────────────
file(GLOB_RECURSE ALL_CPP_SOURCES CONFIGURE_DEPENDS src/*.cpp)
file(GLOB_RECURSE ALL_CU_SOURCES  CONFIGURE_DEPENDS src/*.cu)

# Library sources: exclude main.cpp and example entry points
file(GLOB_RECURSE EXAMPLE_SOURCES CONFIGURE_DEPENDS src/examples/*.cpp)
set(MAIN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)

set(LIB_CPP_SOURCES ${ALL_CPP_SOURCES})
list(REMOVE_ITEM LIB_CPP_SOURCES ${MAIN_SOURCE} ${EXAMPLE_SOURCES})

set(LIB_CU_SOURCES ${ALL_CU_SOURCES})

# TRT version toggle
if(TRT8)
    list(REMOVE_ITEM LIB_CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/common/tensorrt.cpp)
else()
    list(REMOVE_ITEM LIB_CPP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/common/tensorrt8.cpp)
endif()

# ── Shared library: trtsahi ─────────────────────────────────────────────────
add_library(trtsahi SHARED ${LIB_CPP_SOURCES} ${LIB_CU_SOURCES})

target_include_directories(trtsahi
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${TRT_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(trtsahi
    PUBLIC
        ${OpenCV_LIBS}
        ${CUDART_LIB}
    PRIVATE
        ${NVINFER_LIB}
        ${CUDA_DRIVER_LIB}
        dl
)

target_compile_definitions(trtsahi PRIVATE
    $<$<BOOL:${TRT8}>:TRT8>
    $<$<NOT:$<BOOL:${TRT8}>>:TRT10>
)

target_compile_options(trtsahi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=611>
    $<$<COMPILE_LANGUAGE:CXX>:-w>
)

# ── Test executable ─────────────────────────────────────────────────────────
add_executable(trtsahi_test
    ${MAIN_SOURCE}
    ${EXAMPLE_SOURCES}
)

target_include_directories(trtsahi_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRT_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(trtsahi_test PRIVATE trtsahi)
