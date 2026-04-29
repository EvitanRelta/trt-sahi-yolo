# detect_cuda_arch.cmake
#
# Fallback for CMake < 3.18 (which lacks CMAKE_CUDA_ARCHITECTURES=native).
# Compiles and runs a small CUDA program at configure time to query the
# compute capability of the first available GPU, then returns it as a
# string like "75" (for sm_75).
#
# Usage:
#   include(cmake/detect_cuda_arch.cmake)
#   detect_cuda_arch(MY_ARCH)        # sets MY_ARCH to e.g. "75"
#   set(CMAKE_CUDA_ARCHITECTURES ${MY_ARCH})
#
# Errors out via FATAL_ERROR if no GPU is found or compilation fails,
# so the caller never gets an empty/invalid value.

function(detect_cuda_arch OUT_VAR)
    # Write a one-shot CUDA program that prints the GPU's major+minor as two digits.
    set(_DETECT_SRC "${CMAKE_BINARY_DIR}/_detect_cuda_arch.cu")
    file(WRITE "${_DETECT_SRC}" "
#include <cuda_runtime.h>
#include <stdio.h>
int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return 1;
    printf(\"%d%d\", prop.major, prop.minor);
    return 0;
}
")

    # --run compiles then immediately executes; stdout is captured as _CUDA_ARCH_OUTPUT.
    execute_process(
        COMMAND ${CMAKE_CUDA_COMPILER} -o "${CMAKE_BINARY_DIR}/_detect_cuda_arch" "${_DETECT_SRC}" --run
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
        OUTPUT_VARIABLE _CUDA_ARCH_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _CUDA_ARCH_RESULT
        ERROR_QUIET
    )

    # Clean up the temporary source and binary.
    file(REMOVE "${_DETECT_SRC}" "${CMAKE_BINARY_DIR}/_detect_cuda_arch")

    if(_CUDA_ARCH_RESULT EQUAL 0 AND _CUDA_ARCH_OUTPUT MATCHES "^[0-9]+$")
        set(${OUT_VAR} "${_CUDA_ARCH_OUTPUT}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR
            "Could not auto-detect CUDA architecture and CMAKE_CUDA_ARCHITECTURES is not set.\n"
            "  Pass -DCMAKE_CUDA_ARCHITECTURES=<sm> to cmake, e.g.:\n"
            "    cmake -DCMAKE_CUDA_ARCHITECTURES=75 ..")
    endif()
endfunction()
