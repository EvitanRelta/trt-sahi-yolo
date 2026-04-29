# detect_trt_version.cmake
# Finds TensorRT include directory and reads NV_TENSORRT_MAJOR into <out_var>.
#
# Usage:
#   detect_trt_version(MY_TRT_MAJOR)
#   message(STATUS "TRT major: ${MY_TRT_MAJOR}")
#
function(detect_trt_version out_var)
    find_path(_trt_inc NvInfer.h
        HINTS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include)

    if(NOT _trt_inc)
        message(FATAL_ERROR
            "detect_trt_version: NvInfer.h not found. "
            "Set -DTRT_INCLUDE_DIR manually or install TensorRT.")
    endif()

    set(_header "${_trt_inc}/NvInferVersion.h")
    if(NOT EXISTS "${_header}")
        message(FATAL_ERROR
            "detect_trt_version: NvInferVersion.h not found in ${_trt_inc}.")
    endif()

    file(STRINGS "${_header}" _major_line
         REGEX "^#define[ \t]+NV_TENSORRT_MAJOR[ \t]+[0-9]+")
    if(NOT _major_line)
        message(FATAL_ERROR
            "detect_trt_version: Could not parse NV_TENSORRT_MAJOR from ${_header}.")
    endif()

    string(REGEX REPLACE ".*NV_TENSORRT_MAJOR[ \t]+([0-9]+).*" "\\1"
           _ver "${_major_line}")

    set(${out_var} "${_ver}" PARENT_SCOPE)
endfunction()
