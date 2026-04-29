#ifndef DECODE_HPP__
#define DECODE_HPP__

#include <cuda_runtime.h>
#include <memory>

namespace cuda
{

// yolo v8 v11 目标检测后处理kernel
__global__ void decode_kernel_v11(float *predict,
                                  int num_bboxes,
                                  int num_classes,
                                  int output_cdim,
                                  float confidence_threshold,
                                  float *invert_affine_matrix,
                                  float *parray,
                                  int *box_count,
                                  int max_image_boxes,
                                  int num_box_element,
                                  int start_x,
                                  int start_y,
                                  int batch_index);

// yolo nms kernel
__global__ void
fast_nms_kernel(float *bboxes, int *box_count, int max_image_boxes, float threshold, int num_box_element);

__global__ void decode_dfine_kernel(
    int64_t* labels, 
    float* scores, 
    float* boxes, 
    int num_bboxes,
    float confidence_threshold, 
    int *box_count, 
    int start_x,
    int start_y, 
    float* result, 
    int max_image_boxes, 
    int num_box_element
    );

__global__ void decode_dfine_kernel(
    int32_t* labels, 
    float* scores, 
    float* boxes, 
    int num_bboxes,
    float confidence_threshold, 
    int *box_count, 
    int start_x,
    int start_y, 
    float* result, 
    int max_image_boxes, 
    int num_box_element
    );

} // namespace cuda

#endif
