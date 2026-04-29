#include "common/check.hpp"
#include "kernels/kernel_warp.hpp"

#define GPU_BLOCK_THREADS 512

static dim3 grid_dims(int numJobs)
{
    int numBlockThreads = numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS;
    return dim3(((numJobs + numBlockThreads - 1) / (float)numBlockThreads));
}

static dim3 block_dims(int numJobs) { return numJobs < GPU_BLOCK_THREADS ? numJobs : GPU_BLOCK_THREADS; }

void warp_affine_bilinear_and_normalize_plane(uint8_t *src,
                                              int src_line_size,
                                              int src_width,
                                              int src_height,
                                              float *dst,
                                              int dst_width,
                                              int dst_height,
                                              float *matrix_2_3,
                                              uint8_t const_value,
                                              const norm_image::Norm &norm,
                                              cudaStream_t stream)
{
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(cuda::warp_affine_bilinear_and_normalize_plane_kernel<<<grid, block, 0, stream>>>(src,
                                                                                                  src_line_size,
                                                                                                  src_width,
                                                                                                  src_height,
                                                                                                  dst,
                                                                                                  dst_width,
                                                                                                  dst_height,
                                                                                                  const_value,
                                                                                                  matrix_2_3,
                                                                                                  norm));
}

void warp_affine_bilinear_single_channel_plane(float *src,
                                               int src_line_size,
                                               int src_width,
                                               int src_height,
                                               float *dst,
                                               int dst_width,
                                               int dst_height,
                                               float *matrix_2_3,
                                               float const_value,
                                               cudaStream_t stream)
{
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(cuda::warp_affine_bilinear_single_channel_kernel<<<grid, block, 0, stream>>>(src,
                                                                                             src_line_size,
                                                                                             src_width,
                                                                                             src_height,
                                                                                             dst,
                                                                                             dst_width,
                                                                                             dst_height,
                                                                                             const_value,
                                                                                             matrix_2_3));
}

void warp_affine_bilinear_single_channel_mask_plane(float *src,
                                                    int src_line_size,
                                                    int src_width,
                                                    int src_height,
                                                    uint8_t *dst,
                                                    int dst_width,
                                                    int dst_height,
                                                    float *matrix_2_3,
                                                    uint8_t const_value,
                                                    cudaStream_t stream)
{
    dim3 grid((dst_width + 31) / 32, (dst_height + 31) / 32);
    dim3 block(32, 32);

    checkKernel(cuda::warp_affine_bilinear_single_channel_mask_kernel<<<grid, block, 0, stream>>>(src,
                                                                                                  src_line_size,
                                                                                                  src_width,
                                                                                                  src_height,
                                                                                                  dst,
                                                                                                  dst_width,
                                                                                                  dst_height,
                                                                                                  const_value,
                                                                                                  matrix_2_3));
}

// 对 decode_kernel_v11 的包装
void decode_kernel_invoker_v11(float *predict,
                               int num_bboxes,
                               int num_classes,
                               int output_cdim,
                               float confidence_threshold,
                               float nms_threshold,
                               float *invert_affine_matrix,
                               float *parray,
                               int *box_count,
                               int max_image_boxes,
                               int num_box_element,
                               int start_x,
                               int start_y,
                               int batch_index,
                               cudaStream_t stream)
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);

    checkKernel(cuda::decode_kernel_v11<<<grid, block, 0, stream>>>(predict,
                                                                    num_bboxes,
                                                                    num_classes,
                                                                    output_cdim,
                                                                    confidence_threshold,
                                                                    invert_affine_matrix,
                                                                    parray,
                                                                    box_count,
                                                                    max_image_boxes,
                                                                    num_box_element,
                                                                    start_x,
                                                                    start_y,
                                                                    batch_index));
}

// 对 fast_nms_kernel 的包装
void fast_nms_kernel_invoker(
    float *parray, int *box_count, int max_image_boxes, float nms_threshold, int num_box_element, cudaStream_t stream)
{
    auto grid  = grid_dims(max_image_boxes);
    auto block = block_dims(max_image_boxes);
    checkKernel(cuda::fast_nms_kernel<<<grid, block, 0, stream>>>(parray,
                                                                  box_count,
                                                                  max_image_boxes,
                                                                  nms_threshold,
                                                                  num_box_element));
}

void slice_plane(const uint8_t *image,
                 uint8_t *outs,
                 int *slice_start_point,
                 const int width,
                 const int height,
                 const int slice_width,
                 const int slice_height,
                 const int slice_num_h,
                 const int slice_num_v,
                 cudaStream_t stream)
{
    int slice_total      = slice_num_h * slice_num_v;
    dim3 block(32, 32);
    dim3 grid((slice_width + block.x - 1) / block.x, (slice_height + block.y - 1) / block.y, slice_total);
    checkKernel(cuda::slice_kernel<<<grid, block, 0, stream>>>(image,
                                                                outs,
                                                                width,
                                                                height,
                                                                slice_width,
                                                                slice_height,
                                                                slice_num_h,
                                                                slice_num_v,
                                                                slice_start_point));
}


void decode_dfine_plan(
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
    int num_box_element,
    cudaStream_t stream
    )
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkKernel(cuda::decode_dfine_kernel<<<grid, block, 0, stream>>>(
        labels,
        scores,
        boxes,
        num_bboxes,
        confidence_threshold,
        box_count,
        start_x,
        start_y,
        result,
        max_image_boxes,
        num_box_element));

}


void decode_dfine_plan(
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
    int num_box_element,
    cudaStream_t stream
    )
{
    auto grid  = grid_dims(num_bboxes);
    auto block = block_dims(num_bboxes);
    checkKernel(cuda::decode_dfine_kernel<<<grid, block, 0, stream>>>(
        labels,
        scores,
        boxes,
        num_bboxes,
        confidence_threshold,
        box_count,
        start_x,
        start_y,
        result,
        max_image_boxes,
        num_box_element));

}
