#include "kernels/decode.cuh"
#include <stdio.h>

namespace cuda
{

static __device__ void affine_project(float *matrix, float x, float y, float *ox, float *oy)
{
    *ox = matrix[0] * x + matrix[1] * y + matrix[2];
    *oy = matrix[3] * x + matrix[4] * y + matrix[5];
}

static __device__ float
box_iou(float aleft, float atop, float aright, float abottom, float bleft, float btop, float bright, float bbottom)
{
    float cleft   = max(aleft, bleft);
    float ctop    = max(atop, btop);
    float cright  = min(aright, bright);
    float cbottom = min(abottom, bbottom);

    float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
    if (c_area == 0.0f)
        return 0.0f;

    float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
    float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
    return c_area / (a_area + b_area - c_area);
}

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
                                  int batch_index)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    float *pitem            = predict + output_cdim * position;
    float *class_confidence = pitem + 4;
    float confidence        = *class_confidence++;
    int label               = 0;
    for (int i = 1; i < num_classes; ++i, ++class_confidence)
    {
        if (*class_confidence > confidence)
        {
            confidence = *class_confidence;
            label      = i;
        }
    }
    if (confidence < confidence_threshold)
        return;

    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes)
        return;

    float cx     = *pitem++;
    float cy     = *pitem++;
    float width  = *pitem++;
    float height = *pitem++;
    float left   = cx - width * 0.5f;
    float top    = cy - height * 0.5f;
    float right  = cx + width * 0.5f;
    float bottom = cy + height * 0.5f;
    affine_project(invert_affine_matrix, left, top, &left, &top);
    affine_project(invert_affine_matrix, right, bottom, &right, &bottom);

    float *pout_item = parray + index * num_box_element;
    *pout_item++     = left + start_x;
    *pout_item++     = top + start_y;
    *pout_item++     = right + start_x;
    *pout_item++     = bottom + start_y;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
    *pout_item++     = position;
    *pout_item++     = batch_index; // batch_index
}

__global__ void
fast_nms_kernel(float *bboxes, int *box_count, int max_image_boxes, float threshold, int num_box_element)
{
    int position = (blockDim.x * blockIdx.x + threadIdx.x);
    int count    = min((int)*box_count, max_image_boxes);
    if (position >= count)
        return;

    // left, top, right, bottom, confidence, class, keepflag, batch_index
    float *pcurrent = bboxes + position * num_box_element;
    for (int i = 0; i < count; ++i)
    {
        float *pitem = bboxes + i * num_box_element;
        if (i == position || pcurrent[5] != pitem[5])
            continue;

        if (pitem[4] >= pcurrent[4])
        {
            if (pitem[4] == pcurrent[4] && i < position)
                continue;

            float iou =
                box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3], pitem[0], pitem[1], pitem[2], pitem[3]);

            if (iou > threshold)
            {
                pcurrent[6] = 0; // 1=keep, 0=ignore
                return;
            }
        }
    }
}

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
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    int label = (int)*(labels + position);
    float confidence = *(scores + position);

    float *box = boxes + position * 4;

    if (confidence < confidence_threshold)
        return;
    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes)
        return;

    float *pout_item = result + num_box_element * index;

    *pout_item++     = box[0] + start_x;
    *pout_item++     = box[1] + start_y;
    *pout_item++     = box[2] + start_x;
    *pout_item++     = box[3] + start_y;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
    *pout_item++     = position;
    *pout_item++     = 0; // batch_index
}

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
)
{
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if (position >= num_bboxes)
        return;

    int label = (int)*(labels + position);
    float confidence = *(scores + position);

    float *box = boxes + position * 4;

    if (confidence < confidence_threshold)
        return;
    int index = atomicAdd(box_count, 1);
    if (index >= max_image_boxes)
        return;

    float *pout_item = result + num_box_element * index;

    *pout_item++     = box[0] + start_x;
    *pout_item++     = box[1] + start_y;
    *pout_item++     = box[2] + start_x;
    *pout_item++     = box[3] + start_y;
    *pout_item++     = confidence;
    *pout_item++     = label;
    *pout_item++     = 1; // 1 = keep, 0 = ignore
    *pout_item++     = position;
    *pout_item++     = 0; // batch_index
}

} // namespace cuda
