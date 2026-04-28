#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"
#include <cmath>

static std::vector<std::string> classes_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
};

void run_yolo11()
{
    printf("=== run_yolo11 ===\n");
    std::shared_ptr<InferBase> model_ = load("models/yolo11n.engine",
        ModelType::YOLO11,
        classes_names,
        0,
        0.5f,
        0.45f,
        1,
        false,
        0,
        0,
        0.0,
        0.0);
    if (!model_) {
        printf("Failed to load YOLO11 model!\n");
        return;
    }
    cv::Mat image = cv::imread("inference/persons.jpg");
    if (image.empty()) {
        printf("Failed to read image!\n");
        return;
    }
    std::vector<cv::Mat> images = {image};
    auto det = model_->forwards(images);
    for (int i = 0; i < (int)images.size(); i++)
    {
        printf("Batch %d: size : %d\n", i, (int)det[i].size());
        for (auto& d : det[i]) {
            float conf = d.score;
            if (std::isnan(conf) || conf > 1.0f || conf < 0.0f) {
                printf("Warning: unusual confidence value: %f\n", conf);
            }
        }
        osd(images[i], det[i]);
        cv::imwrite("result/yolo11.jpg", images[i]);
    }
    printf("=== run_yolo11 done ===\n");
}

void run_yolo11_sahi()
{
    printf("=== run_yolo11_sahi ===\n");
    std::shared_ptr<InferBase> model_ = load("models/yolo11n.engine",
        ModelType::YOLO11SAHI,
        classes_names,
        0,
        0.5f,
        0.45f,
        32,
        true,
        640,
        640,
        0.3,
        0.3);
    if (!model_) {
        printf("Failed to load YOLO11 SAHI model! (engine may need dynamic batch for SAHI)\n");
        return;
    }
    cv::Mat image = cv::imread("inference/persons.jpg");
    if (image.empty()) {
        printf("Failed to read image!\n");
        return;
    }
    std::vector<cv::Mat> images = {image};
    auto det = model_->forwards(images);
    if (det.empty()) {
        printf("YOLO11 SAHI inference returned empty results!\n");
        return;
    }
    for (int i = 0; i < (int)images.size(); i++)
    {
        printf("Batch %d: size : %d\n", i, (int)det[i].size());
        osd(images[i], det[i]);
        cv::imwrite("result/yolo11_sahi.jpg", images[i]);
    }
    printf("=== run_yolo11_sahi done ===\n");
}
