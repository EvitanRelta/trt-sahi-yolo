#include "trt/infer.hpp"
#include "osd/osd.hpp"
#include "common/object.hpp"
#include "common/timer.hpp"
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

void run_dfine()
{
    printf("=== run_dfine ===\n");
    std::shared_ptr<InferBase> model_ = load("models/dfine_n_coco.engine",
        ModelType::DFINE,
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
        printf("Failed to load D-FINE model!\n");
        return;
    }
    cv::Mat image = cv::imread("inference/persons.jpg");
    if (image.empty()) {
        printf("Failed to read image!\n");
        return;
    }
    std::vector<cv::Mat> images = {image};
    // Warmup
    for (int i = 0; i < 5; i++)
        model_->forwards(images);
    // Benchmark
    nv::EventTimer timer;
    for (int i = 0; i < 20; i++)
    {
        timer.start();
        auto det = model_->forwards(images);
        timer.stop();
    }
    // Final inference with output
    auto det = model_->forwards(images);
    if (det.empty()) {
        printf("D-FINE inference returned empty results!\n");
        return;
    }
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
        cv::imwrite("result/dfine.jpg", images[i]);
    }
    printf("=== run_dfine done ===\n");
}

void run_dfine_sahi()
{
    printf("=== run_dfine_sahi ===\n");
    std::shared_ptr<InferBase> model_ = load("models/dfine_n_coco.engine",
        ModelType::DFINESAHI,
        classes_names,
        0,
        0.5f,
        0.45f,
        1,
        true,
        0,
        0,
        0.0,
        0.0);
    if (!model_) {
        printf("Failed to load D-FINE SAHI model!\n");
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
        printf("D-FINE SAHI inference returned empty results!\n");
        return;
    }
    for (int i = 0; i < (int)images.size(); i++)
    {
        printf("Batch %d: size : %d\n", i, (int)det[i].size());
        osd(images[i], det[i]);
        cv::imwrite("result/dfine_sahi.jpg", images[i]);
    }
    printf("=== run_dfine_sahi done ===\n");
}
