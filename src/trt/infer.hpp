#ifndef INFER_HPP__
#define INFER_HPP__

#include "common/object.hpp"
#include <iostream>
#include <variant>

enum class ModelType : int
{
    YOLO11    = 2,
    YOLO11SAHI = 3,
    DFINE     = 10,
    DFINESAHI = 11
};

namespace ModelTypeConverter
{
inline std::string to_string(ModelType type)
{
    switch (type)
    {
    case ModelType::YOLO11:
        return "YOLO11";
    case ModelType::YOLO11SAHI:
        return "YOLO11SAHI";
    case ModelType::DFINE:
        return "DFINE";
    case ModelType::DFINESAHI:
        return "DFINESAHI";
    default:
        return "UNKNOWN";
    }
}

inline ModelType from_string(const std::string &str)
{
    static const std::unordered_map<std::string, ModelType> str2enum = {
        {"YOLO11", ModelType::YOLO11},
        {"YOLO11SAHI", ModelType::YOLO11SAHI},
        {"DFINE", ModelType::DFINE},
        {"DFINESAHI", ModelType::DFINESAHI}
    };

    auto it = str2enum.find(str);
    if (it != str2enum.end())
    {
        return it->second;
    }
    throw std::invalid_argument("Invalid ModelType string: " + str);
}
} // namespace ModelTypeConverter

inline std::ostream &operator<<(std::ostream &os, ModelType type)
{
    os << ModelTypeConverter::to_string(type);
    return os;
}

using InferResult = std::vector<object::DetectionBoxArray>;
class InferBase
{
  public:
    virtual InferResult forwards(const std::vector<cv::Mat> &inputs, void *stream = nullptr) = 0;

    virtual ~InferBase() = default;

};

std::shared_ptr<InferBase> load(const std::string &model_path,
                                ModelType model_type,
                                const std::vector<std::string> &names,
                                int gpu_id                    = 0,
                                float confidence_threshold    = 0.5f,
                                float nms_threshold           = 0.45f,
                                int max_batch_size            = 1,
                                bool auto_slice               = false,
                                int slice_width               = 640,
                                int slice_height              = 640,
                                double slice_horizontal_ratio = 0.3,
                                double slice_vertical_ratio   = 0.3);

#endif // INFER_HPP__
