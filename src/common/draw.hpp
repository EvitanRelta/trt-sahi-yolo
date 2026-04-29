#pragma once

#ifndef DRAW_HPP
#define DRAW_HPP

#include "common/object.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <iomanip>
#include <functional>
#include <vector>
#include <tuple>

namespace draw {

namespace {

inline std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v) {
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i) {
    case 0: r = v; g = t; b = p; break;
    case 1: r = q; g = v; b = p; break;
    case 2: r = p; g = v; b = t; break;
    case 3: r = p; g = q; b = v; break;
    case 4: r = t; g = p; b = v; break;
    case 5: r = v; g = p; b = q; break;
    default: r = 1; g = 1; b = 1; break;
    }
    return {static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255)};
}

inline cv::Scalar color_for(const std::string &label) {
    std::hash<std::string> hasher;
    unsigned int id = static_cast<int>(hasher(label) & 0x7FFFFFFF);
    float h = ((id << 2) ^ 0x937151) % 100 / 100.0f;
    float s = (((id << 3) ^ 0x315793) % 100 / 100.0f) * 0.5f + 0.5f;
    auto [b, g, r] = hsv2bgr(h, s, 1.0f);
    return cv::Scalar(b, g, r);
}

} // anonymous namespace

inline void draw_detections(cv::Mat &img, const object::DetectionBoxArray &dets) {
    for (const auto &det : dets) {
        cv::Scalar color = color_for(det.class_name);
        cv::Rect rect(cv::Point(det.box.left, det.box.top),
                      cv::Point(det.box.right, det.box.bottom));
        cv::rectangle(img, rect, color, 2);

        std::ostringstream oss;
        oss << det.class_name << " " << std::fixed << std::setprecision(2) << det.score;
        std::string label = oss.str();

        int baseline = 0;
        double font_scale = 0.5;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, font_scale, 1, &baseline);

        cv::Point text_org(rect.x, rect.y - 5);
        if (text_org.y - text_size.height < 0)
            text_org.y = rect.y + text_size.height + 5;

        cv::rectangle(img,
                      cv::Point(text_org.x, text_org.y - text_size.height),
                      cv::Point(text_org.x + text_size.width, text_org.y + baseline),
                      color, cv::FILLED);
        cv::putText(img, label, text_org, cv::FONT_HERSHEY_SIMPLEX,
                    font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    }
}

} // namespace draw

#endif // DRAW_HPP
