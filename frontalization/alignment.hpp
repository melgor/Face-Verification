#ifndef FRONT_ALIGNMENT_HPP
#define FRONT_ALIGNMENT_HPP

#include <opencv2/core/core.hpp>
#include "utils/Utils.hpp"
#include "cameramodel.hpp"

//Class interface for Face Aligment algorithm
class Alignment
{
  public:
    Alignment(){};
    Alignment(Configuration& config, CameraModel* camera){};
    virtual void frontalize(cv::Mat& image, cv::Rect& faceRect, cv::Mat& cameraModel, cv::Mat& outFrontal) = 0;
    virtual ~Alignment(){};

  protected:
    bool           _applySymetry;
    float          ACC_CONST = 800;
    cv::Rect       _centralFace; 
  
};

#endif