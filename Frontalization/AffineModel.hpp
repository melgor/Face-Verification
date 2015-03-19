#ifndef FRONT_AFFINEMODEL_HPP
#define FRONT_AFFINEMODEL_HPP
#include "FaceAttribute.hpp"
//class which will align face in 2D using Homography transformation

class AffineModel
{
public:
  AffineModel(Configuration& config);
  void estimateCamera(
                        FacePoints& facesPoints,
                        cv::Size& imageSize,
                        cv::Mat& cameraModels
                        );
  void estimateCamera(
                        std::vector<FacePoints>& facesPoints,
                        std::vector<cv::Size>& imageSize,
                        std::vector<cv::Mat>& cameraModels
                        );
  ~AffineModel();

  FacePoints& getModel6();  
  FacePoints& getModel68();      

 private: 
  FacePoints       _pointModel6;
  FacePoints       _pointModel68;
  std::vector<int> _idPoints;
  std::vector<int> _leftEyePoints;
  std::vector<int> _rightEyePoints;
};

#endif