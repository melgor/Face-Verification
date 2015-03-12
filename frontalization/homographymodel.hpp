#ifndef FRONT_HOMOGRAPHYMODEL_HPP
#define FRONT_HOMOGRAPHYMODEL_HPP
#include "faceattribute.hpp"
//class which will align face in 2D using Homography transformation

class HomographyModel
{
public:
  HomographyModel(Configuration& config);
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
  ~HomographyModel();

 private: 
  FacePoints       _pointModel;
  std::vector<int> _idPoints;
  std::vector<int> _leftEyePoints;
  std::vector<int> _rightEyePoints;
};

#endif