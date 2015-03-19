#ifndef FRONT_FACEEXTRACTOR_HPP
#define FRONT_FACEEXTRACTOR_HPP

#include "FaceAttribute.hpp"
#include "CameraModel.hpp"
#include "Frontalization3D.hpp"
#include "AffineModel.hpp"

//Class for controling all flow of face Aligment
class FaceExtractor
{
public:
  FaceExtractor(Configuration& config);
  void getFrontalFace(std::vector<cv::Mat>& images, std::vector<cv::Mat>& outFrontal);
  void getFrontalFace(cv::Mat& images, cv::Mat& outFrontal);
  ~FaceExtractor();


private:
  FaceAttribute*    _faceatt;
  CameraModel*      _camera;
  Frontalization3D* _align;
  AffineModel*      _affine;
  std::string       _alignOption;
};

#endif