#ifndef FRONT_FACEEXTRACTOR_HPP
#define FRONT_FACEEXTRACTOR_HPP

#include "FaceAttribute.hpp"
#include "CameraModel.hpp"
#include "Frontalization3D.hpp"
#include "AffineModel.hpp"

//Class for controling all flow of face Alignment
class FaceExtractor
{
public:
  FaceExtractor() {};
  FaceExtractor(Configuration& config);
  void getFrontalFace(cv::Mat& images, std::vector<cv::Mat>& outFrontal);
  void getFrontalFace(cv::Mat& images, cv::Mat& outFrontal);
  ~FaceExtractor();

  std::vector<cv::Rect> _faceRect;
private:
  void (FaceExtractor::*alignment)( 
                  cv::Mat& image
                , std::vector<cv::Rect>& faceRectangle
                , std::vector<FacePoints>& facesPoints
                , std::vector<cv::Mat>& outFrontal
                ) = NULL;
  void alignment2D( 
                  cv::Mat& image
                , std::vector<cv::Rect>& faceRectangle 
                , std::vector<FacePoints>& facesPoints
                , std::vector<cv::Mat>& outFrontal
                );
  void alignment3D( 
                  cv::Mat& image
                , std::vector<cv::Rect>& faceRectangle 
                , std::vector<FacePoints>& facesPoints
                , std::vector<cv::Mat>& outFrontal
                );


  FaceAttribute*    _faceatt;
  CameraModel*      _camera;
  Frontalization3D* _align;
  AffineModel*      _affine;
  std::string       _frontalization;
  //option for output face
  cv::Rect          _face2d;
  cv::Size          _face2dSize;
};

#endif