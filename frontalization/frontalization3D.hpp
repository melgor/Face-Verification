#ifndef FRONT_FRONTALIZATION_HPP
#define FRONT_FRONTALIZATION_HPP
#include "faceattribute.hpp"
#include "cameramodel.hpp"


// Function of frontalization of faces: the purpose is to get same face image regardless of face rotations, translation
// This will help in face recognition
// Frontalization code is based on:
//"Tal Hassner, Shai Harel*, Eran Paz* and Roee Enbar, Effective Face Frontalization in Unconstrained Images"
// This version is rewritten version of it, translated from MATLAB to C++

class Frontalization3D
{
public:
  Frontalization3D(Configuration& config, CameraModel* camera);
  void frontalize(cv::Mat& faceImage, cv::Mat& cameraModel, cv::Mat& outFrontal);//, cv::Rect& rect);
  ~Frontalization3D();

private:
  float          ACC_CONST = 800;
  cv::Rect       _centralFace = cv::Rect(105,85,100,152);

  //variable for frontaliation, which can be calculated only one time
  cv::Mat  _bgind;
  cv::Mat  _bgindReshape;
  cv::Mat  _threedee;
  cv::Size _cameraRefUSize;
  cv::Mat  _eyeMask;
  bool     _applySymetry;

};

#endif