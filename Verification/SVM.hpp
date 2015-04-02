#ifndef VERIFICATION_SVM_H
#define VERIFICATION_SVM_H

#include <opencv2/core/core.hpp>
#include <liblinear-1.96/linear.h>
#include "Utils/Utils.hpp"

//Implement SVM for OpenCV using LibLinear
//Wrapers taken from https://github.com/xiaodongyang/CascadeSVMs

class SVMLinear
{
public:
  SVMLinear(Configuration& config);
  //learn SVM
  void learn(cv::Mat& features, std::vector<int>& labels);
  void saveModel(std::string name);
  void loadModel(std::string name); 
  ~SVMLinear();

private:
  // set parameters for LIBLINEAR
  void setParams();
  //convert Mat to format for LIBLIBEAR
  void loadData(const cv::Mat &features, const std::vector<int> &labels);
  // set class weights
  void setClassWeights(double wtpos, double wtneg);
  // test model on train set
  void evalModel();

  struct problem   _data;       // first half is positive and second half is negative
  struct parameter _param;    // training parameters
  struct model    *_classifier;  // trained classifier

  
};

#endif