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
  //predict class
  int predict_class(cv::Mat& features);
  //predict probability of assignes as same class
  float predict_prob(cv::Mat& features);
  //save and load model
  void saveModel(std::string name, std::string nameMat);
  void loadModel(std::string name, std::string nameMat);
  //convert Mat to format for LIBLIBEAR
  void loadData(const cv::Mat &features, const std::vector<int> &labels);
  // test model on loaded set
  void evalModel();
  ~SVMLinear();

private:
  // set parameters for LIBLINEAR
  void setParams();
  //prepare data for predicting
  void prepareData(cv::Mat& features, feature_node **x);
  // set class weights
  void setClassWeights(double wtpos, double wtneg);
  //extract weight from learned model
  void getDecisionFunction();
 

  struct problem   _data;       // first half is positive and second half is negative
  struct parameter _param;    // training parameters
  struct model    *_classifier;  // trained classifier
  //extracted weights from SVM, need for probability estimation
  struct SVM_Mat  *_classiferMat = NULL;
  // cv::Mat          _classifierWeights;
  // std::vector<float>    _classifierBias;

  
};

#endif