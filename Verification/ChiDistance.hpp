#ifndef VERIFICATION_CHIDISTANCE_H
#define VERIFICATION_CHIDISTANCE_H

#include <opencv2/ml/ml.hpp>
#include "SVM.hpp"

//Class implementing Chi^2 distance, used in "DeepFace" paper by Facebook
class ChiDistance
{

public:
  ChiDistance(Configuration& config);
  void train();
  int compare(cv::Mat& featureOne, cv::Mat& featureTwo);
  void verifyVal();
  ~ChiDistance();



private:
  //transform data from classification task to verfication
  void prepateTrainData(cv::Mat& features, cv::Mat& labels, std::vector<int>& labelsVec);
  //get feature vector of Chi
  void transformData(cv::Mat f1, cv::Mat f2, cv::Mat& featChi);
  CvSVM*           _comparator;
  std::string      _pathComparator;
  float            _threshold;
  //data for training
  std::string      _pathTrainFeatures;
  std::string      _pathValFeatures;
  struct Features* _trainFeatures    = NULL;
  SVMLinear*       _comparatorLinear = NULL;

};

#endif