#ifndef VERIFICATOR_HPP
#define VERIFICATOR_HPP
#include "Utils/Utils.hpp"
#include "SVM.hpp"

//class for Verification task. Get new feature, then classify it based on DataBase

class Verificator
{
public:
  Verificator(struct Configuration& config);
  //train all dependies needed for Face Verification
  void train();
  //load verification data
  void verify();
  //compare if two features descrive same person or not
  int compare(cv::Mat& featureOne, cv::Mat& featureTwo);
  ~Verificator();

private:
  //transform data from classification task to verfication
  void prepareVerificationData(cv::Mat& scaledFetures, cv::Mat& featuresVer, std::vector<int>& labelsVecVer);
  //learn scale value from train data
  void learnScaleParam(cv::Mat& features);
  //scale data
  void scaleData(cv::Mat features, cv::Mat& scaledFeatures);
  void (*distanceFunction) (cv::Mat, cv::Mat, cv::Mat&) = NULL;
  //configuration
  std::string      _metric;
  std::string      _pathTrainFeatures;
  std::string      _pathValFeatures;
  struct Features* _trainFeatures = NULL;
  //data for scaling
  cv::Mat          _maxValue;
  std::string      _pathScaler;
  //learning algorithm
  SVMLinear*       _comparatorLinear = NULL;
  std::string      _pathComparator;
  float            _threshold;

};


#endif //VERIFICATOR_HPP