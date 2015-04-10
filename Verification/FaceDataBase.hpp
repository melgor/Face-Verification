#ifndef FACEDATABASE_HPP
#define FACEDATABASE_HPP
//store information about learned People. 
//For given representation it find closest people or say "Not Found"
#include "Utils/Utils.hpp"
#include "SVM.hpp"

class FaceDataBase
{
public:
  FaceDataBase(struct Configuration& config);
  int returnClosestID(cv::Mat& feature);
  ~FaceDataBase();

private:
  //compare if two features descrive same person or not
  float compare(cv::Mat& featureOne, cv::Mat& featureTwo);
  //scale data
  void scaleData(cv::Mat features, cv::Mat& scaledFeatures);
  //distance function
  void (*distanceFunction) (cv::Mat, cv::Mat, cv::Mat&) = NULL;

  struct Features*         _dataFeatures = NULL;
  std::vector<std::string> _labelsNames;
  //configuration
  std::string      _metric;
  std::string      _pathFaceData;
  //data for scaling
  cv::Mat          _maxValue;
  std::string      _pathScaler;
  //learning algorithm
  SVMLinear*       _comparatorLinear = NULL;
  std::string      _pathComparator;
  std::string      _pathComparatorMat;
  float            _threshold;
};

#endif