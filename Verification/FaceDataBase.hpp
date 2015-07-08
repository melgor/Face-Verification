#ifndef FACEDATABASE_HPP
#define FACEDATABASE_HPP
//store information about learned People. 
//For given representation it find closest people or say "Not Found"
#include "Utils/Utils.hpp"
#include "Verificator.hpp"

class FaceDataBase
{
public:
  FaceDataBase(){};
  FaceDataBase(struct Configuration& config);
  int         returnClosestID(cv::Mat& feature);
  std::string returnClosestIDName(cv::Mat& feature);
  void returnClosestIDNameScore(cv::Mat& feature, int& id, std::string& name, float& score);
  ~FaceDataBase();

private:
  //compare if two features descrive same person or not
  float compare(cv::Mat& featureOne, cv::Mat& featureTwo);
  struct Features*         _dataFeatures = NULL;
  std::vector<std::string> _labelsNames;
  std::string              _unknown = "Unknown";
  //configuration
  std::string             _pathFaceData;
  //learning algorithm
  Verificator*            _verificator;
  float                   _threshold;
};

#endif