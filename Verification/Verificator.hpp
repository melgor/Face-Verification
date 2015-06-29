#ifndef VERIFICATOR_HPP
#define VERIFICATOR_HPP
#include "Utils/Utils.hpp"

//class for Verification task. Get new feature, then classify it based on DataBase

class Verificator
{
public:
  Verificator(struct Configuration& config);
  
  //load verification data
  void verify();
  //compare if two features descrive same person or not
  int compare(cv::Mat& featureOne, cv::Mat& featureTwo);
  float predict(cv::Mat featureOne, cv::Mat featureTwo);
  float predictFull(cv::Mat featureOne, cv::Mat featureTwo);              
  //scale data
  void scaleData(cv::Mat features, cv::Mat& scaledFeatures);
  ~Verificator();

private:
  //load Model from Sklearn  for Face Verification
  void loadModel();
  //transform data from classification task to verfication
  void readVerificationData();
  void evalVerification();

  void (*distanceFunction) (cv::Mat, cv::Mat, cv::Mat&) = NULL;
  //configuration
  std::string      _metric;
  std::string      _pathTrainFeatures;
  std::string      _pathValFeatures;
  std::string      _coeffPath;
  std::string      _biasPath;
  std::string      _scalerMinPath;
  std::string      _scalerDiffPath;
  std::string      _ver1Path;
  std::string      _ver2fPath;
  std::string      _valLabelPath;
  struct Features* _trainFeatures = NULL;
  //learning algorithm
  float            _threshold;
  //Sklearn model
  cv::Mat          _coeffSK;
  cv::Mat          _biasSK;
  cv::Mat          _skalerMinSK;
  cv::Mat          _skalerDiffSK;
  //Verification data
  std::vector<int> _idxVer1;
  std::vector<int> _idxVer2;
  std::vector<int> _idxLabels;
};


#endif //VERIFICATOR_HPP