#ifndef FEATUREEXTRACTOR_HPP
#define FEATUREEXTRACTOR_HPP
#include "NetExtractor.hpp"
//Feature extraction for Images. Used to build database for similarity search

class FetureExtractor
{
public:
  FetureExtractor(struct Configuration& config);
  void extractAllFeatures();
  ~FetureExtractor();

private:
  void extractFromMat(std::vector<std::string>& imageList, cv::Mat& features);
  void scaleData(cv::Mat& features);
  //extractor of data
  NetExtractor*  _netExtractor;
  //data to normalize features
  cv::Mat        _statisticFeatures;
  std::string    _folder;
  std::string    _imageList;
  bool           _scaleFeatures;
  uint           _batchSize = 64;
  uint           _numBatch = 0;
};

#endif