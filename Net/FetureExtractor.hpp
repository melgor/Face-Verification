#ifndef FEATUREEXTRACTOR_HPP
#define FEATUREEXTRACTOR_HPP
#include "NetExtractor.hpp"
//Feature extraction for Images. Used to build database for similarity search

class FetureExtractor
{
public:
  FetureExtractor(struct Configuration& config);
  void extractAllFeatures();
  void extractFeature(cv::Mat& image, cv::Mat& feature);
  ~FetureExtractor();

private:
  void extractFromMat(std::vector<std::string>& imageList, cv::Mat& features);
  //extractor of data
  NetExtractor*  _netExtractor;
  std::string    _folder;
  std::string    _imageList;
  uint           _batchSize = 64;
  uint           _numBatch = 0;
};

#endif