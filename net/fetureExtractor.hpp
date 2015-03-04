#ifndef FEATUREEXTRACTOR_HPP
#define FEATUREEXTRACTOR_HPP
#include "netExtractor.hpp"
//Feature extraction for Images. Used to build database for similarity search

class FetureExtractor
{
public:
  FetureExtractor(struct Configuration& config);
  void extractAllFeatures();
  ~FetureExtractor();

private:
  void extractFromMat(std::vector<std::string>& imageList, cv::Mat& features);
  NetExtractor*  _netExtractor;
  std::string    _folder;
  std::string    _imageList;
  uint           _batchSize = 374;
  uint           _numBatch = 0;
};

#endif