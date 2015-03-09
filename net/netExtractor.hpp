#ifndef NETEXTRACTOR_HPP
#define NETEXTRACTOR_HPP
#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "utils/Utils.hpp"

//interface for Caffe Net

class NetExtractor
{
  public:
    NetExtractor(struct Configuration& config);
    //extract features using Net
    void extractFeatures(std::vector<cv::Mat>& images, cv::Mat& features);
    ~NetExtractor();

  private:
    //reshape Net to extract as many images as can: performance reason
    void reshape(int size);
    void extract(cv::Mat& feature);
    std::string                                       _protoPath;
    std::string                                       _caffeModelPath;
    bool                                              _gpu;
    int                                               _gpuID;
    std::string                                       _featFromLayer;
    caffe::Net<float>*                                _net;
    caffe::shared_ptr<caffe::MemoryDataLayer<float> > _memoryDataLayer;


  
};

#endif