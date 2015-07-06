/* 
* @Author: blcv
* @Date:   2015-03-03 10:56:35
* @Last Modified 2015-06-18
* @Last Modified time: 2015-06-18 20:24:36
*/

#include "NetExtractor.hpp"

using namespace caffe;
using namespace std;

NetExtractor::NetExtractor(struct Configuration& config)
{

  _protoPath       = config.prototxt;
  _caffeModelPath  = config.caffemodel;
  _gpu             = config.gpu;
  _gpuID           = config.gpuID;
  _featFromLayer   = config.layer;
  
  //init Caffe Net
  if (_gpu)
  {
    Caffe::SetDevice(_gpuID);
    Caffe::set_mode(Caffe::GPU);
  }
  else
  {
    Caffe::set_mode(Caffe::CPU);
  } 

  _net = new Net<float>(_protoPath,caffe::TEST);
  _net->CopyTrainedLayersFrom(_caffeModelPath);
  const boost::shared_ptr<Blob<float> >& dataLayer = _net->blob_by_name("data");
  dataLayer->Reshape(1, 3, 100, 100);
  _net->Reshape();
  
  _memoryDataLayer = boost::static_pointer_cast<MemoryDataLayer<float> >(_net->layer_by_name("data"));
}

void 
NetExtractor::extractFeatures(
                    vector<cv::Mat>& images
                  , cv::Mat& features
                  )
{
  //need label vector, can be random value
  vector<int> labelVector(images.size(),0);
  //add images to bottom of Net
  _memoryDataLayer->AddMatVector(images,labelVector);

  //run prediction
  // TODO:now it run only one batch==1 image. It should be changed using Reshape 

  for(uint i = 0; i < images.size();i++)
  {
    cv::Mat feat;
    extract(feat);
    features.push_back(feat);
  }
  // cerr<<features<<endl;

}

void 
NetExtractor::extract(
                      cv::Mat& feature
                     )
{
  float loss = 0.0;
  vector<Blob<float>*> results = _net->ForwardPrefilled(&loss);
  const boost::shared_ptr<Blob<float> >& feat_Layer = _net->blob_by_name(_featFromLayer);
  int lenght_feat = feat_Layer->channels();
  float* feat_out = feat_Layer->mutable_cpu_data();
  
  //as batch size in deploy ==1, function will only extract one image
  feature =  cv::Mat(1, lenght_feat, CV_32FC1, feat_out);

}
void 
NetExtractor::reshape(int size)
{

}

NetExtractor::~NetExtractor()
{
  delete _net;
}