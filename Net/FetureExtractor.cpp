/* 
* @Author: blcv
* @Date:   2015-03-03 15:35:29
* @Last Modified 2015-04-08
* @Last Modified time: 2015-04-08 09:39:17
*/
#include <iostream>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include "FetureExtractor.hpp"
#include "Utils/Serialization.hpp"


using namespace std;
using namespace cv;

FetureExtractor::FetureExtractor(struct Configuration& config)
{
  _netExtractor  = new NetExtractor(config);
  _folder        = config.extractorFolder;
  _imageList     = config.extractorImageList;
  _scaleFeatures = config.scaleFeature;
}

void 
FetureExtractor::extractAllFeatures()
{
  string line;
  ifstream myfile (_imageList);
  vector<string> image_path;
  image_path.reserve(_batchSize);
  Mat features;
  uint num_images = 0;
  vector<int> labels;
  vector<string> names;
  if (myfile.is_open())
  {
    vector<string> splitteds;
    while ( getline (myfile,line) )
    {
      //cout<<"Num Image: "<<num_images<<endl;
      boost::split(splitteds, line, boost::is_any_of(" "));
      image_path.push_back(splitteds[0]);
      labels.push_back(std::stoi(splitteds[1]));
      names.push_back(splitteds[0]);
      splitteds.clear();
      if(image_path.size() == _batchSize)
      {
        cout<<"Num Image: "<<image_path.size()<<endl;
        //extract featues from that package
        extractFromMat(image_path, features);

        image_path.clear();
        image_path.reserve(_batchSize);
        _numBatch++;
      }
      num_images++;
    }
    myfile.close();
    //extract data from last package, smaller than _batchSize
    if (image_path.size())
    {
      cout<<"Num Image: "<<image_path.size()<<endl;
      extractFromMat(image_path, features);
      _numBatch++;
    }

  }
  else 
  {
    std::cerr << "Unable to open file"; 
  }
  
  //scale features
  if(_scaleFeatures)
    scaleData(features);
  //serialize feature
  Features feat = {features,labels,names};
  string name_file = _folder + to_string(_numBatch) + ".bin";
  cout<<"Save: "<< name_file<<endl;
  compress(feat,name_file);
}

void 
FetureExtractor::extractFromMat(
                                  vector<string>& imageList
                                , Mat& features
                                )
{
  vector<Mat> image_data(imageList.size(),cv::Mat());
  int num_img = 0;
  for(auto& path : imageList)
  {
    cerr<<path<<endl;
    cv::Mat img = cv::imread(path);
    cv::resize(img, image_data[num_img], cv::Size(47,55));
    num_img++;
  }
  _netExtractor->extractFeatures(image_data, features);;
  
}            

void 
FetureExtractor::scaleData( Mat& features)   
{
  _statisticFeatures = Mat(Size(features.cols,1),CV_32FC1);
  //For each column find maxiumum value (minimum is 0 because of ReLU)
  double min, max;
  for(int cols = 0; cols < features.cols; cols++)
  {
    Mat col = features.col(cols);
    cv::minMaxLoc(col, &min, &max);
    _statisticFeatures.at<float>(0,cols) = float(max);
    if (float(max) != float(0.0))
      col /= float(max);
  }

}

FetureExtractor::~FetureExtractor()
{
  delete _netExtractor;
}