/* 
* @Author: blcv
* @Date:   2015-03-03 15:35:29
* @Last Modified 2015-04-10
* @Last Modified time: 2015-04-10 15:55:07
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

  //serialize feature
  Features feat = {features,labels,names};
  string name_file = _folder + to_string(_numBatch) + ".bin";
  cout<<"Save: "<< name_file<<endl;
  compress(feat,name_file);
}

void
FetureExtractor::extractFeature(
                                  Mat& image
                                , Mat& feature
                               )
{
  vector<Mat> image_data(1,cv::Mat());
  cv::resize(image, image_data[0], cv::Size(100,100));
  _netExtractor->extractFeatures(image_data, feature);;
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
    cv::resize(img, image_data[num_img], cv::Size(100,100));
    num_img++;
  }
  _netExtractor->extractFeatures(image_data, features);;

}

FetureExtractor::~FetureExtractor()
{
  delete _netExtractor;
}