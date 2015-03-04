/* 
* @Author: blcv
* @Date:   2015-03-03 15:35:29
* @Last Modified 2015-03-03
* @Last Modified time: 2015-03-03 16:47:30
*/
#include <iostream>
#include <fstream>
#include <string>
#include "fetureExtractor.hpp"
#include "utils/serialization.hpp"
#include <boost/algorithm/string.hpp>

using namespace std;

FetureExtractor::FetureExtractor(struct Configuration& config)
{
  _netExtractor = new NetExtractor(config);
  _folder       = config.extractorFolder;
  _imageList    = config.extractorImageList;
}

void 
FetureExtractor::extractAllFeatures()
{
  string line;
  ifstream myfile (_imageList);
  vector<string> image_path;
  image_path.reserve(_batchSize);
  cv::Mat features;
  uint num_images = 0;
  vector<int> labels;
  if (myfile.is_open())
  {
    vector<string> splitteds;
    while ( getline (myfile,line) )
    {
      cout<<"Num Image: "<<num_images<<endl;
      boost::split(splitteds, line, boost::is_any_of(" "));
      image_path.push_back(splitteds[0]);
      labels.push_back(std::stoi(splitteds[1]));
      splitteds.clear();
      if(image_path.size() == _batchSize)
      {
        //extract featues from that package
        extractFromMat(image_path, features);

        image_path.clear();
        image_path.reserve(_batchSize);
        _numBatch++;
      }
      num_images++;
    }
    myfile.close();
  }
  else 
  {
    std::cerr << "Unable to open file"; 
  }
  
  //serialize feature
  
  Features feat = {features,labels};
  string name_file = _folder + to_string(_numBatch) + ".bin";
  cout<<"Save: "<< name_file<<endl;
  compress(feat,name_file);
}

void 
FetureExtractor::extractFromMat(
                                  vector<string>& imageList
                                , cv::Mat& features
                                )
{
  vector<cv::Mat> image_data(imageList.size(),cv::Mat());
  int num_img = 0;
  for(auto& path : imageList)
  {
    cv::Mat img = cv::imread(path);
    cv::resize(img, image_data[num_img], cv::Size(45,45));
    num_img++;
  }
  _netExtractor->extractFeatures(image_data, features);;
  
}                                

FetureExtractor::~FetureExtractor()
{
  delete _netExtractor;
}