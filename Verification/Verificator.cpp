/*
* @Author: melgor
* @Date:   2015-04-09 09:05:37
* @Last Modified 2015-06-24
* @Last Modified time: 2015-06-24 09:26:44
*/
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include "Verificator.hpp"
#include "Distances.hpp"
#include "Utils/Serialization.hpp"

using namespace std;
using namespace cv;

void readData( string& path, vector<string>& names, vector<int>& labels)
{
  string line;
  ifstream myfile (path);
  vector<string> splitteds;
  while ( getline (myfile, line) )
  {
    boost::split(splitteds, line, boost::is_any_of(" "));
    labels.push_back(std::stoi(splitteds[1]));
    names.push_back(splitteds[0]);
    splitteds.clear();
   
  }
  myfile.close();
}

Verificator::Verificator(struct Configuration& config)
{
  _pathTrainFeatures = config.trainData;
  _pathValFeatures   = config.valData;
  _threshold         = config.threshold;
  _metric            = config.metric;
  _coeffPath         = config.coeffPath;
  _biasPath          = config.biasPath;
  _scalerMinPath     = config.scalerMinPath;
  _scalerDiffPath    = config.scalerDiffPath;
  _ver1Path          = config.ver1Label;
  _ver2fPath         = config.ver2Label;
  _valLabelPath      = config.valLabel;
  if (_metric == "Cosine")
    distanceFunction = &cosineDistance;
  else if(_metric == "Chi")
    distanceFunction = &chiSquaredDistance;

  loadModel();

}

void
Verificator::loadModel()
{
  load(_coeffSK, _coeffPath);
  load(_biasSK, _biasPath);
  load(_skalerMinSK, _scalerMinPath);
  load(_skalerDiffSK, _scalerDiffPath);

  cerr<<"Model Loaded"<<endl;
}


float 
Verificator::predictFull(
                       Mat featureOne
                     , Mat featureTwo
                    )
{
  Mat scaled_data_1, scaled_data_2;
  scaleData(featureOne, scaled_data_1 );
  scaleData(featureTwo, scaled_data_2 );
  
  Mat feat;
  (*distanceFunction)(scaled_data_1,scaled_data_2,feat);
  //apply learned coefficient
  float prob_class = _coeffSK.dot(feat) + _biasSK.at<double>(0,0);
  return prob_class;
}

float 
Verificator::predict(
                       Mat featureOne
                     , Mat featureTwo
                    )
{
   Mat feat;
  (*distanceFunction)(featureOne,featureTwo,feat);
  //apply learned coefficient
  float prob_class = _coeffSK.dot(feat) + _biasSK.at<double>(0,0);
  return prob_class;
}

//compare if two features descrive same person or not
int
Verificator::compare(
                      Mat& featureOne
                    , Mat& featureTwo
                    )
{
  //predict value or apply threshold
  float score = predict(featureOne, featureTwo);
  if (score > _threshold)
    return 1;
  return 0;
}

//scale data
void 
Verificator::scaleData(
                        Mat  features
                      , Mat& scaledFeatures
                      )
{
  scaledFeatures =  ((features - _skalerMinSK).mul(1.0/_skalerDiffSK));
}

void 
Verificator::verify()
{ 
  readVerificationData();
  evalVerification();
}

void 
Verificator::evalVerification()
{
  
  Features* train_Features      = new Features;
  load( *train_Features, _pathValFeatures);
  //scale all features
  Mat scaled_features;
  for(int row = 0; row < train_Features->data.rows; row++)
  {
    Mat scaled_data;
    scaleData(train_Features->data.row(row), scaled_data );
    scaled_features.push_back(scaled_data);
  }

  int good = 0;
  for(uint i = 0; i < _idxVer1.size(); i++)
  {
    float score = predict(scaled_features.row(_idxVer1[i]), scaled_features.row(_idxVer2[i]));
    int label = 0;
    if(score > _threshold)
      label = 1;
    
    if( label == _idxLabels[i])
      good++;

  }
  
  cerr<<"ACC: "<< (float(good)/_idxVer1.size()) << endl;
  delete train_Features;
}

void 
Verificator::readVerificationData()
{
  vector<string> names_ver_1, names_ver_2, names_val;
  vector<int>    labels_ver_1, labels_ver_2, labels_val;
  
  readData(_ver1Path, names_ver_1, labels_ver_1);
  readData(_ver2fPath, names_ver_2, labels_ver_2);
  readData(_valLabelPath, names_val, labels_val);
  cerr<<"Sizes: "<< labels_ver_1.size() << " "<< labels_ver_2.size() <<" " << labels_val.size() << endl;
  
  //find index of images from feature vector
  _idxVer1.reserve( names_ver_1.size());
  _idxVer2.reserve( names_ver_1.size());
  for(uint i = 0; i < names_ver_1.size(); i++)
  {
    string ver_1 = names_ver_1[i];
    string ver_2 = names_ver_2[i];
    int idx_ver_1 = std::distance(names_val.begin(), std::find(names_val.begin(), names_val.end(), ver_1));
    int idx_ver_2 = std::distance(names_val.begin(), std::find(names_val.begin(), names_val.end(), ver_2));
    _idxVer1.push_back(idx_ver_1);
    _idxVer2.push_back(idx_ver_2);
    if( labels_ver_1[i] == labels_ver_2[i])
    {
      _idxLabels.push_back(1);
    }
    else
    {
      _idxLabels.push_back(0);
    }
    if (i%1000  == 0)
      cerr<<"Set " << i << endl;
  }
}



Verificator::~Verificator()
{
  if (_trainFeatures != NULL)
      delete _trainFeatures;

}

