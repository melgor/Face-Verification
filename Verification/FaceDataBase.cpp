/* 
* @Author: melgor
* @Date:   2015-04-09 14:53:35
* @Last Modified 2015-06-25
* @Last Modified time: 2015-06-25 11:07:45
*/

#include <sstream>
#include <fstream>
#include <chrono>
#include <boost/algorithm/string/replace.hpp>
#include "FaceDataBase.hpp"
#include "Distances.hpp"
#include "Utils/Serialization.hpp"

using namespace std;
using namespace cv;

FaceDataBase::FaceDataBase(struct Configuration& config)
{
  _config             = config;
  _threshold          = config.threshold;
  _verificator        = new Verificator(config);

  //load train data
  _dataFeatures       = new Features;
  load( *_dataFeatures, config.faceData);
  Mat scaled_features;
  for(int row = 0; row < _dataFeatures->data.rows; row++)
  {
    Mat scaled_data;
    _verificator->scaleData(_dataFeatures->data.row(row), scaled_data );
    scaled_features.push_back(scaled_data);
  }
  _dataFeatures->data = scaled_features.clone();
  //load labels name
  ifstream infile(config.faceLabels);
  for( string line; getline( infile, line ); )
  {
    _labelsNames.push_back(line);
  }

}

string 
FaceDataBase::returnClosestIDName(cv::Mat& feature)
{
  int label = returnClosestID(feature);
  if(label < 0)
    return _unknown;
  return _labelsNames[label];
}

//return id >= 0 if find in database. Otherwise, not
int
FaceDataBase::returnClosestID(Mat& feature)
{
  string name;
  float score;
  int id;
  returnClosestIDNameScore(feature, id, name, score);

  return id;
}

void 
FaceDataBase::returnClosestIDNameScore( 
                                        Mat& feature
                                      , int& id
                                      , string& name
                                      , float& score
                                      )
{
  #ifdef __MSTIME
  auto t12 = std::chrono::high_resolution_clock::now();
  #endif
  vector<pair<float,int>> result_prob;
  Mat scaled_feature;
  _verificator-> scaleData(feature,scaled_feature);
  for(int row = 0; row < _dataFeatures->data.rows; row++)
  {
    float prob_value =_verificator->predict(scaled_feature, _dataFeatures->data.row(row));
    result_prob.push_back(make_pair(prob_value, _dataFeatures->labels[row]));
  }

  auto result = std::max_element(result_prob.begin(), result_prob.end()
                                    , []( const pair<float,int>& f1
                                        , const pair<float,int>& f2 )
                                    {
                                      return f1.first < f2.first;
                                    });
  #ifdef __MSTIME
  auto t22 = std::chrono::high_resolution_clock::now();
  std::cout  << "FaceDataBase took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";
   #endif
  LOG(WARNING)<<"Max prob: "<< (*result).first << " label: "<< (*result).second ;//<< " name:  "<< _labelsNames[(*result).second];
  if ((*result).first > _threshold)
  {
    score = (*result).first;
    name  = _labelsNames[(*result).second];
    id    = (*result).second;
  }
  else
  {
    score = -1;
    name  = _unknown;
    id    = -1;
  }

 
}

void 
FaceDataBase::addNewPerson( 
                            string& name
                          , Mat& feature
                          , Mat& image
                          )
{
  Mat scaled_data;
  _verificator->scaleData(feature, scaled_data );
  _dataFeatures->data.push_back(scaled_data);\
  _dataFeatures->labels.push_back( _dataFeatures->labels.back() + 1);
  //add name to library
  _labelsNames.push_back(name);

  //save original Image to the disk
  saveNewPersonImage(image);
  //backup
  backupData();
}

bool 
FaceDataBase::checkName(string& name)
{
  if ( std::find(_labelsNames.begin(), _labelsNames.end(), name)!=_labelsNames.end() )
  {  //this mean, that 'name' exist in database
     return true;
  } 
  return false;
}

void  
FaceDataBase::backupData()
{
 //TODO: think aboout backup, the issue:
// - data in DataBase are scaled to proper value. So before saving they should be to normal value or store 2 database at once (consume more memory)

  //Scale back the data and save data
  Mat scaled_features;
  for(int row = 0; row < _dataFeatures->data.rows; row++)
  {
    Mat scaled_data;
    _verificator->getValueBeforeScaling(_dataFeatures->data.row(row), scaled_data );
    scaled_features.push_back(scaled_data);
  }

  Features* dataFeatures = new Features;
  dataFeatures->data     = scaled_features.clone();
  dataFeatures->labels   = _dataFeatures->labels;
  compress(*dataFeatures, _config.faceData);
  delete dataFeatures;

  //save named to file
  ofstream face_labels (_config.faceLabels);
  if (face_labels.is_open())
  {
    for(auto& name : _labelsNames)
      face_labels << name << "\n";
    face_labels.close();
  }


}

void 
FaceDataBase::saveNewPersonImage(cv::Mat& image)
{
  //save Image
  std::string name = boost::replace_all_copy(_labelsNames.back(), " ", "_");
  string path_to_save = _config.faceFolder + name + "_" + to_string(_dataFeatures->labels.back()) + _extension;
  cv::imwrite(path_to_save, image);

  //Add image to current person list, associated with ID
  fstream fs;
  fs.open (_config.faceImages, std::fstream::in | std::fstream::out | std::fstream::app);
  fs <<path_to_save << " "<< to_string(_dataFeatures->labels.back()) << "\n";
  fs.close();
}



FaceDataBase::~FaceDataBase()
{ 
  backupData();
  delete _verificator;
  if (_dataFeatures != NULL)
      delete _dataFeatures;
}