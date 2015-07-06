/* 
* @Author: melgor
* @Date:   2015-04-09 14:53:35
* @Last Modified 2015-06-25
* @Last Modified time: 2015-06-25 11:07:45
*/

#include "FaceDataBase.hpp"
#include "Distances.hpp"
#include "Utils/Serialization.hpp"
#include <sstream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;

FaceDataBase::FaceDataBase(struct Configuration& config)
{
  _pathFaceData       = config.faceData;
  _threshold          = config.threshold;
  _verificator        = new Verificator(config);

  //load train data
  _dataFeatures       = new Features;
  load( *_dataFeatures, _pathFaceData);
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
  returnClosestIDNameScore(feature, name, score);

  return score;
}

void 
FaceDataBase::returnClosestIDNameScore( 
                                        Mat& feature
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
//     if(prob_value > 0.0)
//       cerr<<"Prob: "<< prob_value <<" Label: "<< _dataFeatures->labels[row] << endl;
    if( row == 1208)
      cerr<<"Prob: "<< prob_value <<" Label: "<< _dataFeatures->labels[row] << endl;
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
  LOG(WARNING)<<"Max prob: "<< (*result).first << " label: "<< (*result).second << " name:  "<< _labelsNames[(*result).second];
  if ((*result).first > _threshold)
  {
    score = (*result).second;
    name = _labelsNames[(*result).second];
  }
  else
  {
    score = -1;
    name  = _unknown;
  }

 
}

FaceDataBase::~FaceDataBase()
{
  delete _verificator;
  if (_dataFeatures != NULL)
      delete _dataFeatures;
}