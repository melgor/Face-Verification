/* 
* @Author: melgor
* @Date:   2015-04-09 14:53:35
* @Last Modified 2015-04-27
* @Last Modified time: 2015-04-27 15:22:49
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
  _threshold          = config.threshold;
  _pathComparator     = config.pathComparator;
  _pathComparatorMat  = config.pathComparatorMat;
  _metric             = config.metric;
  _pathScaler         = config.pathScaler;
  _pathFaceData       = config.faceData;

  if (_metric == "Cosine")
    distanceFunction = &cosineDistance;
  else if(_metric == "Chi")
    distanceFunction = &chiSquaredDistance;
  _comparatorLinear   = new SVMLinear(config);

  //load train data
  _dataFeatures      = new Features;
  load( *_dataFeatures, _pathFaceData);
  //load scaler value
  load(_maxValue, _pathScaler);
  //load learned model
  _comparatorLinear->loadModel(_pathComparator,_pathComparatorMat);
  //load labels name
  ifstream infile(config.faceLabels);
  for( string line; getline( infile, line ); )
  {
    _labelsNames.push_back(line);
  }
  _unknown = "Unknown";

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
  returnClosestIDNameScore(feature,name,score);

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
  Mat scaled_feature, scaled_feature2;
  scaleData(feature,scaled_feature);
  for(int row = 0; row < _dataFeatures->data.rows; row++)
  {

    Mat data_feature = _dataFeatures->data.row(row);
    scaleData(data_feature,scaled_feature2);
    float prob_value = compare(scaled_feature,scaled_feature2);
    result_prob.push_back(make_pair(prob_value, _dataFeatures->labels[row]));
    // if(prob_value > 0.0)
    //   cerr<<"Prob: "<< prob_value <<" Label: "<< _dataFeatures->labels[row] << endl;
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
  LOG(WARNING)<<"Max prob: "<< (*result).first;
  if ((*result).first > _threshold)
  {
    score = (*result).first;
    name = _labelsNames[(*result).second];
  }
  else
  {
    score = -1;
    name  = _unknown;
  }

 
}


void
FaceDataBase::scaleData(
                        Mat features
                      , Mat& scaledFeatures
                      )
{
  scaledFeatures =  features.mul(1.0/(_maxValue));
}

float
FaceDataBase::compare(
                      Mat& featureOne
                    , Mat& featureTwo
                    )
{
  //compute feature representation
  Mat feat;
  (*distanceFunction)(featureOne,featureTwo,feat);

  //predict value or apply threshold
  return _comparatorLinear->predict_prob(feat);
}

FaceDataBase::~FaceDataBase()
{
  delete _comparatorLinear;
  if (_dataFeatures != NULL)
      delete _dataFeatures;
}