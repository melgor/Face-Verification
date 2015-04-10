/* 
* @Author: melgor
* @Date:   2015-04-09 14:53:35
* @Last Modified 2015-04-10
* @Last Modified time: 2015-04-10 09:56:46
*/

#include "FaceDataBase.hpp"
#include "Distances.hpp"
#include "Utils/Serialization.hpp"
using namespace std;
using namespace cv;

FaceDataBase::FaceDataBase(struct Configuration& config)
{
  _threshold          = config.threshold;
  _pathComparator     = config.pathComparator;
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
  _comparatorLinear->loadModel(_pathComparator);
}

//return id >= 0 if find in database. Otherwise, not
int
FaceDataBase::returnClosestID(Mat& feature)
{
  vector<pair<float,int>> result_prob;
  Mat scaled_feature;
  scaleData(feature,scaled_feature);
  for(int row = 0; row < _dataFeatures->data.rows; row++)
  {
    Mat data_feature = _dataFeatures->data.row(row);
    float prob_value = compare(scaled_feature,data_feature);
    result_prob.push_back(make_pair(prob_value, _dataFeatures->labels[row]));
  }

  auto result = std::max_element(result_prob.begin(), result_prob.end()
                                    , []( const pair<float,int>& f1
                                        , const pair<float,int>& f2 )
                                    {
                                      return f1.first > f2.first;
                                    });

  if ((*result).first > _threshold)
    return (*result).second;

  return -1;
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