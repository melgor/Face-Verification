/*
* @Author: melgor
* @Date:   2015-04-09 09:05:37
* @Last Modified 2015-04-10
* @Last Modified time: 2015-04-10 12:52:02
*/

#include "Verificator.hpp"
#include "Distances.hpp"
#include "Utils/Serialization.hpp"

using namespace std;
using namespace cv;

Verificator::Verificator(struct Configuration& config)
{
  _pathTrainFeatures  = config.trainData;
  _pathValFeatures    = config.valData;
  _threshold          = config.threshold;
  _pathComparator     = config.pathComparator;
  _pathComparatorMat  = config.pathComparatorMat;
  _metric             = config.metric;
  _pathScaler         = config.pathScaler;
  if (_metric == "Cosine")
    distanceFunction = &cosineDistance;
  else if(_metric == "Chi")
    distanceFunction = &chiSquaredDistance;
  _comparatorLinear   = new SVMLinear(config);
}

void
Verificator::train()
{
  //load train data
  _trainFeatures      = new Features;
  load( *_trainFeatures, _pathTrainFeatures);
  cerr<<"Scale Data"<<endl;
  learnScaleParam(_trainFeatures->data);
  //scale all features
  Mat scale_data_all;
  for(int row = 0; row < _trainFeatures->data.rows; row++)
  {
    Mat scaled_data;
    scaleData(_trainFeatures->data.row(row), scaled_data );
    scale_data_all.push_back(scaled_data);
  }
  Mat features;
  vector<int> labelsVec;
  cerr<<"Create Verification Task"<<endl;
  prepareVerificationData(scale_data_all, features,labelsVec);
  cerr<<"Learn Model"<<endl;
  _comparatorLinear->learn(features,labelsVec);
  _comparatorLinear->saveModel(_pathComparator, _pathComparatorMat);
  compress(_maxValue, _pathScaler);

  cerr<<"Model Saved"<<endl;
}

void 
Verificator::verify()
{
  //load validation data
  _trainFeatures      = new Features;
  load( *_trainFeatures, _pathValFeatures);
  load(_maxValue, _pathScaler);
  _comparatorLinear->loadModel(_pathComparator, _pathComparatorMat);
  //scale all features
  Mat scale_data_all;
  for(int row = 0; row < _trainFeatures->data.rows; row++)
  {
    Mat scaled_data;
    scaleData(_trainFeatures->data.row(row), scaled_data );
    scale_data_all.push_back(scaled_data);
  }
  // Mat f1 = scale_data_all.row(0), f2 = scale_data_all.row(0);
  // cerr<<"Compare: "<<compare(f1,f2) << endl;
  Mat features;
  vector<int> labelsVec;
  cerr<<"Create Verification Task"<<endl;
  prepareVerificationData(scale_data_all, features,labelsVec);
  _comparatorLinear->loadData(features,labelsVec);
  _comparatorLinear->evalModel();

}

//compare if two features descrive same person or not
int
Verificator::compare(
                      Mat& featureOne
                    , Mat& featureTwo
                    )
{
  //compute feature representation
  Mat feat;
  (*distanceFunction)(featureOne,featureTwo,feat);
  
  //predict value or apply threshold
  return _comparatorLinear->predict_class(feat);
}

//scale data
void 
Verificator::scaleData(
                        Mat features
                      , Mat& scaledFeatures
                      )
{
  scaledFeatures =  features.mul(1.0/(_maxValue));
}

void 
Verificator::prepareVerificationData(
                              Mat& scaledFetures
                            , Mat& featuresVer
                            , vector<int>& labelsVecVer
                            )
{
  int num_example =  _trainFeatures->labels.size();
  labelsVecVer.resize(2 * num_example,0);
  RNG rng;
  int feat_1 = 0, feat_2 = 0;
  //produce only positive example
  for(int i = 0; i < 1.0* num_example; i++)
  {
    feat_1 = 0;
    feat_2 = 10;
    //cerr<<i<<" "<<(feat_1 == feat_2) << " "<< !feat_1<<endl;
    while(_trainFeatures->labels[feat_2] != _trainFeatures->labels[feat_1])
    {
      //get two random example and create vector for feature based on distance
      feat_1 = rng.uniform(int(0), num_example -1);
      feat_2 = rng.uniform(int(0), num_example -1);
       //cerr<<i<<" "<<(feat_1 == feat_2) << " "<< !feat_1<<endl;
    }

    // cerr<<"Labels: "<<_trainFeatures->labels[feat_1] <<" "<< _trainFeatures->labels[feat_2]<<" "  << feat_1 << "  "<<feat_2<<" names: "<<_trainFeatures->names[feat_1]<<" "<<_trainFeatures->names[feat_2] << endl;
    Mat feat;
    (*distanceFunction)(scaledFetures.row(feat_1),scaledFetures.row(feat_2),feat);
    featuresVer.push_back(feat);
    // if (_trainFeatures->labels[feat_1] == _trainFeatures->labels[feat_2])
    labelsVecVer[i] = 1;
  }

  //produce only negative example
  for(int i = 1.0* num_example; i < 2*num_example; i++)
  {
    feat_1 = 0;
    feat_2 = 0;
    while(_trainFeatures->labels[feat_2] == _trainFeatures->labels[feat_1])
    {
      //get two random example and create vector for Chi distance
      feat_1 = rng.uniform(int(0), num_example -1);
      feat_2 = rng.uniform(int(0), num_example -1);
    }

    Mat feat;
    (*distanceFunction)(scaledFetures.row(feat_1),scaledFetures.row(feat_2),feat);
    featuresVer.push_back(feat);

  }
}

//learn scale value from train data
void
Verificator::learnScaleParam(Mat& features)
{
  //find max value of each feature value
  _maxValue = Mat::zeros(Size(features.size().width,1),features.type());
  double min, max;
  for(int col = 0; col < features.cols; col++)
  {
    //get row of features from column
    Mat feat_col = features.col(col);
    // cerr<<"Size: "<< feat_col.size()  <<" Type: "<<feat_col.type() <<" "<<  col<< endl;
    minMaxLoc(feat_col, &min, &max);
     // cerr<<"min: "<< min  <<" max: "<<max <<" "<<  col<< endl;
    //TODO: set one type of Mat
    if(float(max) != 0.0f)
      _maxValue.at<float>(0,col) = float(max);
    else
      _maxValue.at<float>(0,col) = 1.0f;
  }
  cerr<<"Scale Mat: "<< _maxValue << endl;
}

Verificator::~Verificator()
{

  delete _comparatorLinear;
  if (_trainFeatures != NULL)
      delete _trainFeatures;

}

