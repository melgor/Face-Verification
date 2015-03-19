/* 
* @Author: blcv
* @Date:   2015-03-18 13:14:12
* @Last Modified 2015-03-19
* @Last Modified time: 2015-03-19 11:55:41
*/
#include "ChiDistance.hpp"
#include "Utils/Serialization.hpp"

using namespace std;
using namespace cv;

ChiDistance::ChiDistance(Configuration& config)
{
  _pathTrainFeatures  = config.trainData;
  _pathValFeatures    = config.valData;
  _threshold          = config.thresholdChi;
  _pathComparator     = config.pathComparatorChi;
  _comparator         = new CvSVM();
} 

int 
ChiDistance::compare( 
                      Mat& featureOne
                    , Mat& featureTwo
                    )
{
  Mat chi_data;
  transformData(featureOne,featureTwo,chi_data);
  float res = _comparator->predict(chi_data); 
  cerr<<"Predicted: "<<res<<endl;
  if (res > _threshold)
    return 1;
  return 0;
}

void
ChiDistance::train()
{
  //load train data
  _trainFeatures      = new Features;
  load( *_trainFeatures, _pathTrainFeatures);
  //set SVM parameters
  CvSVMParams params;
  params.svm_type    = CvSVM::C_SVC;
  params.kernel_type = CvSVM::LINEAR;
  params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 10000, 1e-6);
  Mat features,labels;
  prepateTrainData(features, labels);
  cerr<<features.size()<<" "<<labels.size()<<endl;
  _comparator->train(features, labels, Mat(), Mat(), params);
  _comparator->save(_pathComparator.c_str());

}

void 
ChiDistance::verifyVal()
{
  _comparator->load(_pathComparator.c_str());
  Features val_features;
  load(val_features,_pathValFeatures);
  int good_img_true = 0, good_img_false = 0, false_pos = 0;
  int positive_pair = 0, negative_pair = 0;
  for(uint feature = 0; feature < val_features.labels.size(); feature++)
  {
    Mat tmp_1 = val_features.data.row(feature);
    int rand_num = rand() % val_features.labels.size();
    Mat tmp_2 = val_features.data.row(rand_num);
    bool res = compare(tmp_1,tmp_2);
    int lab_ver = 0;
    if (val_features.labels[feature] == val_features.labels[rand_num])
      lab_ver = 1;
    
     if(lab_ver)
     {
      positive_pair++;
     }
     else
     {
      negative_pair++;
     }
     if ((lab_ver == res) && lab_ver)
       good_img_true++;
     else if (lab_ver == res) 
       good_img_false++;
     else
       false_pos++;
  }

  cout<<"Accuracy: true "<< float(good_img_true)/positive_pair<<" false "<< float(good_img_false)/negative_pair <<" Overall "<< float(good_img_false + good_img_true)/(negative_pair +positive_pair) <<" False Pos: "<< float(false_pos)/val_features.labels.size()<< endl;
}

void 
ChiDistance::prepateTrainData( 
                                Mat& features
                              , Mat& labels
                              )
{
  int num_example =  _trainFeatures->labels.size();
  vector<int> labels_vec(num_example,0);
  RNG rng;
  for(int i = 0; i < num_example; i++)
  {
     //get two random example and create vector for Chi distance
    int feat_1 = rng.uniform(int(0), num_example -1);
    int feat_2 = rng.uniform(int(0), num_example -1);
    Mat feat;

    transformData(_trainFeatures->data.row(feat_1),_trainFeatures->data.row(feat_2),feat);
    features.push_back(feat);
    if (_trainFeatures->labels[feat_1] == _trainFeatures->labels[feat_2])
      labels_vec[i] = 1;

  }

  labels = Mat(labels_vec);

}

void 
ChiDistance::transformData(
                            Mat f1
                          , Mat f2
                          , Mat& featChi
                          )
{ 
  //dist = (f1 + f2)^2 elementwise
  Mat distance = f1 - f2;
  distance = distance.mul(distance);
  //out = dist/(f1 + f2) elementwise
  featChi = distance.mul((f1 + f2)); 

}

ChiDistance::~ChiDistance()
{
  
  delete _comparator;
  if (_trainFeatures != NULL)
      delete _trainFeatures;
}