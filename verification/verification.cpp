#include <iostream>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <algorithm>   
#include "verification.hpp"
#include "utils/serialization.hpp"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;


//Calculate cosine Similarity for two vectors
double cosineSimilarity(cv::Mat& vec1, cv::Mat& vec2)
{
  double dot_prodcut = vec1.dot(vec2);
  double sum_vec1 = cv::sum(vec1.mul(vec1))[0];
  double sum_vec2 = cv::sum(vec2.mul(vec2))[0];
  
  return dot_prodcut/sqrt(sum_vec1 * sum_vec2);
}


Verificator::Verificator(struct Configuration& config)
{
  _metric             = config.metric;
  _pathTrainFeatures  = config.trainData;
  _pathValFeatures    = config.valData;
  _trainFeatures      = new Features;
  load( *_trainFeatures, _pathTrainFeatures);
}

int
Verificator::verify(cv::Mat& features)
{
  vector<double> similarity(_trainFeatures->data.rows,0);
  for(int feature = 0; feature < _trainFeatures->data.rows; feature++)
  {
   cv::Mat tmp = _trainFeatures->data.row(feature);
   similarity[feature] = cosineSimilarity(tmp,features);
  }
  //get highest value
  auto result = std::max_element(similarity.begin(), similarity.end());
  //get label of feature which produce high value
  return _trainFeatures->labels[std::distance(similarity.begin(), result)];
}

void 
Verificator::verifyVal()
{
 Features val;
 if (_pathValFeatures == "")
 {  
    cerr<<"Path to validation not set. Use config.ini to set it";
    exit(0);
 }
 load(val,_pathValFeatures);
 int good_img = 0;
 int used_val = 10000;
 srand (time(NULL));
 for(int feature = 0; feature < used_val; feature++)
 { 
   cerr<<"Image: "<< feature<<endl;
   cv::Mat tmp = val.data.row(feature);
   int label = verify(tmp);
   if (label == val.labels[feature])
     good_img++;
 }
 
 cout<<"Accuracy: "<< float(good_img)/used_val << endl;
}
void 
Verificator::verifyValPerson()
{
 Features val;
 if (_pathValFeatures == "")
 {  
    cerr<<"Path to validation not set. Use config.ini to set it";
    exit(0);
 }
 load(val,_pathValFeatures);
 float threshold = 0.5;
 int good_img_true = 0, good_img_false = 0, false_pos = 0;
 int positive_pair = 0, negative_pair = 0;
 int used_val = 100000;

 for(int feature = 0; feature < used_val; feature++)
 { 
//    cerr<<"Image: "<< feature<<endl;
   cv::Mat tmp = val.data.row(feature);
   int rand_num = rand() % used_val;
   cv::Mat rand = val.data.row(rand_num);
   double sim = cosineSimilarity(tmp,rand);
   int lab = 0;
   if (sim > threshold)
      lab = 1;
//    if (sim > threshold)
   int lab_ver = 0;
   if (val.labels[feature] == val.labels[rand_num])
      lab_ver = 1;
   if(lab_ver)
   {
//     cerr<<lab_ver <<" "<<lab<<" "<<sim<<endl;
    positive_pair++;
   }
   else
   {
    negative_pair++;
   }
   if ((lab_ver == lab) && lab_ver)
     good_img_true++;
   else if (lab_ver == lab) 
     good_img_false++;
   else
     false_pos++;
 }
 cout<<"Accuracy: true "<< float(good_img_true)/positive_pair<<" false "<< float(good_img_false)/negative_pair<<" False Pos: "<< float(false_pos)/used_val<< endl;
  
}
Verificator::~Verificator()
{
  delete _trainFeatures;
}