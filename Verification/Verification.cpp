#include <iostream>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <algorithm>   
#include "Verification.hpp"
#include "Utils/Serialization.hpp"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

using namespace std;
using namespace cv;

//Calculate cosine Similarity for two vectors
double cosineSimilarity(Mat& vec1, Mat& vec2)
{
  double dot_prodcut = vec1.dot(vec2);
  double sum_vec1 = sum(vec1.mul(vec1))[0];
  double sum_vec2 = sum(vec2.mul(vec2))[0];
  
  return dot_prodcut/sqrt(sum_vec1 * sum_vec2);
}

void chiSquaredDistance(
                          Mat  f1
                        , Mat  f2
                        , Mat& featChi
                        )
{
  //dist = (f1 + f2)^2 elementwise
  Mat distance = f1 - f2;
  distance = distance.mul(distance);
  //out = dist/(f1 + f2) elementwise
  Mat sum_f = f1 + f2;
  MatIterator_<float> it_dst = sum_f.begin<float>(), it_end_dst = sum_f.end<float>();
  for(MatIterator_<float> j = it_dst; j != it_end_dst ;++j)
  {
    if (*j == 0.0f)
    {
      *j = 1.0f;
    }
  }
  featChi = distance.mul(1.0/(sum_f)); 
  // std::cerr<<featChi<<std::endl;

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
Verificator::verify(Mat& features)
{
  vector<double> similarity(_trainFeatures->data.rows,0);
  for(int feature = 0; feature < _trainFeatures->data.rows; feature++)
  {
   Mat tmp = _trainFeatures->data.row(feature);
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
   Mat tmp = val.data.row(feature);
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
 float threshold = 0.3;
 int good_img_true = 0, good_img_false = 0, false_pos = 0;
 int positive_pair = 0, negative_pair = 0;
 int used_val = 7000;

 for(int feature = 0; feature < used_val; feature++)
 { 
//    cerr<<"Image: "<< feature<<endl;
   Mat tmp = val.data.row(feature);
   int rand_num = rand() % used_val;
   Mat rand = val.data.row(rand_num);
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
 cout<<"Accuracy: true "<< float(good_img_true)/positive_pair<<" false "<< float(good_img_false)/negative_pair <<" Overall "<< float(good_img_false + good_img_true)/(negative_pair +positive_pair) <<" False Pos: "<< float(false_pos)/used_val<< endl;
  
}
Verificator::~Verificator()
{
  delete _trainFeatures;
}