/* 
* @Author: blcv
* @Date:   2015-03-19 13:08:33
* @Last Modified 2015-04-10
* @Last Modified time: 2015-04-10 14:54:18
*/

#include "SVM.hpp"
#include "Utils/Serialization.hpp"

using namespace std;
using namespace cv;
void print_null(const char *s) {};

SVMLinear::SVMLinear(Configuration& config)
{
  _data.x     = NULL;
  _data.y     = NULL;
   // bias
  _data.bias = 1.0;
  _classifier = NULL;
  _param.C    = 0;
}

void
SVMLinear::learn(
            Mat& features
          , vector<int>& labels
          )
{
  loadData(features,labels);
  setParams();
  setClassWeights(1.0,1.0);
  _classifier = train(&_data, &_param);
  evalModel();
  getDecisionFunction();
}

int
SVMLinear::predict_class(cv::Mat& features)
{
  // feature_node **x = (struct feature_node**)calloc(1, sizeof(struct feature_node*));
  //fill data
  // prepareData(features, x);
  // int num_clas = 2;
  // std::vector<float> prob_class(num_clas,0);
  // for(int cla = 0; cla < num_clas; num_clas++ )
  // {
  //   Mat weight_class = _classiferMat->weights.row(cla);
  //   prob_class[cla] = weight_class.dot(features) + _classiferMat->bias[cla];
  // }

  // auto result = std::max_element(prob_class.begin(), prob_class.end()
  //                                   , []( const float& f1
  //                                       , const float& f2 )
  //                                   {
  //                                     return f1 > f2;
  //                                   });

  // return predict(_classifier, x[0]);//,prob);

  float pred_sim = predict_prob(features);
  if(pred_sim > 0.5)
    return 1;
  return 0;
  // return std::distance(prob_class.begin(), result);
}

float 
SVMLinear::predict_prob(Mat& features)
{
  // feature_node **x = (struct feature_node**)calloc(1, sizeof(struct feature_node*));
  // //fill data
  // prepareData(features, x);
  // double prob[2];
  // int lable = predict_probability(_classifier, x[0] ,&prob[0]);
  // cerr<<"prob: "<< prob[0] << " "<< prob[1] << " "<< x[0] << " "<< features <<endl;
  int num_clas = 1;
  std::vector<float> prob_class(num_clas,0);
  for(int cla = 0; cla < num_clas; cla++ )
  {
    Mat weight_class = _classiferMat->weights.row(cla);
    prob_class[cla] = weight_class.dot(features) + _classiferMat->bias[cla];
  }
  // float sum_of_elems = std::accumulate(prob_class.begin(),prob_class.end(),0);
  // cerr<<"SVM: "<< prob_class[0]  << endl;
  return prob_class[0];//sum_of_elems;
}

void 
SVMLinear::prepareData(
                         Mat& features
                        ,feature_node **x
                        )
{ int ncol = features.cols + 2;
  struct feature_node *xbuff = (struct feature_node*)calloc( (size_t)1 * (size_t)ncol, sizeof(struct feature_node) );
 //fill data

  x[0] = &(xbuff[0]);
  const float *ptr = features.ptr<float>(0);

  for (int j = 0; j < features.cols; ++j) 
  {
    xbuff[j].index = j + 1;
    xbuff[j].value = ptr[j];
  }

  if (_data.bias > 0) 
  {
    xbuff[features.cols].index     = features.cols + 1;
    xbuff[features.cols].value     = _data.bias;
    xbuff[features.cols + 1].index = -1;
    xbuff[features.cols + 1].value = -1;
  } 
  else 
  {
    xbuff[features.cols].index = -1;
    xbuff[features.cols].value = -1;
  }
}


void 
SVMLinear::setParams() 
{
  //many types of SVMLinear
  _param.solver_type = L2R_L2LOSS_SVC;
  _param.C = 1.0;
  _param.eps = 0.0001;
  _param.p = 0.1;
  _param.nr_weight = 2;

  _param.weight_label = (int*)calloc(2, sizeof(int));
  _param.weight_label[0] = 0;
  _param.weight_label[1] = 1;

  // void (*print_func)(const char*) = NULL;
  // print_func = &print_null;
  set_print_string_function(NULL);
}

void 
SVMLinear::setClassWeights(
                      double wtpos
                    , double wtneg
                    ) 
{
  if (_param.weight != NULL) {
    free(_param.weight);
    _param.weight = NULL;
  }

  _param.weight = (double*)calloc(2, sizeof(double));
  _param.weight[0] = wtneg;
  _param.weight[1] = wtpos;
}

void 
SVMLinear::getDecisionFunction()
{
  _classiferMat = new struct SVM_Mat;
  //get parameter describing classifier
  int num_feature = get_nr_feature(_classifier);
  int num_classes = 1;//get_nr_class(_classifier);
  _classiferMat->weights = Mat::zeros(Size(num_feature,num_classes), CV_32FC1);
  _classiferMat->bias.resize(num_classes);
  //extract weights and bias
  for(int lab = 0; lab < num_classes; lab++)
  {
    for(int feat = 0; feat < num_feature; feat++)
    {
      //extract weights
      float weight = float(get_decfun_coef(_classifier, feat, lab));
      _classiferMat->weights.at<float>(lab,feat) = weight;
      // cerr<<feat <<" " << lab << " "<< weight << endl;
    }
    //extract bias
    _classiferMat->bias[lab] = float(get_decfun_bias(_classifier, lab));

  }
  // cerr<<"SVM: "<< _classiferMat->bias[0]<< " "<< _classiferMat->bias[1]<< endl;
}

void 
SVMLinear::loadData(
                const Mat&         features
              , const vector<int>& labels
              ) 
{
  // int npos = pos.rows;
  // int nneg = (int)hards.size();

  // number of samples
  _data.l = labels.size();

  // bias
  _data.bias = 1.0;

  // feature dimension
  if (_data.bias >= 0)
    _data.n = features.cols + 1;
  else
    _data.n = features.cols;

  // targets
  _data.y = (double*)calloc(_data.l, sizeof(double));
  for (uint i = 0; i < labels.size(); ++i)
    _data.y[i] = labels[i];

  // features
  int ncol;
  if (_data.bias >= 0)
    ncol = features.cols + 2;
  else
    ncol = features.cols + 1;

  _data.x = (struct feature_node**)calloc(_data.l, sizeof(struct feature_node*));
  struct feature_node *xbuff = (struct feature_node*)calloc( (size_t)_data.l * (size_t)ncol, sizeof(struct feature_node) );
  
  // fill in positive and negative _data
  for (uint i = 0; i < labels.size(); ++i) 
  {
    _data.x[i] = &(xbuff[i * (size_t)ncol]);
    const float *ptr = features.ptr<float>(i);

    for (int j = 0; j < features.cols; ++j) {
      xbuff[i * (size_t)ncol + j].index = j + 1;
      xbuff[i * (size_t)ncol + j].value = ptr[j];
    }

    if (_data.bias > 0) 
    {
      xbuff[i * (size_t)ncol + features.cols].index     = features.cols + 1;
      xbuff[i * (size_t)ncol + features.cols].value     = _data.bias;
      xbuff[i * (size_t)ncol + features.cols + 1].index = -1;
      xbuff[i * (size_t)ncol + features.cols + 1].value = -1;
    } 
    else 
    {
      xbuff[i * (size_t)ncol + features.cols].index = -1;
      xbuff[i * (size_t)ncol + features.cols].value = -1;
    }
  }
}

void 
SVMLinear::evalModel() 
{
  double npos, nneg,all;
  npos = nneg = all = 0;

  double cpos, cneg;
  cpos = cneg = 0;

  double label;
  double* prob = new double[2];
  // predict samples
  for (int i = 0; i < (int)_data.l; ++i) {
    label = predict(_classifier, _data.x[i]);//,prob);
    // cerr<<prob[0] << " "<< prob[1] <<" "<<_classifier->nr_class <<endl;

    // cerr<<label<<" "<< double(_data.y[i]) <<endl;
    if (label == _data.y[i] && _data.y[i] == double(1.0))
    {
      ++cpos;
      ++npos;
      // cerr<<"0"<<endl;
    }
    else if (_data.y[i] == double(1.0))
    {
      ++npos;
      // cerr<<"1"<<endl;
    }
    else if((label == _data.y[i] && _data.y[i] == double(0.0)))
    {
      ++cneg;
      ++nneg;
      // cerr<<"2"<<endl;
    }
    else if (_data.y[i] == double(0.0))
    {
      ++nneg;
      // cerr<<"3"<<endl;
    }
    
    if(label == _data.y[i])
    {
      all++;
    }
  }
  delete prob;
  cerr<<"Acc: True: " <<cpos / npos<<" False: "<< cneg / nneg<< " all: "<< all/_data.l<< endl;

}

void 
SVMLinear::saveModel( 
                      string name
                    , string nameMat
                    ) 
{
  save_model(name.c_str(), _classifier);
  compress(*_classiferMat,nameMat);
}
void 
SVMLinear::loadModel( 
                      string name
                    , string nameMat
                    )
{
  _classifier   = load_model(name.c_str());
  _classiferMat = new SVM_Mat;
  load(*_classiferMat, nameMat);
}

SVMLinear::~SVMLinear()
{
  if (_data.x != NULL) {
    free(_data.x[0]);
    free(_data.x);
    _data.x = NULL;
  }

  if (_data.y != NULL) {
    free(_data.y);
    _data.y = NULL;
  }


  if( _param.C != 0)
    destroy_param(&_param);
  
  if (_classifier != NULL)
  {
    free_and_destroy_model(&_classifier);
    _classifier = NULL;
  }

  if(_classiferMat != NULL)
    delete _classiferMat;

  
}

