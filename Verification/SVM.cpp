/* 
* @Author: blcv
* @Date:   2015-03-19 13:08:33
* @Last Modified 2015-04-02
* @Last Modified time: 2015-04-02 13:14:03
*/

#include "SVM.hpp"

using namespace std;
using namespace cv;
void print_null(const char *s) {};

SVMLinear::SVMLinear(Configuration& config)
{
  _data.x     = NULL;
  _data.y     = NULL;
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
}

void 
SVMLinear::setParams() 
{
  //many types of SVMLinear
  _param.solver_type = L2R_L2LOSS_SVC;
  _param.C = 1;
  _param.eps = 0.1;
  _param.p = 0.1;
  _param.nr_weight = 2;

  _param.weight_label = (int*)calloc(2, sizeof(int));
  _param.weight_label[0] = 0;
  _param.weight_label[1] = 1;

  void (*print_func)(const char*) = NULL;
  print_func = &print_null;
  set_print_string_function(print_func);
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
SVMLinear::loadData(
                const Mat   &features
              , const vector<int> &labels
              ) 
{
  // int npos = pos.rows;
  // int nneg = (int)hards.size();

  // number of samples
  _data.l = labels.size();

  // bias
  _data.bias = 0.0;

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
  
  // fill in positive _data
  for (uint i = 0; i < labels.size(); ++i) {
    _data.x[i] = &(xbuff[i * (size_t)ncol]);
    const float *ptr = features.ptr<float>(i);

    for (int j = 0; j < features.cols; ++j) {
      xbuff[i * (size_t)ncol + j].index = j + 1;
      xbuff[i * (size_t)ncol + j].value = ptr[j];
    }

    if (_data.bias >= 0) 
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
    cerr<<prob[0] << " "<< prob[1] <<" "<<_classifier->nr_class <<endl;

    
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
SVMLinear::saveModel(string name) 
{
  save_model(name.c_str(), _classifier);
}
void 
SVMLinear::loadModel(string name) 
{
  _classifier = load_model(name.c_str());
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

  
}

