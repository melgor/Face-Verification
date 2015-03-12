/*
* @Author: melgor
* @Date:   2015-02-09 10:09:01
* @Last Modified 2015-03-10
* @Last Modified time: 2015-03-10 11:42:16
*/

#include "frontalization3D.hpp"
#include "utils/MatFunc.cpp"
#include <vector>


using namespace std;
using namespace cv;


Frontalization3D::Frontalization3D(Configuration& config, CameraModel* camera)
{
  //create main class
  _applySymetry = config.symetry;
  //init value for Frontalization3D 
  vector<Mat> channels_refU(3);
  split(abs(camera->getRefU()), channels_refU);
  Mat tmp = channels_refU[0] + channels_refU[1] + channels_refU[2];
  bitwise_and(tmp ,Scalar(0),_bgind);
  _bgindReshape = _bgind.reshape(1,102400);
  _bgindReshape.convertTo(_bgindReshape,CV_8UC1);

  //count the number of times each pixel in the query is accessed
  _threedee = camera->getRefU().reshape(3,1);
  vector<Mat> splited_threedee;
  split(_threedee,splited_threedee);
  Mat channel_merge_threedee;
  vconcat(splited_threedee[0],splited_threedee[1],channel_merge_threedee);
  vconcat(channel_merge_threedee,splited_threedee[2],_threedee);

  //transform 3 channel to one channel
  tmp = Mat::ones(Size(102400,1),CV_32FC1);
  vconcat(_threedee,tmp,_threedee);
 
  //set outpur crop
  _centralFace = cv::Rect(105,85,100,152);
  _eyeMask        = camera->getEyeMask();
  _cameraRefUSize = camera->getRefU().size();
}

void
Frontalization3D::frontalize(
                            Mat& faceImage
                          , Mat& cameraModel
                          , Mat& outFrontal
                          )
{
  //this matrix is from same image from MatLab fot test image
  // Mat cameraMatlab = cv::Mat::zeros(cv::Size(4,3),CV_64FC1);
  // cameraMatlab.at<double>(0,0) = 516.4864165190479;
  // cameraMatlab.at<double>(0,1) = 33.734888321579350;
  // cameraMatlab.at<double>(0,2) = -8.403603185344464;
  // cameraMatlab.at<double>(0,3) = 8.776728994984608e+04;
  // cameraMatlab.at<double>(1,0) = 30.728277398107327;
  // cameraMatlab.at<double>(1,1) =  1.665485839148806e+02;
  // cameraMatlab.at<double>(1,2) =  -4.891668670804949e+02;
  // cameraMatlab.at<double>(1,3) =  1.147506495507202e+05;
  // cameraMatlab.at<double>(2,0) = 0.246798852750523;
  // cameraMatlab.at<double>(2,1) =  0.968893929497200;
  // cameraMatlab.at<double>(2,2) =  0.018299717607126;
  // cameraMatlab.at<double>(2,3) = 7.730182790263748e+02;

  // cameraMatlab.convertTo(cameraMatlab,CV_32FC1);

  Mat tmp;
  Mat tmp_proj =  _threedee.t() * cameraModel.t();
  Mat tmp_proj2 = tmp_proj.colRange(0,2).clone();
  repeat(tmp_proj.colRange(2,3), 1, 2,  tmp);
  tmp_proj2 = tmp_proj2 / tmp;

  min(tmp_proj2.colRange(0,1),tmp_proj2.colRange(1,2),tmp);
  Mat tmp_1 = tmp < 1;
  Mat tmp_2  = tmp_proj2.colRange(1,2) > faceImage.size().height;
  Mat tmp_3  = tmp_proj2.colRange(0,1) > faceImage.size().width;
  Mat bad = tmp_1 | tmp_2 | tmp_3 | _bgindReshape;

  Mat good_tmp_proj2;
  MatConstIterator_<uchar> it_bad = bad.begin<uchar>(), it_bad_end = bad.end<uchar>();
  int row = 0;
  vector<int> ind_frontal;
  for(MatConstIterator_<uchar> j = it_bad; j != it_bad_end ;++j)
  {
     if ( int(*j) == 0)
     {
      //TODO: how to iterate to take all row?
      good_tmp_proj2.push_back(tmp_proj2.row(row));
      ind_frontal.push_back( row );
     }
     row++;
  }

  Mat good_16;
  good_tmp_proj2.convertTo(good_16,CV_16SC1);
  vector<int> ind = Sub2Ind<short>(faceImage.size().width,faceImage.size().height,good_16.col(1),good_16.col(0));
  sort( ind.begin(), ind.end() );
  vector<int> uniq,ic;
  unique<int>(ind,uniq,ic);

  //calculate histogram
  vector<int> histogram(uniq.size(),0);
  auto iter_uniq = uniq.begin(), iter_uniq_end = uniq.end();
  int index_un = 0;
  for(auto& num : ind)
  {
    //if value is equal the current unique value, add 1 to histogram
    if (num == *iter_uniq)
    {
      histogram[index_un] += 1;
    }
    else
    {
      //else move value to the next unique value. And add value to histogram. 
      iter_uniq++;
      index_un++;
      histogram[index_un] += 1;
      if (iter_uniq == iter_uniq_end)
      {
        break;
      }
    }
  }

  //count(ic)
  vector<int> count_ic;
  int sum_val = 0;
  for(auto& num : ic)
  {
    count_ic.push_back( histogram[num] );
    sum_val += histogram[num];
  }
  // cerr<<"Sum ic: "<< sum_val <<endl;

  Mat synth_frontal_acc = Mat::zeros(_cameraRefUSize, CV_32FC1);
  modify<float>(count_ic,synth_frontal_acc,ind_frontal);
  // cerr<<sum(synth_frontal_acc)<<endl;
  //-----------------------------------
  MatConstIterator_<uchar> it_bgind = _bgind.begin<uchar>(), it_bgind_end = _bgind.end<uchar>();
  MatIterator_<uchar> it_synth = synth_frontal_acc.begin<uchar>();//, it_synth_end = synth_frontal_acc.end<uchar>();
  //int row = 0;
  for(MatConstIterator_<uchar> j = it_bgind; j != it_bgind_end ;++j,++it_synth)
  {
     if ( int(*j) == 1)
     {
      *it_synth = uchar(0);
     }
  }
  GaussianBlur(synth_frontal_acc, synth_frontal_acc, Size(29,29),16.0);

  //Create frontal_raw file
  vector<Mat> channel_photo;
  split(faceImage,channel_photo);
  vector<Mat> f(3,Mat::zeros(_cameraRefUSize,CV_8UC1));
  Mat f1 = Mat::zeros(_cameraRefUSize,CV_8UC1);
  Mat f2 = Mat::zeros(_cameraRefUSize,CV_8UC1);
  Mat f3 = Mat::zeros(_cameraRefUSize,CV_8UC1);
  // cerr<<"Sum: "<< sum(good_tmp_proj2.col(1)) << sum(good_tmp_proj2.col(0))<<endl;
  remap(channel_photo[0],f[0],good_tmp_proj2.col(0),good_tmp_proj2.col(1),CV_INTER_CUBIC);
  //fill image with value from f[0]
  modify<uchar>(f[0],f1,ind_frontal);
  remap(channel_photo[1],f[1],good_tmp_proj2.col(0),good_tmp_proj2.col(1),CV_INTER_CUBIC);
  modify<uchar>(f[1],f2,ind_frontal);
  remap(channel_photo[2],f[2],good_tmp_proj2.col(0),good_tmp_proj2.col(1),CV_INTER_CUBIC);
  modify<uchar>(f[2],f3,ind_frontal);

  //Crete BGR image 
  Mat frontal_raw,color;
  vector<Mat> f_good;
  f_good.push_back(f1);
  f_good.push_back(f2);
  f_good.push_back(f3);
  merge(f_good,frontal_raw);

  // which side has more occlusions?
  //sum of synth_frontal_acc column wise
  int midcolumn =  int(_cameraRefUSize.height /2);
  vector<int> sumaccs(_cameraRefUSize.width,0);
  for(int i = 0; i < _cameraRefUSize.width; i++)
  {
    //TODO:How to iterate efficienty by columns
    sumaccs[i] = sum(synth_frontal_acc.col(i))[0];
  }
  //TODO: Right summations based in iterators
  int sum_left = 0;
  for(uint i = 0; i < sumaccs.size()/2; i++)
  {
    sum_left += sumaccs[i];
  }
  int sum_right = 0;
  for(uint i = sumaccs.size()/2;i < sumaccs.size(); i++)
  {
    sum_right += sumaccs[i];
  }

  int sum_diff = sum_left - sum_right;

  ///----Apply symetry transfrom if one side of face is more ocludded------
  ///----Occlusion if measure by sum_diff. Threshold is set by ACC_CONST---

   #ifdef __DEBUG
    Mat tttt;
    double min2,max2;
    minMaxLoc(frontal_raw, &min2, &max2);
    frontal_raw.convertTo(tttt,CV_8UC3,255.0/(max2-min2),-255.0*min2/(max2-min2));
    imwrite("raw.jpg", tttt);
  #endif


  //Does it help?
  if (abs(sum_diff) > ACC_CONST && _applySymetry)
  {

    Mat weights;
    //apply symetry transform
    Mat weights_1 = Mat::zeros(Size(midcolumn,midcolumn*2),CV_32FC1);
    Mat weights_2 = Mat::ones(Size(midcolumn,midcolumn*2),CV_32FC1);
    if (sum_diff > ACC_CONST) 
    { // left side of face has more occlusions
      hconcat(weights_1,weights_2,weights);
    }
    else
    {// right side of face has occlusions
      hconcat(weights_2,weights_1,weights);
    }
    GaussianBlur(weights, weights, Size(33,33),60.5,0,cv::BORDER_REPLICATE);
    double min,max;
    minMaxLoc(synth_frontal_acc, &min, &max);
    synth_frontal_acc = synth_frontal_acc / max;
    Mat tt;
    exp(  synth_frontal_acc + 0.5f, tt );
    Mat weight_take_from_org = 1.0f / ( tt);
    Mat weight_take_from_sym = 1.0f -weight_take_from_org;
    Mat weights_flip;
    //Good?
    flip(weights, weights_flip, 1);
    weight_take_from_org = weight_take_from_org.mul(weights_flip);
    weight_take_from_sym = weight_take_from_sym.mul(weights_flip);

    //Crete 3-channel mask
    vector<Mat> weight_take_from_org_c(3,weight_take_from_org);
    merge(weight_take_from_org_c,weight_take_from_org);

    vector<Mat> weight_take_from_sym_c(3,weight_take_from_sym);
    merge(weight_take_from_sym_c,weight_take_from_sym);

    vector<Mat> weights_c(3,weights.clone());
    merge(weights_c,weights);

    Mat denominator = weight_take_from_org + weight_take_from_sym + weights;

    Mat frontal_raw_f;
    frontal_raw.convertTo(frontal_raw_f,CV_32FC3);
    Mat frontal_raw_flip;
    flip(frontal_raw_f,frontal_raw_flip,1);
    Mat frontal_sym = (frontal_raw_f.mul(weights) + frontal_raw_f.mul(weight_take_from_org) 
                          + frontal_raw_flip.mul(weight_take_from_sym))/denominator;

    //Exclude eyes from symmetry
    //Does it help??
    frontal_raw = frontal_sym.mul((Scalar(1.0f,1.0f,1.0f)-_eyeMask)) + frontal_raw_f.mul(_eyeMask);


  }

  double min,max;
  minMaxLoc(frontal_raw, &min, &max);
  frontal_raw.convertTo(frontal_raw,CV_8UC3,255.0/(max-min),-255.0*min/(max-min));
  outFrontal = frontal_raw(_centralFace);

  //-------------------END--------------------------

}

Frontalization3D::~Frontalization3D()
{
}