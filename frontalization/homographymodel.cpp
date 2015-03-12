/* 
* @Author: blcv
* @Date:   2015-03-11 17:00:34
* @Last Modified 2015-03-11
* @Last Modified time: 2015-03-11 17:25:20
*/
#include "homographymodel.hpp"
#include <opencv2/calib3d/calib3d.hpp>

using namespace std;
using namespace cv;

HomographyModel::HomographyModel(Configuration& config)
{
  loadPoints(config.model2D,_pointModel);
  
  //nose
  _idPoints.push_back(30);
  //left mouth
  _idPoints.push_back(48);
  //right mouth
  _idPoints.push_back(54);
  //middle mouth
  _idPoints.push_back(62);
  
  //left eye
  _leftEyePoints.push_back(36);
  _leftEyePoints.push_back(37);
  _leftEyePoints.push_back(38);
  _leftEyePoints.push_back(39);
  _leftEyePoints.push_back(40);
  _leftEyePoints.push_back(41);
  //right eye
  _rightEyePoints.push_back(42);
  _rightEyePoints.push_back(43);
  _rightEyePoints.push_back(44);
  _rightEyePoints.push_back(45);
  _rightEyePoints.push_back(46);
  _rightEyePoints.push_back(47);
}


void
HomographyModel::estimateCamera(
                              vector<FacePoints>& facesPoints
                            , vector<Size>& imageSize
                            , vector<Mat>& cameraModels
                            )
{
   for(uint i = 0; i < facesPoints.size(); i++)
   {
      estimateCamera(facesPoints[i], imageSize[0], cameraModels[i]);
   }
}

void 
HomographyModel::estimateCamera(
                        FacePoints& facesPoints,
                        cv::Size& imageSize,
                        cv::Mat& cameraModels
                        )
{
  //1. Extract right point from 
  FacePoints face_point_for_homo(_pointModel.size());
  int i = 0;
  for(auto& point : _idPoints)
  {
    face_point_for_homo[i] = facesPoints[point];
    i++;
  }
  std::vector<cv::Point2f> left_eye_model;
  for(auto& point : _leftEyePoints)
  {
    left_eye_model.push_back(facesPoints[point]);
  }
  std::vector<cv::Point2f> right_eye_model;
  for(auto& point : _rightEyePoints)
  {
    right_eye_model.push_back(facesPoints[point]);
  }
  //calculate mean_point using eye_model
  cv::Point2f center_left_eye, center_right_eye;
  calculateMeanPoint(left_eye_model,center_left_eye);
  calculateMeanPoint(right_eye_model,center_right_eye);
  face_point_for_homo[i] = center_left_eye;
  face_point_for_homo[i+1] = center_right_eye;

  cameraModels = findHomography(face_point_for_homo, _pointModel, CV_RANSAC );
}                        

HomographyModel::~HomographyModel()
{

}