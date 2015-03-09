#ifndef FRONT_CAMERA_MODEL_HPP
#define FRONT_CAMERA_MODEL_HPP
// Camera is a model, which transform rotated, translated face to better coordiante system
// It use face attributes for calculating transformation matrix.
#include "faceattribute.hpp"
#include "utils/parseYAML.hpp"
///Model of reference face. Name same like in MatLab
// struct Model3D
// {
//   cv::Mat                       refU;
//   cv::Mat                       outA;
//   std::vector<cv::Point2f>      ref_XY;
//   cv::Size                      sizeU;
//   std::vector<cv::Point3f>      threedee;
// };



class CameraModel
{
public:
  CameraModel  (Configuration& config);
	void estimateCamera(
												std::vector<FacePoints>& facesPoints,
                        std::vector<cv::Size>& imageSize,
												std::vector<cv::Mat>& cameraModels
												);
  void estimateCamera(
                        FacePoints& facesPoints,
                        cv::Size& imageSize,
                        cv::Mat& cameraModels
                        );
  cv::Mat& getRefU();
  cv::Mat& getEyeMask();
  ~CameraModel();

private:
  //compute pose using reference 3D points + query 2D points
  void    doCalib( FacePoints& facesPoints, cv::Size& imageSize ,cv::Mat& model);
  void    calcCamera( FacePoints& facesPoints,cv::Mat& model );
  int     calcInside(  cv::Mat& A, cv::Mat& R, cv::Mat& T
                        ,cv::Size& size);
  cv::Mat extractFrustum(cv::Mat& A,cv::Mat&  R, cv::Mat& T, cv::Size& size);
  bool    pointInFrustum(cv::Point3f&,cv::Mat& frustum);
  void    getOpenGLMatrices(cv::Mat& A,cv::Mat& R,cv::Mat& T, cv::Size& size
                              , cv::Mat& projMat, cv::Mat& mv);

  struct  Model3D _model3D;
  int             _calibFlags;
  cv::Mat         _tVec;
  cv::Mat         _rMat;
  bool            USE_CALIBRATE = 0;


};

#endif