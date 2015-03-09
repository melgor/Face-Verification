/* 
* @Author: melgor
* @Date:   2015-02-09 10:03:31
* @Last Modified 2015-03-06
* @Last Modified time: 2015-03-06 11:33:34
*/

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <math.h>
#include "cameramodel.hpp"

//TODO: eliminate YAML dependies by serialzing DLIB camera model
// 1. Camera calibration checked with Matlab: Done
// 2. Check calcinside with Matlab: TODO
using namespace std;
using namespace cv;

CameraModel::CameraModel(Configuration& config)
{
  parseYAML(config.facemodel,_model3D);
  _calibFlags = CV_CALIB_FIX_PRINCIPAL_POINT + CV_CALIB_ZERO_TANGENT_DIST
      + CV_CALIB_FIX_ASPECT_RATIO + CV_CALIB_USE_INTRINSIC_GUESS;
  
  if(config.calibOption == "STANDARD")
    USE_CALIBRATE = 1;
  //else use EXTRINIC
}

void
CameraModel::estimateCamera(
                              vector<FacePoints>& facesPoints
                            , vector<Size>& imageSize
                            , vector<Mat>& cameraModels
                            )
{
   for(uint i = 0; i < facesPoints.size(); i++)
   {
    doCalib(facesPoints[i], imageSize[0], cameraModels[i]);
   }
}

void
CameraModel::estimateCamera(
                              FacePoints& facesPoints
                            , Size& imageSize
                            , Mat& cameraModels
                            )
{
  doCalib(facesPoints, imageSize,cameraModels);
}

void
CameraModel::doCalib(
                      FacePoints& facesPoints
                    , Size& imageSize
                    , Mat& model
                    )
{
  calcCamera(facesPoints,model);
  // int i = calcInside(model,_rMat,_tVec, imageSize);
  //TODO: it does not work. Check with MatLab
  // if (i == 0)
  // {
  //    _tVec = -_tVec;
  //   float t = CV_PI;
  //   Mat RRz180 = Mat::zeros(Size(3,3),CV_32FC1);
  //   RRz180.at<float>(0,0) =  cos(t);
  //   RRz180.at<float>(0,1) = -sin(t);
  //   RRz180.at<float>(1,0) =  sin(t);
  //   RRz180.at<float>(1,1) =  cos(t);
  //   RRz180.at<float>(2,2) =  1;
  //   _rMat = RRz180 * _rMat;
  // }
  
  Mat RT;
  hconcat(_rMat,_tVec,RT);
  model = model * RT;
}

Mat&
CameraModel::getRefU()
{
  return _model3D.refU;
}

Mat&
CameraModel::getEyeMask() 
{
  return _model3D.eyeMask;
}

void
CameraModel::calcCamera(
                          FacePoints& facesPoints
                        , Mat& model
                        )
{
  Mat distCoeffs = Mat::zeros(4, 1, CV_64F);
  model = _model3D.outA.clone();
  _tVec = Mat();
  _rMat = Mat();
  if (USE_CALIBRATE)
  {
    vector<vector<Point3f>> objectPoints(1,_model3D.threedee);
    vector<FacePoints> imagePoints(1,facesPoints);
    vector<Mat> rvecs_vec, tvecs_vec;
    double rms = calibrateCamera(objectPoints, imagePoints, _model3D.sizeU, model,
                    distCoeffs, rvecs_vec, tvecs_vec, _calibFlags);
    cerr<<"RMS "<<rms << endl;
    Rodrigues(rvecs_vec[0].t(),_rMat);
    _rMat.convertTo(_rMat, CV_32FC1);
    tvecs_vec[0].convertTo(_tVec,CV_32FC1);
  }
  else
  {
    Mat rvecs_v = Mat::zeros(3, 1, CV_64F);
    Mat tvecs_v = Mat::zeros(3, 1, CV_64F);
    solvePnP(_model3D.threedee, facesPoints, model, distCoeffs, rvecs_v, tvecs_v);
    Rodrigues(rvecs_v.t(),_rMat);
    _rMat.convertTo(_rMat, CV_32FC1);
    tvecs_v.convertTo(_tVec,CV_32FC1);
  }

  model.convertTo(model, CV_32FC1);

}

int
CameraModel::calcInside(
                         Mat& A, Mat& R
                        ,Mat& T, Size& size
                        )
{
  Mat frustum = extractFrustum(A, R, T, size);
  int inside = 0;
  for(auto& point : _model3D.threedee)
  {
    if (pointInFrustum(point,frustum) == true)
        inside = inside + 1;
  }
  return inside;

}

//T( cv::Range(0,3), cv::Range(0,3) ) = R * 1; // copies R into T
//T( cv::Range(0,3), cv::Range(3,4) ) = tvec * 1; // copies tvec into T
Mat
CameraModel::extractFrustum(
                             Mat& A, Mat&  R
                           , Mat& T, Size& size
                           )
{
  Mat projMat,mv;
  getOpenGLMatrices(A, R, T, size,projMat ,mv);
  Mat clip    = projMat * mv;
  Mat frustum =  Mat::zeros(Size(4,6),CV_32FC1);
  /* Extract the numbers for the RIGHT plane */
  frustum.at<float>(0,0) = clip.at<float>(0,3) - clip.at<float>(0,0);
  frustum.at<float>(0,1) = clip.at<float>(1,3) - clip.at<float>(1,0);
  frustum.at<float>(0,2) = clip.at<float>(2,3) - clip.at<float>(2,0);
  frustum.at<float>(0,3) = clip.at<float>(3,3) - clip.at<float>(3,0);

   /* Normalize the result */
  float v = sqrt(frustum.at<float>(0,0)*frustum.at<float>(0,0) + frustum.at<float>(0,1)*frustum.at<float>(0,1) 
                          + frustum.at<float>(0,2)*frustum.at<float>(0,2));
  frustum.at<float>(0,0) /= v;
  frustum.at<float>(0,1) /= v;
  frustum.at<float>(0,2) /= v;
  frustum.at<float>(0,3) /= v;


  /* Extract the numbers for the LEFT plane */
  frustum.at<float>(1,0) = clip.at<float>(0,3) + clip.at<float>(0,0);
  frustum.at<float>(1,1) = clip.at<float>(1,3) + clip.at<float>(1,0);
  frustum.at<float>(1,2) = clip.at<float>(2,3) + clip.at<float>(2,0);
  frustum.at<float>(1,3) = clip.at<float>(3,3) + clip.at<float>(3,0);

  /* Normalize the result */
  v = sqrt(frustum.at<float>(1,0)*frustum.at<float>(1,0) + frustum.at<float>(1,1)*frustum.at<float>(1,1) 
                          + frustum.at<float>(1,2)*frustum.at<float>(1,2));
  frustum.at<float>(1,0) /= v;
  frustum.at<float>(1,1) /= v;
  frustum.at<float>(1,2) /= v;
  frustum.at<float>(1,3) /= v;


  /* Extract the BOTTOM plane */
  frustum.at<float>(2,0) = clip.at<float>(0,3) + clip.at<float>(0,1);
  frustum.at<float>(2,1) = clip.at<float>(1,3) + clip.at<float>(1,1);
  frustum.at<float>(2,2) = clip.at<float>(2,3) + clip.at<float>(2,1);
  frustum.at<float>(2,3) = clip.at<float>(3,3) + clip.at<float>(3,1);

  /* Normalize the result */
  v = sqrt(frustum.at<float>(2,0)*frustum.at<float>(2,0) + frustum.at<float>(2,1)*frustum.at<float>(2,1)
                          + frustum.at<float>(2,2)*frustum.at<float>(2,2));

  frustum.at<float>(2,0) /= v;
  frustum.at<float>(2,1) /= v;
  frustum.at<float>(2,2) /= v;
  frustum.at<float>(2,3) /= v;


  /* Extract the TOP plane */
  frustum.at<float>(3,0) = clip.at<float>(0,3) - clip.at<float>(0,1);
  frustum.at<float>(3,1) = clip.at<float>(1,3) - clip.at<float>(1,1);
  frustum.at<float>(3,2) = clip.at<float>(2,3) - clip.at<float>(2,1);
  frustum.at<float>(3,3) = clip.at<float>(3,3) - clip.at<float>(3,1);

  /* Normalize the result */
  v = sqrt(frustum.at<float>(3,0)*frustum.at<float>(3,0) + frustum.at<float>(3,1)*frustum.at<float>(3,1) 
                          + frustum.at<float>(3,2)*frustum.at<float>(3,2));

  frustum.at<float>(3,0) /= v;
  frustum.at<float>(3,1) /= v;
  frustum.at<float>(3,2) /= v;
  frustum.at<float>(3,3) /= v;


  /* Extract the FAR plane */
  frustum.at<float>(4,0) = clip.at<float>(0,3) - clip.at<float>(0,2);
  frustum.at<float>(4,1) = clip.at<float>(1,3) - clip.at<float>(1,2);
  frustum.at<float>(4,2) = clip.at<float>(2,3) - clip.at<float>(2,2);
  frustum.at<float>(4,3) = clip.at<float>(3,3) - clip.at<float>(3,2);

  /* Normalize the result */
  v = sqrt(frustum.at<float>(4,0)*frustum.at<float>(4,0) + frustum.at<float>(4,1)*frustum.at<float>(4,1) 
                          + frustum.at<float>(4,2)*frustum.at<float>(4,2));

  frustum.at<float>(4,0) /= v;
  frustum.at<float>(4,1) /= v;
  frustum.at<float>(4,2) /= v;
  frustum.at<float>(4,3) /= v;

  /* Extract the NEAR plane */
  frustum.at<float>(5,0) = clip.at<float>(0,3) + clip.at<float>(0,2);
  frustum.at<float>(5,1) = clip.at<float>(1,3) + clip.at<float>(1,2);
  frustum.at<float>(5,2) = clip.at<float>(2,3) + clip.at<float>(2,2);
  frustum.at<float>(5,3) = clip.at<float>(3,3) + clip.at<float>(3,2);

  /* Normalize the result */
  v = sqrt(frustum.at<float>(5,0)*frustum.at<float>(5,0) + frustum.at<float>(5,1)*frustum.at<float>(5,1) 
                          + frustum.at<float>(5,2)*frustum.at<float>(5,2));

  frustum.at<float>(5,0) /= v;
  frustum.at<float>(5,1) /= v;
  frustum.at<float>(5,2) /= v;
  frustum.at<float>(5,3) /= v;


  return frustum;
}

bool
CameraModel::pointInFrustum(
                              Point3f& point
                            , Mat& frustum
                            )
{
  for(uint p = 0; p < 4; p++)
  {
    if ((frustum.at<float>(p,0) * point.x + frustum.at<float>(p,1) * point.y 
           + frustum.at<float>(p,2) * point.z + frustum.at<float>(p,3)) <= 0)
        return false;
  }

  return true;
}

void
CameraModel::getOpenGLMatrices(
                                Mat& A,Mat& R,Mat& T
                              , Size& size, Mat& projMat
                              , Mat& mv
                              )
{
  projMat = Mat::zeros(Size(4,4),CV_32FC1);

  float nearPlane = 0.0001;
  float farPlane = 10000;

  float fx = A.at<float>(0,0);
  float fy = A.at<float>(1,1);
  float px = A.at<float>(0,2);
  float py = A.at<float>(1,2);
  projMat.at<float>(0,0) = 2.0 * fx / size.width;
  projMat.at<float>(0,1) = 0.0;
  projMat.at<float>(0,2) = 0.0;
  projMat.at<float>(0.3) = 0.0;

  projMat.at<float>(1,0) = 0.0;
  projMat.at<float>(1,1) = 2.0 * fy / size.height;
  projMat.at<float>(1,2) = 0.0;
  projMat.at<float>(1,3) = 0.0;

  projMat.at<float>(2,0) = 2.0 * (px / size.width) - 1.0;
  projMat.at<float>(2,1) = 2.0 * (py / size.height) - 1.0;
  projMat.at<float>(2,2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  projMat.at<float>(2,3) = -1.0;

  projMat.at<float>(3,0) = 0.0;
  projMat.at<float>(3,1) = 0.0;
  projMat.at<float>(3,2) = -2.0 * farPlane * nearPlane / (farPlane - nearPlane);
  projMat.at<float>(3,3) = 0.0;


  //OpenGL's Y and Z axis are opposite to the camera model (OpenCV)
  //same as RRz(180)*RRy(180)*R:
  //  1.0000    0.0000    0.0000
  //  0.0000   -1.0000    0.0000    *   R
  //  0.0000         0   -1.0000
  float deg  = 180;
  float t    = deg * CV_PI / 180.0f;

  Mat RRz = Mat::zeros(Size(3,3),CV_32FC1);
  RRz.at<float>(0,0) =  cos(t);
  RRz.at<float>(0,1) = -sin(t);
  RRz.at<float>(1,0) =  sin(t);
  RRz.at<float>(1,1) =  cos(t);
  RRz.at<float>(2,2) =  1;

  Mat RRy = Mat::zeros(Size(3,3),CV_32FC1);
  RRy.at<float>(0,0) =  cos(t);
  RRy.at<float>(0,2) =  sin(t);
  RRy.at<float>(1,1) =  1;
  RRy.at<float>(2,0) =  -sin(t);
  RRy.at<float>(2,2) =  cos(t);

  Mat R_GL = RRz * RRy * R;

  mv = Mat::zeros(Size(4,4),CV_32FC1);
  for (uint x = 0; x < 3; x++)
    for (uint y = 0; y < 3; y++)
    {
      mv.at<float>(y,x) = R_GL.at<float>(y,x);
    }

  mv.at<float>(0,3) = T.at<float>(0);
  // also invert Y and Z of translation
  mv.at<float>(1,3) = -T.at<float>(1);
  mv.at<float>(2,3) = -T.at<float>(2);
  mv.at<float>(3,3) = 1.0;

}


CameraModel::~CameraModel()
{

}