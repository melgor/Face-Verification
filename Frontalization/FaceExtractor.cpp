/* 
* @Author: blcv
* @Date:   2015-03-10 11:14:56
* @Last Modified 2015-05-14
* @Last Modified time: 2015-05-14 19:58:46
*/

#include "FaceExtractor.hpp"
#include "Frontalization3D.hpp"
#include <glog/logging.h>
#include <chrono>

using namespace std;
using namespace cv;

FaceExtractor::FaceExtractor(Configuration& config)
{
  _faceatt        = new FaceAttribute(config);
  _camera         = new CameraModel(config);
  _align          = new Frontalization3D(config, _camera);
  _affine         = new AffineModel(config);
  _frontalization = config.frontalization;
  _face2d         = Rect(40,60,239,346);
  _face2dSize     = Size(320,407);
  if (_frontalization == "2D")
    alignment = &FaceExtractor::alignment2D;
  else if(_frontalization == "3D")
    alignment = &FaceExtractor::alignment3D;
}

void
FaceExtractor::getFrontalFace(
                              Mat& image
                            , vector<Mat>& outFrontal
                            )
{
  LOG(WARNING)<<"Face Detection";
  #ifdef __MSTIME
  auto t12 = std::chrono::high_resolution_clock::now();
  #endif
  _faceRect.clear();
  vector<FacePoints> face_points, face_points_align;
  _faceatt->detectFaceAndPoint(image, face_points, _faceRect);

  if(!face_points.size())
  {
    LOG(WARNING)<<"Exit, no Face";
    // outFrontal.push_back(Mat::zeros(100,100,CV_8UC3));
    return;
  }  
  (this->*alignment)(image, _faceRect, face_points, outFrontal);

  #ifdef __MSTIME
  auto t22 = std::chrono::high_resolution_clock::now();
  std::cout << "Face Atribute took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";
  #endif
 
}

void 
FaceExtractor::getFrontalFace(
                                Mat& images
                              , Mat& outFrontal
                              )
{
  //TODO: convert to new Flow +2D Transformation
  LOG(WARNING)<<"Face Detection";
  FacePoints face_points,face_points_align;
  _faceatt->detectFacePoint(images,face_points);
  _faceRect.push_back(Rect(0,0,images.size().width,images.size().height));
  vector<Mat> out_frontal;
  vector<FacePoints> face_points_vec(1,face_points);

 (this->*alignment)(images,_faceRect,face_points_vec,out_frontal);
  outFrontal = out_frontal[0].clone();
  cerr<<"End"<< endl;
}

void 
FaceExtractor::alignment2D( 
                            Mat& image
                          , vector<Rect>& faceRectangle 
                          , vector<FacePoints>& facesPoints
                          , vector<Mat>& outFrontal
                          )
{
  LOG(WARNING)<<"Alignment 2D";
 
  //get size of each detected face
  vector<Size> imageSizes(faceRectangle.size());
  vector<Mat> camera_model_2D(facesPoints.size(),Mat());
  vector<FacePoints> face_points_align(facesPoints.size());
  int num = 0;
  for(auto& img : faceRectangle)
  {
    imageSizes[num] = img.size();
    num++;
  }

  //estimate align using 2D transformation
  _affine->estimateCamera(facesPoints, imageSizes, camera_model_2D);
  //get align face and their face points
  vector<Mat> face_align(facesPoints.size());
  outFrontal.resize(facesPoints.size());
  for(uint i = 0; i < facesPoints.size(); i++)
  {
    Mat face = image(faceRectangle[i]);
    if (camera_model_2D[i].empty())
    {
      //Affine transformation was not calculated
      face_align[i] = face.clone();
      face_points_align[i] = facesPoints[i];
      LOG(WARNING)<<"Copy original";
    }
    else
    {
      warpAffine(face, face_align[i], camera_model_2D[i],_face2dSize);
      Mat per_mat = Mat::zeros(3,3,CV_32FC1);
      for(uint col = 0; col < 3; col++)
        for(uint row = 0; row < 2; row++)
        {
          per_mat.at<float>(row,col) = camera_model_2D[i].at<double>(row,col);
        }
      per_mat.at<float>(2,2) = 1.0f;
      perspectiveTransform(facesPoints[i], facesPoints[i], per_mat);
    }
    // outFrontal[i] = face_align[i](_face2d).clone();
    outFrontal[i] = face_align[i].clone();

    #ifdef __DEBUG
    Mat cc = face_align[i].clone();  
    vector<Point2f> ref_XY = _camera->getRefXY();    
    for(uint j = 0; j <   face_points_align[i].size(); j++)
    {
      cv::circle(cc,face_points_align[i][j],3,cv::Scalar::all(255),-1);
      cv::circle(cc,_affine->getModel68()[j],3,cv::Scalar::all(0),-1);
    }
    for(uint j = 0; j <   6; j++)
    {
      cv::circle(cc,_affine->getModel6()[j],3,cv::Scalar::all(0),-1);
    }
    cv::imwrite("homo_2.jpg",cc);
    #endif
  }
}

void 
FaceExtractor::alignment3D( 
                              Mat& image
                            , vector<Rect>& faceRectangle 
                            , vector<FacePoints>& facesPoints
                            , vector<Mat>& outFrontal
                          )
{
  alignment2D(image,faceRectangle,facesPoints,outFrontal);
  LOG(WARNING)<<"Alignment 3D";
  vector<Mat> camera_model_3D(facesPoints.size(),Mat());
  vector<Size> imageSizes_aling(outFrontal.size(),_face2dSize);
  _camera->estimateCamera(facesPoints, imageSizes_aling, camera_model_3D);

  for(uint i = 0; i < facesPoints.size(); i++)
  {
    _align->frontalize(outFrontal[i],camera_model_3D[i],outFrontal[i]);
  }

}

FaceExtractor::~FaceExtractor()
{
  delete _faceatt;
  delete _camera;
  delete _align;
  delete _affine;
}