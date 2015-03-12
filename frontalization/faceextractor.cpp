/* 
* @Author: blcv
* @Date:   2015-03-10 11:14:56
* @Last Modified 2015-03-10
* @Last Modified time: 2015-03-10 18:36:03
*/

#include "faceextractor.hpp"
#include "frontalization3D.hpp"
#include <chrono>

using namespace std;
using namespace cv;

FaceExtractor::FaceExtractor(Configuration& config)
{
  _faceatt      = new FaceAttribute(config);
  _camera       = new CameraModel(config);
  _align        = new Frontalization3D(config, _camera);
  _homo         = new HomographyModel(config);
}

void
FaceExtractor::getFrontalFace(
                              vector<Mat>& images
                            , vector<Mat>& outFrontal
                            )
{
  cerr<<"Face Detection"<< endl;
  auto t12 = std::chrono::high_resolution_clock::now();
  vector<FacePoints> face_points, face_points_align;
  vector<Rect> face_rect;
  _faceatt->detectFaceAndPoint(images,face_points, face_rect);
  auto t22 = std::chrono::high_resolution_clock::now();

  std::cout  << "detectFacePoint took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";

  t12 = std::chrono::high_resolution_clock::now();
  cerr<<"Camera Model"<<endl;
  vector<Mat> camera_model_2D(face_points.size(),Mat());
  vector<Mat> camera_model_3D(face_points.size(),Mat());
  vector<Size> imageSizes;
  for(auto& img : images)
  {
    imageSizes.push_back(img.size());
  }
  #ifdef __DEBUG
   //save image with detection
    Mat cc = images[0].clone();             
    for(uint i = 0; i <   face_rect.size(); i++)
    {
      cv::rectangle(cc,face_rect[i],cv::Scalar::all(255),3);
    }
    cv::imwrite("detection.jpg",cc);
    
    //save image with facial_points
    for(uint j = 0; j <   face_rect.size(); j++)
    {
      Mat cc = images[0](face_rect[j]).clone();             
      for(uint i = 0; i <   face_points[j].size(); i++)
      {
        cv::circle(cc,face_points[j][i],3,cv::Scalar::all(255),-1);
      }
      cv::imwrite( to_string(j) +  "face_point.jpg",cc);  
    } 
  #endif
  //estimate align using 2D transformation
  _homo->estimateCamera(face_points ,imageSizes, camera_model_2D);
  //get align face and their face points
  vector<Mat> face_align(face_points.size());
  face_points_align.resize(face_points.size());
  for(uint i = 0; i<face_points.size(); i++)
  {
    Mat faceee = images[0](face_rect[i]);
    warpPerspective(faceee, face_align[i], camera_model_2D[i], cv::Size(400,400));
    perspectiveTransform(face_points[i], face_points_align[i], camera_model_2D[i]);
    Mat cc = face_align[i].clone();      
    for(uint j = 0; j <   face_points_align[i].size(); j++)
    {
      cv::circle(cc,face_points_align[i][j],3,cv::Scalar::all(255),-1);
    }
    cv::imwrite("homo_2.jpg",cc);
  }
  vector<Size> imageSizes_aling(imageSizes.size(),cv::Size(400,400));
  _camera->estimateCamera(face_points_align, imageSizes_aling, camera_model_3D);
  t22 = std::chrono::high_resolution_clock::now();

  std::cout << "estimateCamera took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";

  outFrontal.resize(face_points.size());
  t12 = std::chrono::high_resolution_clock::now();
  for(uint i = 0; i < face_points.size(); i++)
  {
    _align->frontalize(face_align[i],camera_model_3D[i],outFrontal[i]);
  }
  t22 = std::chrono::high_resolution_clock::now();
  std::cout << "frontalize took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";

  cerr<<"End"<< endl;
}

void 
FaceExtractor::getFrontalFace(
                               Mat& images
                              , Mat& outFrontal
                              )
{
  //TODO: convert to new Flow +2D Transformation
  cerr<<"Face Detection"<< endl;
  auto t12 = std::chrono::high_resolution_clock::now();  
  FacePoints face_points,face_points_align;
  _faceatt->detectFacePoint(images,face_points);
  auto t22 = std::chrono::high_resolution_clock::now();
  std::cout << "detectFacePoint took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";
  cerr<<"Camera Model"<<endl;
  t12 = std::chrono::high_resolution_clock::now();
  Mat camera_model_2D, camera_model_3D;
  Size image_sizes = images.size();
  //estimate align using 2D transformation
  _homo->estimateCamera(face_points ,image_sizes, camera_model_2D);
  //get align face and their face points
  Mat face_align;
  warpPerspective(images, face_align, camera_model_2D, cv::Size(400,400));
  perspectiveTransform(face_points, face_points_align, camera_model_2D);
  cv::imwrite("homo.jpg",face_align);
  Size imageSizes_aling = Size(400,400);
  
  _camera->estimateCamera(face_points_align, imageSizes_aling, camera_model_3D);
  t22 = std::chrono::high_resolution_clock::now();
  std::cout << "camera took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";
  cerr<<"Frontalization"<<endl;
  t12 = std::chrono::high_resolution_clock::now();
  _align->frontalize(face_align , camera_model_3D , outFrontal);
  t22 = std::chrono::high_resolution_clock::now();
  std::cout << "frontalize took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";
  cerr<<"End"<< endl;
}

FaceExtractor::~FaceExtractor()
{
 delete _faceatt;
 delete _camera;
 delete _align;
 delete _homo;
}