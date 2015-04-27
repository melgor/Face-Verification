/* 
* @Author: blcv
* @Date:   2015-03-10 11:14:56
* @Last Modified 2015-04-27
* @Last Modified time: 2015-04-27 10:45:32
*/

#include "FaceExtractor.hpp"
#include "Frontalization3D.hpp"
#include <glog/logging.h>
#include <chrono>

using namespace std;
using namespace cv;

FaceExtractor::FaceExtractor(Configuration& config)
{
  _faceatt      = new FaceAttribute(config);
  _camera       = new CameraModel(config);
  _align        = new Frontalization3D(config, _camera);
  _affine       = new AffineModel(config);
}

void
FaceExtractor::getFrontalFace(
                              vector<Mat>& images
                            , vector<Mat>& outFrontal
                            )
{
  LOG(WARNING)<<"Face Detection";
  #ifdef __MSTIME
  auto t12 = std::chrono::high_resolution_clock::now();
  #endif
  _faceRect.clear();
  vector<FacePoints> face_points, face_points_align;
  // vector<Rect> _faceRect;
  _faceatt->detectFaceAndPoint(images,face_points, _faceRect);

  if(!face_points.size())
  {
    LOG(WARNING)<<"Exit, no Face";
    outFrontal.push_back(Mat::zeros(100,100,CV_8UC3));
    return;
  }  
  // t12 = std::chrono::high_resolution_clock::now();
  LOG(WARNING)<<"Camera Model";
  vector<Mat> camera_model_2D(face_points.size(),Mat());
  vector<Mat> camera_model_3D(face_points.size(),Mat());
  vector<Size> imageSizes;
  #ifdef __DEBUG
   //save image with detection
    Mat cc = images[0].clone();
    for(uint i = 0; i <   _faceRect.size(); i++)
    {
      cv::rectangle(cc,_faceRect[i],cv::Scalar::all(255),3);
    }
    cv::imwrite("detection.jpg",cc);

    //save image with facial_points
    for(uint j = 0; j <   _faceRect.size(); j++)
    {
      Mat cc = images[0](_faceRect[j]).clone();
      for(uint i = 0; i <   face_points[j].size(); i++)
      {
        cv::circle(cc,face_points[j][i],3,cv::Scalar::all(255),-1);
      }
      cv::imwrite( to_string(j) +  "face_point.jpg",cc);
    }
  #endif
  for(auto& img : _faceRect)
  {
    imageSizes.push_back(img.size());
  }
  //estimate align using 2D transformation
  _affine->estimateCamera(face_points, imageSizes, camera_model_2D);
  //get align face and their face points
  vector<Mat> face_align(face_points.size());
  face_points_align.resize(face_points.size());
  outFrontal.resize(face_points.size());
  Rect face2d(40,60,239,346);
  for(uint i = 0; i<face_points.size(); i++)
  {
    Mat faceee = images[0](_faceRect[i]);
    if (camera_model_2D[i].empty())
    {
      //Affine transformation was not calculated
      face_align[i] = faceee.clone();
      face_points_align[i] = face_points[i];
      LOG(WARNING)<<"Copy original";
    }
    else
    {
      // warpPerspective(faceee, face_align[i], camera_model_2D[i], cv::Size(400,400));
      warpAffine(faceee, face_align[i], camera_model_2D[i], cv::Size(320,407));
      // _faceatt->detectFacePoint(face_align[i], face_points_align[i]);
      Mat per_mat = Mat::zeros(3,3,CV_32FC1);
      for(uint col = 0; col < 3; col++)
        for(uint row = 0; row < 2; row++)
        {
          per_mat.at<float>(row,col) = camera_model_2D[i].at<double>(row,col);
        }

      per_mat.at<float>(2,2) = 1.0f;
      // cerr<<per_mat<<endl;
      // cerr<<camera_model_2D[i]<<endl;
      perspectiveTransform(face_points[i], face_points_align[i], per_mat);
    }
    outFrontal[i] = face_align[i](face2d).clone();
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
  //Calculate points which does not aling to Reference point 
  //TODO: It does not work, because face point are in different positions
  //      Morover, it does not improve result
  //  //compare position of Face Point: detected vs model
  // vector<vector<double>> errors_value(face_points_align.size());
  // for(uint i =0; i < face_points_align.size();i++)
  // {
  //   for(uint j =0; j < face_points_align[i].size();j++)
  //   {
  //     errors_value[i].push_back(calcDistance(face_points_align[i][j],_homo->_pointModel68[j]));
  //   }
  // }
  // vector<vector<int>> good_points(face_points_align.size());
  // for(uint i =0; i < face_points_align.size();i++)
  // {
  //   double sum = std::accumulate(errors_value[i].begin(), errors_value[i].end(), 0.0);
  //   double mean = sum / errors_value[i].size();
  //   cerr<<" mean: "<< mean<<endl;
  //   //get idx which are below threshold
  //   for(uint j = 0; j < errors_value[i].size();j++)
  //   {
  //     if(errors_value[i][j] < (mean * 10.3))
  //       good_points[i].push_back(j);
  //   }
  //   cerr<<"good points: "<< good_points[i].size() << endl;
  // }
  


  // #ifdef __MSTIME
  // auto t12F = std::chrono::high_resolution_clock::now();
  // #endif

  // vector<Size> imageSizes_aling(imageSizes.size(),cv::Size(320,407));
  // _camera->estimateCamera(face_points_align, imageSizes_aling, camera_model_3D);

  // for(uint i = 0; i < face_points.size(); i++)
  // {
  //   _align->frontalize(face_align[i],camera_model_3D[i],outFrontal[i]);
  // }

  // #ifdef __MSTIME
  // auto t22 = std::chrono::high_resolution_clock::now();
  // std::cout << "Aligment took "
  //            << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12F).count()
  //            << " milliseconds\n";
  // std::cout << "Face Atribute took "
  //            << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
  //            << " milliseconds\n";

  // cerr<<"End"<< endl;
  // #endif
}

void 
FaceExtractor::getFrontalFace(
                               Mat& images
                              , Mat& outFrontal
                              )
{
  //TODO: convert to new Flow +2D Transformation
  LOG(WARNING)<<"Face Detection";
  auto t12 = std::chrono::high_resolution_clock::now();  
  FacePoints face_points,face_points_align;
  _faceatt->detectFacePoint(images,face_points);
  auto t22 = std::chrono::high_resolution_clock::now();
  std::cout << "detectFacePoint took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";
  LOG(WARNING)<<"Camera Model";
  t12 = std::chrono::high_resolution_clock::now();
  Mat camera_model_2D, camera_model_3D;
  Size image_sizes = images.size();
  //estimate align using 2D transformation
  _affine->estimateCamera(face_points ,image_sizes, camera_model_2D);
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
  Rect rect(0,0,images.size().width, images.size().height);
  _align->frontalize(face_align , camera_model_3D , outFrontal);//, rect);
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
 delete _affine;
}