/* 
* @Author: melgor
* @Date:   2015-02-09 10:07:08
* @Last Modified 2015-05-04
* @Last Modified time: 2015-05-04 11:41:22
*/
#include "FaceAttribute.hpp"
#include <dlib/opencv.h>
#include <chrono>
#include <glog/logging.h>

using namespace std;
using namespace cv;

//chekc if point are in Rectangle
bool pointInRect( 
                 Point& leftTop
                ,Point& bottomRight
                ,Rect&  rect
                )
{

 return (rect.contains(leftTop) && rect.contains(bottomRight));
}


FaceAttribute::FaceAttribute(Configuration& config)
{
  _frontalFaceDetector = dlib::get_frontal_face_detector();
  _padValue            = config.padDetection;
  _resizeImageRatio    = config.resizeImageRatio;
  dlib::deserialize(config.posemodel) >> _poseModel;
}

void
FaceAttribute::detectFaceAndPoint(
                                    Mat& img
                                  , vector<FacePoints>& facesPoints
                                  , vector<Rect>& face_rectangle
                                  )
{
 
  DLOG(WARNING) << "Image: "<< img.size();
  //1.Detect face in image
  // Optional resizing of image
  if (_resizeImageRatio != 1.0)
  {
    DLOG(WARNING) << "Resize: "<< _resizeImageRatio;
    cv::resize(img, img, cv::Size(), _resizeImageRatio,_resizeImageRatio, cv::INTER_CUBIC);
  }
  dlib::cv_image<dlib::bgr_pixel> cimg(img);
  #ifdef __MSTIME
  auto t12 = std::chrono::high_resolution_clock::now();
  #endif
  vector<dlib::rectangle> detected_faces = _frontalFaceDetector(cimg);
  #ifdef __MSTIME
  auto t22 = std::chrono::high_resolution_clock::now();
  std::cout  << "Face Detection took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";
  #endif
       
  if (!detected_faces.size())
  {
    DLOG(WARNING) <<"No Face Detected";
    // return;
    dlib::rectangle rect(img.size().width,img.size().height);
    detected_faces.push_back(rect);
  }
  else
  {
    DLOG(WARNING) << "Detected Faces: "<< detected_faces.size();      
  }
  //2. For each face detect their facial points
  vector<dlib::full_object_detection> shapes;
  Rect image_rect(cv::Point(), img.size());
  #ifdef __MSTIME
  t12 = std::chrono::high_resolution_clock::now();
  #endif
  for (auto& face_rect : detected_faces)
  {    
      Rect face_cv(cv::Point(face_rect.left(),face_rect.top()),cv::Point(face_rect.right(),face_rect.bottom()));
      Rect out;
      if(rectIntersection(image_rect,face_cv) != 1.0)
      {
        //rectangle is outside image
        continue;
      }
      shapes.push_back(_poseModel(cimg, face_rect));
      //add pad only if Rect will be inside of image
      getBoundingRect(image_rect,face_rect, out);
      face_rectangle.push_back( out);
  }
  #ifdef __MSTIME
  t22 = std::chrono::high_resolution_clock::now();
  std::cout  << "Facial Points took "
             << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
             << " milliseconds\n";
  #endif               

  facesPoints.reserve(shapes.size());
  //3. Transform point to OpenCV format
  int idx = 0;
  for(auto& shape : shapes)
  {
    FacePoints tmp;
    for(uint i = 0; i < shape.num_parts(); i++)
    {
      tmp.push_back(Point2f(float(shape.part(i).x()) - face_rectangle[idx].x ,
                                  float(shape.part(i).y() - face_rectangle[idx].y )));
    }
    facesPoints.push_back(tmp);
    idx++;
  }
  
}

void
FaceAttribute::detectFacePoint(
                                vector<Mat>& faces
                              , vector<FacePoints>& facesPoints
                              )
{
  for(auto& img : faces)
  {
    facesPoints.push_back(FacePoints());
    detectFacePoint(img,facesPoints.back());
  }
}

void
FaceAttribute::detectFacePoint(
                                Mat& faces
                              , FacePoints& facesPoints
                              )
{
 
    dlib::cv_image<dlib::bgr_pixel> cimg(faces);
    dlib::rectangle face(faces.size().width,faces.size().height);
    dlib::full_object_detection shape = _poseModel(cimg, face);
    for(uint i = 0; i < shape.num_parts(); i++)
    {
      facesPoints.push_back(cv::Point2f(float(shape.part(i).x()), float(shape.part(i).y())));
    }

}


void
FaceAttribute::getBoundingRect(
                                Rect& imageRect
                              , dlib::rectangle& faceRect
                              , Rect& outRect
                              )
{
  float pad_tmp = _padValue;
  float width_pad  = pad_tmp * (faceRect.right() - faceRect.left());
  float height_pad = pad_tmp * (faceRect.bottom() - faceRect.top());
 
  Point left_top     = Point(faceRect.left() - width_pad,faceRect.top() - height_pad);
  Point right_bottom = Point(faceRect.right() + width_pad, faceRect.bottom() + height_pad );
  // cerr<<"W: "<< width_pad <<"   "<< imageRect <<"  "<< left_top << "    "<< right_bottom <<endl;                             
  while(!pointInRect(left_top,right_bottom,imageRect) && width_pad > 0.0)
  {
     pad_tmp -= 0.1;
     width_pad  = pad_tmp * (faceRect.right() - faceRect.left());
     height_pad = pad_tmp * (faceRect.bottom() - faceRect.top());
     left_top     = Point(faceRect.left() - width_pad,faceRect.top() - height_pad);
     right_bottom = Point(faceRect.right() + width_pad, faceRect.bottom() + height_pad );
     // cerr<<"W: "<< width_pad <<"   "<< imageRect <<"  "<< left_top << "    "<< right_bottom <<endl; 
  }
  DLOG(WARNING) <<"W: "<< width_pad <<"   "<< imageRect <<"  "<< left_top << "    "<< right_bottom; 
  outRect = Rect(left_top,right_bottom);
}

FaceAttribute::~FaceAttribute()
{

}