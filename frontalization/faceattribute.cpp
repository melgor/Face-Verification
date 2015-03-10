/* 
* @Author: melgor
* @Date:   2015-02-09 10:07:08
* @Last Modified 2015-03-10
* @Last Modified time: 2015-03-10 11:56:28
*/

#include "faceattribute.hpp"
#include <dlib/opencv.h>


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
                                    vector<Mat>& faces
                                  , vector<FacePoints>& facesPoints
                                  , vector<Rect>& face_rectangle
                                  )
{
  for(auto& img : faces)
  {
    //1.Detect face in image
    // Optional resizing of image
    if (_resizeImageRatio != 1.0)
      cv::resize(img, img, cv::Size(), _resizeImageRatio,_resizeImageRatio, cv::INTER_CUBIC);
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    vector<dlib::rectangle> detected_faces = _frontalFaceDetector(cimg);
    if (!detected_faces.size())
    {
      cerr<<"No Face Detected"<<endl;
      return;
    }
    //2. For each face detect their facial points
    vector<dlib::full_object_detection> shapes;
    Rect image_rect(cv::Point(), img.size());
    for (auto& face_rect : detected_faces)
    {    
        //TODO: add pad only if Rect will be inside of image
        shapes.push_back(_poseModel(cimg, face_rect));
        Rect out;
        getBoundingRect(image_rect,face_rect, out);
        face_rectangle.push_back( out);
    }

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
  cerr<<"W: "<< width_pad<<endl;
  Point left_top     = Point(faceRect.left() - width_pad,faceRect.top() - height_pad);
  Point right_bottom = Point(faceRect.right() + width_pad, faceRect.bottom() + height_pad );
                           
  while(!pointInRect(left_top,right_bottom,imageRect))
  {
     pad_tmp -= 0.1;
     width_pad  = pad_tmp * (faceRect.right() - faceRect.left());
     height_pad = pad_tmp * (faceRect.bottom() - faceRect.top());
     cerr<<"W: "<< width_pad<<endl;
     left_top     = Point(faceRect.left() - width_pad,faceRect.top() - height_pad);
     right_bottom = Point(faceRect.right() + width_pad, faceRect.bottom() + height_pad );

  }

  outRect = Rect(left_top,right_bottom);
}

FaceAttribute::~FaceAttribute()
{

}