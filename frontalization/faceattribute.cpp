/* 
* @Author: melgor
* @Date:   2015-02-09 10:07:08
* @Last Modified 2015-02-23
* @Last Modified time: 2015-02-23 15:14:09
*/

#include "faceattribute.hpp"
#include <dlib/opencv.h>


using namespace std;
using namespace cv;

FaceAttribute::FaceAttribute(Configuration& config)
{
  _frontalFaceDetector = dlib::get_frontal_face_detector();
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
    // TODO: Optional resizing of image
    cv::resize(img, img, cv::Size(), 2, 2, cv::INTER_CUBIC);
    dlib::cv_image<dlib::bgr_pixel> cimg(img);
    vector<dlib::rectangle> detected_faces = _frontalFaceDetector(cimg);
    //2. For each face detect their facial points
    vector<dlib::full_object_detection> shapes;
    for (unsigned long i = 0; i < detected_faces.size(); ++i)
    {    
        //TODO: add pad only if Rect will be inside of image
        shapes.push_back(_poseModel(cimg, detected_faces[i]));
        face_rectangle.push_back(Rect(Point(detected_faces[i].left() -_padValue,detected_faces[i].top() -_padValue) ,
                                 Point(detected_faces[i].right() + _padValue, detected_faces[i].bottom() + _padValue )) );
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

FaceAttribute::~FaceAttribute()
{

}