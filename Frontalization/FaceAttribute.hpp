#ifndef FRONT_FACE_ATTRIBUTE_HPP
#define FRONT_FACE_ATTRIBUTE_HPP

#include <opencv2/core/core.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include "Utils/Utils.hpp"
// Class for detecting face and characteristig face point
// There are several algorithm, here will be used implemented by DLIB:
// "One Millisecond Face Alignment with an Ensemble of Regression Trees by Vahid Kazemi and Josephine Sullivan"
// Remember, that every Face Atribute algorithm has own face model. Face model is delivered by original 
// Frontalization code in MATLAB (here you can create model for every 68-point detector)

typedef std::vector<cv::Point2f> FacePoints;

//TODO: Add easier Face Aligment algorithm. 
class FaceAttribute
{
public:
	FaceAttribute(Configuration& config);
  //detect faces in images and find their face points
	void detectFaceAndPoint( 
                          cv::Mat& img
                        , std::vector<FacePoints>& facesPoints
                        , std::vector<cv::Rect>& face_rectangle);
  //detect face point, where on each image is only one face
  void detectFacePoint(std::vector<cv::Mat>& faces, std::vector<FacePoints>& facesPoints);
  void detectFacePoint(cv::Mat& faces, FacePoints& facesPoints);
	~FaceAttribute();

private:
  //get BB of face, including added Padding (detection rect exlude some parts of face)
  void getBoundingRect(cv::Rect& imageRect, dlib::rectangle& faceRect, cv::Rect& outRect);
  dlib::frontal_face_detector _frontalFaceDetector;
  dlib::shape_predictor       _poseModel;
  float                       _padValue;
  float                       _resizeImageRatio;
  int                         _numNotDetected = 0;

};


#endif