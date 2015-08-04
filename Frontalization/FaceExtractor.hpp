#ifndef FRONT_FACEEXTRACTOR_HPP
#define FRONT_FACEEXTRACTOR_HPP

#include "FaceAttribute.hpp"
#include "CameraModel.hpp"
#include "Frontalization3D.hpp"
#include "AffineModel.hpp"

//Class for controling all flow of face Alignment:
//1. Detection of face
//2. Alignent
//3. Cropping of face

class FaceExtractor
{
  public:
    FaceExtractor() {};
    FaceExtractor(Configuration& config);
    void getFrontalFace(cv::Mat& images, std::vector<cv::Mat>& outFrontal);
    void getFrontalFace(cv::Mat& images, cv::Mat& outFrontal);
    ~FaceExtractor();

    std::vector<cv::Rect> _faceRect;
  private:
    void (FaceExtractor::*alignment)( 
                    cv::Mat& image
                  , std::vector<cv::Rect>& faceRectangle
                  , std::vector<FacePoints>& facesPoints
                  , std::vector<cv::Mat>& outFrontal
                  ) = NULL;
    //Method where 2D alignment is used, using AffineModel
    void alignment2D( 
                    cv::Mat& image
                  , std::vector<cv::Rect>& faceRectangle 
                  , std::vector<FacePoints>& facesPoints
                  , std::vector<cv::Mat>& outFrontal
                  );
    //Method where 3D alignment is used, using Frontalization3D
    void alignment3D( 
                    cv::Mat& image
                  , std::vector<cv::Rect>& faceRectangle 
                  , std::vector<FacePoints>& facesPoints
                  , std::vector<cv::Mat>& outFrontal
                  );



    //Models need for Face Extraction
    FaceAttribute*    _faceatt;
    CameraModel*      _camera;
    Frontalization3D* _align;
    AffineModel*      _affine;
    //configuration
    std::string       _frontalization;
    std::string       _cropping;
    std::string       _croppingNormal = "Standard";
    std::string       _croppingTight  = "Tight";
    //option for output face
    cv::Rect          _face2d;
    cv::Size          _face2dSize;
};

#endif