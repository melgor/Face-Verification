#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <fstream>
#include "Parser.hpp"
#include <memory>
#include "opencv2/opencv.hpp"

struct Configuration
{
  bool         reset;
  //mode of program
  std::string  mode;
  //input
  std::string  nameScene;
  std::string  folderpath;
  //face detection
  std::string  posemodel;
  std::string  facemodel;
  float        padDetection;
  float        resizeImageRatio;
  std::string  calibOption;
  bool         symetry;
  std::string  model2D_6;
  std::string  model2D_68;
  //Net
  std::string  prototxt;
  std::string  caffemodel;
  bool         gpu;
  int          gpuID;
  std::string  layer;
  //Extractor
  std::string  extractorFolder;
  std::string  extractorImageList;
  //Verification
  std::string  trainData;
  std::string  valData;
  std::string  metric;
  std::string  pathComparator;
  std::string  pathComparatorMat;
  std::string  pathScaler;
  std::string  faceData;
  std::string  faceLabels;
  float        threshold;
  bool         scaleFeature;




  void read(int argc, char** argv)
  {
    Parser parser;
    parser.read(argc, argv);

    reset          = parser.reset;
    nameScene      = parser.scene;
    folderpath     = parser.folderpath;

    boost::property_tree::ptree pt;
    boost::property_tree::ini_parser::read_ini(parser.config, pt);
    //mode
    mode           = pt.get<std::string>("Mode.Mode");
    //face detection
    posemodel      = pt.get<std::string>("FaceDecetion.PoseModel");
    facemodel      = pt.get<std::string>("FaceDecetion.FaceModel");
    padDetection   = pt.get<float>("FaceDecetion.PadDetection");
    resizeImageRatio =  pt.get<float>("FaceDecetion.ResizeImageRatio");
    calibOption      =  pt.get<std::string>("FaceDecetion.CalibOption");
    symetry          =  pt.get<bool>("FaceDecetion.Symetry");
    model2D_6        =  pt.get<std::string>("FaceDecetion.Model2D_6points");
    model2D_68       =  pt.get<std::string>("FaceDecetion.Model2D_68points");
    //net 
    prototxt       = pt.get<std::string>("Net.Prototxt");
    caffemodel     = pt.get<std::string>("Net.CaffeModel");
    layer          = pt.get<std::string>("Net.Layer");
    gpu            = pt.get<bool>("Net.GPU");
    gpuID          = pt.get<int>("Net.GPU_ID");
    //Extractor
    extractorFolder    = pt.get<std::string>("Extract.Folder");
    extractorImageList = pt.get<std::string>("Extract.ImageListDB");
    //Verificator
    trainData         = pt.get<std::string>("Verification.TrainData");
    valData           = pt.get<std::string>("Verification.ValData");
    metric            = pt.get<std::string>("Verification.Metric");
    pathComparator    = pt.get<std::string>("Verification.ComparatorPath");
    pathComparatorMat = pt.get<std::string>("Verification.ComparatorPathMat");
    threshold         = pt.get<float>("Verification.Thres");
    pathScaler        = pt.get<std::string>("Verification.ScalerPath");
    scaleFeature      = pt.get<bool>("Verification.ScaleFeature");
    faceData          = pt.get<std::string>("Verification.FaceData");
    faceLabels        = pt.get<std::string>("Verification.FaceLabels");
  }

  void print()
  {
    std::cerr<<"-------------------------------------" << std::endl;
    std::cerr<<"Configuration " << std::endl;
    std::cerr<<"nameScene: "<<nameScene << std::endl;
    std::cerr<<"folderpath: "<<folderpath << std::endl;
    std::cerr<<"mode: "<<mode << std::endl;
    std::cerr<<"------------Face Detection----------------" << std::endl;
    std::cerr<<"posemodel: "<<posemodel << std::endl;
    std::cerr<<"facemodel: "<<facemodel << std::endl;
    std::cerr<<"padDetection: "<<padDetection << std::endl;
    std::cerr<<"ResizeImageRatio: "<<resizeImageRatio << std::endl;
    std::cerr<<"CalibOption: "<<calibOption << std::endl;
    std::cerr<<"Symetry: "<<symetry << std::endl;
    std::cerr<<"Model2D_6: "<<model2D_6 << std::endl;
    std::cerr<<"Model2D_68: "<<model2D_68 << std::endl;
    std::cerr<<"------------Net---------------------------" << std::endl;
    std::cerr<<"prototxt: "<<prototxt << std::endl;
    std::cerr<<"caffemodel: "<<caffemodel << std::endl;
    std::cerr<<"layer: "<<layer << std::endl;
    std::cerr<<"gpu: "<<gpu << std::endl;
    std::cerr<<"-------------------------------------" << std::endl;
  };

};

std::vector<std::string> importImages(std::string path);
std::vector<std::string> getFolderInPath(std::string path);
double findAngle( cv::Point p1, cv::Point center, cv::Point p2);
double calcDistance( cv::Point2f p1, cv::Point2f p2);
// Finds the intersection of two lines, or returns false.
// The lines are defined by (o1, p1) and (o2, p2).
bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2);

void calculateMeanPoint(std::vector<cv::Point2f>& points, cv::Point2f& mean_point);

float rectIntersection(cv::Rect& r1, cv::Rect& r2);

template<class T>
void savePoints(std::string& name, std::vector<T>& points)
{
   cv::FileStorage fs(name, cv::FileStorage::WRITE);
   write( fs , "points", points );
   fs.release();
}

template<class T>
void  loadPoints(std::string& name, std::vector<T>& points)
{
  cv::FileStorage fs2(name, cv::FileStorage::READ);
  cv::FileNode kptFileNode = fs2["points"];
  read( kptFileNode, points );
  fs2.release();
}

#endif