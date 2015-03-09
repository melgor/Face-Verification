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
    //net 
    prototxt       = pt.get<std::string>("Net.Prototxt");
    caffemodel     = pt.get<std::string>("Net.CaffeModel");
    layer          = pt.get<std::string>("Net.Layer");
    gpu            = pt.get<bool>("Net.GPU");
    gpuID          = pt.get<int>("Net.GPU_ID");
    //Extractor
    extractorFolder    = pt.get<std::string>("Extract.folder");
    extractorImageList = pt.get<std::string>("Extract.imageListDB");
    //Verificator
    trainData       = pt.get<std::string>("Verification.trainData");
    valData         = pt.get<std::string>("Verification.valData");
    metric          = pt.get<std::string>("Verification.metric");

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

#endif