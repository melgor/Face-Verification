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
#include <fstream>
#include "Parser.hpp"
#include <memory>

struct Configuration
{
  bool         reset;
  std::string  nameScene;
  std::string  posemodel;
  std::string  facemodel;
  std::string  folderpath;


  void read(int argc, char** argv)
  {
    Parser parser;
    parser.read(argc, argv);

    reset          = parser.reset;
    nameScene      = parser.scene;
    posemodel      = parser.posemodel;
    facemodel      = parser.facemodel;
    folderpath     = parser.folderpath;
  }

  void print()
  {
    std::cerr<<"-------------------------------------" << std::endl;
    std::cerr<<"Configuration " << std::endl;
    std::cerr<<"nameScene: "<<nameScene << std::endl;
    std::cerr<<"posemodel: "<<posemodel << std::endl;
    std::cerr<<"facemodel: "<<facemodel << std::endl;
    std::cerr<<"folderpath: "<<folderpath << std::endl;
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

BOOST_SERIALIZATION_SPLIT_FREE(::cv::Mat)
namespace boost {
  namespace serialization {
 
    /** Serialization support for cv::Mat */
    template<class Archive>
    void save(Archive & ar, const ::cv::Mat& m, const unsigned int version)
    {
      size_t elem_size = m.elemSize();
      size_t elem_type = m.type();
 
      ar & m.cols;
      ar & m.rows;
      ar & elem_size;
      ar & elem_type;
 
      const size_t data_size = m.cols * m.rows * elem_size;
      ar & boost::serialization::make_array(m.ptr(), data_size);
    }
 
    /** Serialization support for cv::Mat */
    template<class Archive>
    void load(Archive & ar, ::cv::Mat& m, const unsigned int version)
    {
      int cols, rows;
      size_t elem_size, elem_type;
 
      ar & cols;
      ar & rows;
      ar & elem_size;
      ar & elem_type;
 
      m.create(rows, cols, elem_type);
 
      size_t data_size = m.cols * m.rows * elem_size;
      ar & boost::serialization::make_array(m.ptr(), data_size);
    }

    // Try read next object from archive
    template<class Archive, class Stream, class Obj>
    bool try_stream_next(Archive &ar, const Stream &s, Obj &o)
    {
      bool success = false;
     
      try {
        ar >> o;
        success = true;
      } catch (const boost::archive::archive_exception &e) {
        if (e.code != boost::archive::archive_exception::input_stream_error) {
          throw;
        }
      }
     
      return success;
    }
 
  }
}

#endif