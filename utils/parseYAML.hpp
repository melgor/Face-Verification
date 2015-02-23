#ifndef YAML_HPP
#define YAML_HPP
#include <boost/filesystem.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/opencv.hpp>

struct Model3D
{
  cv::Mat                       refU;
  cv::Mat                       outA;
  cv::Mat                       eyeMask;
  std::vector<cv::Point2f>      ref_XY;
  cv::Size                      sizeU;
  std::vector<cv::Point3f>      threedee;
};


namespace YAML {

template<>
struct convert<cv::Point3f> {
  static Node encode(const cv::Point3f& rhs) {
    Node node;
    node.push_back(rhs.x);
    node.push_back(rhs.y);
    node.push_back(rhs.z);
    return node;
  }

  static bool decode(const Node& node, cv::Point3f& rhs) {
    if(!node.IsSequence() || node.size() != 3) {
      return false;
    }

    rhs.x = node[0].as<double>();
    rhs.y = node[1].as<double>();
    rhs.z = node[2].as<double>();
    return true;
  }
};

template<>
struct convert<cv::Point2f> {
  static Node encode(const cv::Point2f& rhs) {
    Node node;
    node.push_back(rhs.x);
    node.push_back(rhs.y);
    return node;
  }

  static bool decode(const Node& node, cv::Point2f& rhs) {
    if(!node.IsSequence() || node.size() != 2) {
      return false;
    }

    rhs.x = node[0].as<double>();
    rhs.y = node[1].as<double>();
    return true;
  }
};

template<>
struct convert<std::vector<cv::Point3f> > {
  static Node encode(const std::vector<cv::Point3f>& rhs) {
    Node node;

    return node;
  }

  static bool decode(const Node& node, std::vector<cv::Point3f>& rhs) {

    for(uint i = 0; i < node.size();i++)
    {
     cv::Point3f tmp = node[i].as<cv::Point3f>();
     rhs.push_back(tmp);
    }
    return true;
  }
};

template<>
struct convert<std::vector<cv::Point2f> > {
  static Node encode(const std::vector<cv::Point2f>& rhs) {
    Node node;

    return node;
  }

  static bool decode(const Node& node, std::vector<cv::Point2f>& rhs) {

    for(uint i = 0; i < node.size();i++)
    {
     cv::Point2f tmp = node[i].as<cv::Point2f>();
     rhs.push_back(tmp);
    }
    return true;
  }
};

template<>
struct convert< std::vector<std::vector<cv::Point3f> > > {
  static Node encode(const std::vector<cv::Point3f>& rhs) {
    Node node;

    return node;
  }

  static bool decode(const Node& node, std::vector<std::vector<cv::Point3f> >& rhs) {

    for(uint i = 0; i < node.size();i++)
    {
     std::vector<cv::Point3f> tmp = node[i].as< std::vector<cv::Point3f> >();
     rhs.push_back(tmp);
    }
    return true;
  }
};

}

void  parseYAML(std::string& pathModel3D, Model3D& model3D);

#endif 

