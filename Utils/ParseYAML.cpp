/* 
* @Author: melgor
* @Date:   2015-02-09 15:54:21
* @Last Modified 2015-03-19
* @Last Modified time: 2015-03-19 11:56:02
*/

#include "ParseYAML.hpp"

void  parseYAML(std::string& pathModel3D, Model3D& model3D)
{
  // Make sure the config file exists.
    boost::filesystem::path configPath(pathModel3D);

    if (boost::filesystem::exists(configPath) == false)
    {
        std::cout << "Unable to find config file '" << boost::filesystem::absolute(configPath).c_str() << "'." << std::endl;
        return ;
    }
    std::string configFile = boost::filesystem::absolute(configPath).native();
    YAML::Node config = YAML::LoadFile(configFile);

    std::vector<std::vector<cv::Point3f> > refU = config['0'].as<  std::vector<std::vector<cv::Point3f> > >();
    std::vector<cv::Point3f>  outA = config['1'].as<  std::vector<cv::Point3f> >();
    std::vector<cv::Point2f>  ref_XY = config['2'].as<  std::vector<cv::Point2f> >();
    std::vector<cv::Point2f>  sizeU = config['5'].as<  std::vector<cv::Point2f> >();
    std::vector<cv::Point3f>  threedee = config['7'].as<  std::vector<cv::Point3f> >();
    std::vector<std::vector<cv::Point3f> > eyemask = config['9'].as<  std::vector<std::vector<cv::Point3f> > >();

    //transform refu to cv::Mat
    cv::Mat refU_Mat = cv::Mat(cv::Size(320,320),CV_32FC3);
    for (uint x = 0; x < refU.size(); x++)
      for (uint y = 0; y < refU[0].size(); y++)
      {
        // std::cerr<<"x :" << x << " y " << y << std::endl;
        refU_Mat.at<cv::Vec3f>(x,y)[0] = refU[x][y].x;
        refU_Mat.at<cv::Vec3f>(x,y)[1] = refU[x][y].y;
        refU_Mat.at<cv::Vec3f>(x,y)[2] = refU[x][y].z;
        // std::cerr<<"end"<<std::endl;
      }

    //transform eyemask to cv::Mat
    cv::Mat eyemask_Mat = cv::Mat(cv::Size(320,320),CV_32FC3);
    for (uint x = 0; x < eyemask.size(); x++)
      for (uint y = 0; y < eyemask[0].size(); y++)
      {
        // std::cerr<<"x :" << x << " y " << y << std::endl;
        eyemask_Mat.at<cv::Vec3f>(x,y)[0] = eyemask[x][y].x;
        eyemask_Mat.at<cv::Vec3f>(x,y)[1] = eyemask[x][y].y;
        eyemask_Mat.at<cv::Vec3f>(x,y)[2] = eyemask[x][y].z;
        // std::cerr<<"end"<<std::endl;
      }

    //transform outA to cv::Mat
    cv::Mat outA_Mat = cv::Mat(cv::Size(3,3),CV_32FC1);
    for (uint p = 0; p < outA.size(); p++)
    {
      // std::cerr<<"x :" << x << " y " << y << std::endl;
      outA_Mat.at<float>(p,0) = outA[p].x;
      outA_Mat.at<float>(p,1) = outA[p].y;
      outA_Mat.at<float>(p,2) = outA[p].z;
      // std::cerr<<"end"<<std::endl;
    }

    model3D.refU     = refU_Mat;
    model3D.outA     = outA_Mat;
    model3D.eyeMask  = eyemask_Mat;
    model3D.ref_XY   = ref_XY;
    model3D.sizeU    = cv::Size(sizeU[0].x, sizeU[0].y) ;
    model3D.threedee = threedee;

}