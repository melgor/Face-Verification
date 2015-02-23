/*
* @Author: melgor
* @Date:   2014-05-26 22:22:02
* @Last Modified 2015-02-23
*/
#include <chrono>
#include "frontalization/frontalization.hpp"
#include <iostream>
#include <boost/filesystem.hpp>
#include <chrono>

using namespace std;
int main(int argc, char **argv)
{
  struct Configuration conf;
  conf.read(argc, argv);
  #ifdef __DEBUG
  conf.print();
  #endif
  if (conf.folderpath != "")
  {
    //so far used when in image is only one face, centered for image. Does not run Face Detection Algorithm
    Frontalization front(conf);
    namespace fs = boost::filesystem;
    fs::path p(conf.folderpath);
    std::vector<string> path;
    std::vector<string> path_save;
    try
    {
      if(exists(p) && is_directory(p))// does p actually exist?
      {
         fs::directory_iterator end_iter;
         for (fs::directory_iterator dir_itr(p); dir_itr != end_iter; ++dir_itr)
         {
           std::string extension = dir_itr->path().extension().string();
           if(extension == ".png" || extension == ".jpg" || extension == ".jpeg")
           {
              path.push_back(dir_itr->path().string());
              std::string out_path = std::to_string(1) +dir_itr->path().string();
              path_save.push_back(out_path);
              std::cerr << "adding  " << dir_itr->path().string() << std::endl;
              cv::Mat out, image = cv::imread(dir_itr->path().string());
              front.getFrontalFace(image,out);
              std::cerr << "save  " << out_path << std::endl;
              cv::imwrite(out_path,out);
           }
         }
      }
      else
      {
       std::cerr << p << " does not exist\n";
       assert(exists(p));
       std::cerr << p << " patterns path is not a directory\n";
       assert(is_directory(p));
      }
      // std::vector<cv::Mat> v;
      // for(auto& p : path)
      // {
      //   v.push_back(cv::imread(p));
      // }
      // std::vector<cv::Mat> outFrontal;
      // front.getFrontalFace(v,outFrontal);
    }
    catch (const fs::filesystem_error& ex)
    {
     std::cerr << ex.what() << '\n';
    }
  }
  else
  {
    cv::Mat image = cv::imread(conf.nameScene);
    std::vector<cv::Mat> v(1,image);
    std::vector<cv::Mat> outFrontal;
    Frontalization front(conf);
    auto t12 = std::chrono::high_resolution_clock::now();
    front.getFrontalFace(v,outFrontal);
    auto t22 = std::chrono::high_resolution_clock::now();
    std::cout << "program took "
           << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
           << " milliseconds\n";
    int i = 0;
    for(auto& img : outFrontal)
    {
     cv::imwrite(std::to_string(i) +  conf.nameScene,img);
     i++;
    }
  }
  return 0;
}
