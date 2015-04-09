/*
* @Author: melgor
* @Date:   2014-05-26 22:22:02
* @Last Modified 2015-04-09
*/
#include <chrono>
#include <iostream>
#include <boost/filesystem.hpp>
#include "Frontalization/FaceExtractor.hpp"
#include "Net/FetureExtractor.hpp"
#include "Verification/Verificator.hpp"


using namespace std;
int main(int argc, char **argv)
{
  struct Configuration conf;
  conf.read(argc, argv);
  #ifdef __DEBUG
  conf.print();
  #endif
  if(conf.mode == "extract")
  {
    FetureExtractor net_ext(conf);
    net_ext.extractAllFeatures();
  }
  else if(conf.mode == "verify")
  {
    cout<<"Verify"<<endl;
    Verificator verificator(conf);
    verificator.verify();
    
  }
  else if(conf.mode == "train")
  {
    Verificator verificator(conf);
    verificator.train();

  }
  else if(conf.mode == "detect")
  {
    if (conf.folderpath != "")
    {
      //so far used when in image is only one face, centered for image. Does not run Face Detection Algorithm
      FaceExtractor front(conf);
      namespace fs = boost::filesystem;
      fs::path p(conf.folderpath);
      std::vector<string> path;
      std::vector<string> path_save;
      try
      {
        if(exists(p) && is_directory(p))// does p actually exist?
        {
          fs::recursive_directory_iterator itr(p);
          while (itr != boost::filesystem::recursive_directory_iterator())
          {
          std::string extension = itr->path().extension().string();
            if(extension == ".png" || extension == ".jpg" || extension == ".jpeg")
            {
              path.push_back(itr->path().string());
              std::string out_path = itr->path().string();
              path_save.push_back(out_path);
              std::cerr << "adding  " << itr->path().string() << std::endl;
              cv::Mat out, image = cv::imread(itr->path().string());
              std::vector<cv::Mat> images(1,image), outs;
              front.getFrontalFace(images,outs);
              if (outs[0].size().width != 0)
                cv::imwrite(out_path,outs[0]);
            }
            ++itr;
          }
        }
        else
        {
        std::cerr << p << " does not exist\n";
        assert(exists(p));
        std::cerr << p << " patterns path is not a directory\n";
        assert(is_directory(p));
        }
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
      FaceExtractor front(conf);
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
  }
  else if (conf.mode == "create_model")
  {
      std::cerr<<"Create model 2D"<<endl;
      cv::Mat image = cv::imread(conf.nameScene);
      FaceAttribute faceAtt(conf);
      std::vector<cv::Mat> v(1,image);
      std::vector<FacePoints> face_points;
      std::vector<cv::Rect> rect;
      faceAtt.detectFaceAndPoint(v, face_points, rect);
      std::cerr<<image.size() << " "<< rect[0] << std::endl;
      cv::Mat cc = v[0](rect[0]).clone();
      std::vector<int> values_point;
      std::vector<int> left_eye;
      std::vector<int> right_eye;
      //nose
      values_point.push_back(30);
      //left eye
      left_eye.push_back(36);
      left_eye.push_back(37);
      left_eye.push_back(38);
      left_eye.push_back(39);
      left_eye.push_back(40);
      left_eye.push_back(41);
      //right eye
      right_eye.push_back(42);
      right_eye.push_back(43);
      right_eye.push_back(44);
      right_eye.push_back(45);
      right_eye.push_back(46);
      right_eye.push_back(47);
      //left mouth
      values_point.push_back(48);
      //right mouth
      values_point.push_back(54);
      //middle mouth
      values_point.push_back(62);
      
      std::vector<cv::Point2f> point_model_6,point_model_68;

      for(auto& elem : values_point)
      { 
        point_model_6.push_back(face_points[0][elem]);
  //       cv::Mat cct = cc.clone();
  //       cv::circle(cct,face_points[0][elem],3,cv::Scalar::all(255),-1);
  //       cv::imwrite(std::to_string(elem) + "facepoint.jpg",cct);

      }
      std::vector<cv::Point2f> left_eye_model;
      for(auto& elem : left_eye)
      {
        left_eye_model.push_back(face_points[0][elem]);
      }
      std::vector<cv::Point2f> right_eye_model;
      for(auto& elem : right_eye)
      {
        right_eye_model.push_back(face_points[0][elem]);
      }
      //collect all 68 points
      for(auto& elem : face_points[0])
      {
        point_model_68.push_back(elem);
      }
      //calculate mean_point using eye_model
      cv::Point2f center_left_eye, center_right_eye;
      calculateMeanPoint(left_eye_model,center_left_eye);
      calculateMeanPoint(right_eye_model,center_right_eye);
      
      point_model_6.push_back(center_left_eye);
      point_model_6.push_back(center_right_eye);
      std::string name = "model2d_6poinst.xml";
      savePoints(name,point_model_6);
      name = "model2d_68poinst.xml";
      savePoints(name,point_model_68);
      
  }
  return 0;
}
