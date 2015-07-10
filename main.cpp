/*
* @Author: melgor
* @Date:   2014-05-26 22:22:02
* @Last Modified 2015-06-25
*/
#include <chrono>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include "Frontalization/FaceExtractor.hpp"
#include "Net/FetureExtractor.hpp"
#include "Verification/Verificator.hpp"
#include "Verification/FaceDataBase.hpp"
#include "Utils/Daemon.hpp"
#include "Utils/ServerTCP.hpp"

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
    //Extract featutes from images pointed by config:Extract.ImageListDB
    FetureExtractor net_ext(conf);
    net_ext.extractAllFeatures();
  }
  else if(conf.mode == "verify")
  {
    //Check accuracy of model. It read data from config: TestModel and return final accuracy
    cout<<"Verify"<<endl;
    Verificator verificator(conf);
    verificator.verify();
  }
  else if(conf.mode == "detect")
  {
    //Detect faces on given images: 
    //if folder: for each image in folder
    //if scene: for one scene
    //and save detected faces (alignment)
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
              std::vector<cv::Mat>  outs;
              front.getFrontalFace(image,outs);
              if (outs[0].size().width != 0)
                cv::imwrite(out_path, outs[0]);
              else
                cv::imwrite(out_path, image);
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
      std::vector<cv::Mat> outFrontal;
      FaceExtractor front(conf);
      auto t12 = std::chrono::high_resolution_clock::now();
      front.getFrontalFace(image,outFrontal);
      auto t22 = std::chrono::high_resolution_clock::now();
      std::cout << "program took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
            << " milliseconds\n";
      int i = 0;
      for(auto& img : outFrontal)
      {
        cv::imwrite(std::to_string(i) +  conf.nameScene, img);
        i++;
      }
    }
  }
  else if (conf.mode == "demo")
  { 
    //Run demo of Face Verification process:
    //1. Read images given by --scene ( can be multiple images, paths separated by ',')
    //2. Run all verification process: detection of all face and verificate each
    //3. Display image with rectangle and names 
    FaceExtractor front(conf);
    FetureExtractor net_ext(conf);
    FaceDataBase face_data(conf);
    std::vector<std::string> splitteds;
    boost::split(splitteds, conf.nameScene, boost::is_any_of(","));
    for(auto& scene : splitteds)
    {
      std::cerr <<"Scene: "<< scene << std::endl;
      //get frontal face
      cv::Mat image = cv::imread(scene);;
      std::vector<cv::Mat> outFrontal;
      #ifdef __MSTIME
      auto t12 = std::chrono::high_resolution_clock::now();
      #endif
      front.getFrontalFace(image, outFrontal);
      int num_face = 0;
    
      for(auto& face : outFrontal)
      {
        //extract feature
        cv::Mat features;
        #ifdef __MSTIME
        auto t12Ef = std::chrono::high_resolution_clock::now();
        #endif
        net_ext.extractFeature(face, features);
        #ifdef __MSTIME
        auto t22Ef = std::chrono::high_resolution_clock::now();
        std::cout << "Feature Extraction took "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t22Ef - t12Ef).count()
                << " milliseconds\n";
        #endif              
        //classify image
        std::string label = face_data.returnClosestIDName(features);
        std::cerr<<"Label: "<< label << std::endl;
        cv::putText(image, label, front._faceRect[num_face].tl() + cv::Point(50,50), 
              cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, cvScalar(255,255,255), 1, CV_AA);
        cv::rectangle(image,front._faceRect[num_face],cv::Scalar::all(255),3);
        num_face++;
        // cv::imwrite("demo.jpg",outFrontal[0]);
      }
      #ifdef __MSTIME
      auto t22 = std::chrono::high_resolution_clock::now();
      std::cout << "program took "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t22 - t12).count()
              << " milliseconds\n";
      #endif
      cv::namedWindow("Demo", CV_WINDOW_AUTOSIZE );
      cv::imshow("Demo", image);
      cv::waitKey();
    }
    // cv::namedWindow("frontalize",CV_WINDOW_NORMAL);
    // cv::imshow("frontalize",outFrontal[0]);
    // cv::waitKey();
  }
  else if (conf.mode == "daemon")
  {
    //Run Deamon with watching the folder. When new image will be placed, the FV will be runned and result saved to file
    Daemon daemon(conf);
    daemon.run();
  }
  else if (conf.mode == "server")
  {
    //Run Server side of Face-Verification. Read more at Drive
    ServerTCP_Face server(conf);
    server.run();
  }
  else if( conf.mode == "compare_image")
  { //Take to faces from disk (after face detection and alignment) and return score of similarity
    FetureExtractor net_ext(conf);
    Verificator verificator(conf);
    std::vector<std::string> splitteds;
    boost::split(splitteds, conf.nameScene, boost::is_any_of(","));
    cv::Mat image_1 = cv::imread(splitteds[0]);
    cv::Mat image_2 = cv::imread(splitteds[1]);
    //extract features
    cv::Mat features_1, features_2;
    net_ext.extractFeature(image_1, features_1);
    net_ext.extractFeature(image_2, features_2);
    float score = verificator.predictFull(features_1, features_2);
    cerr<<"Score: "<< score << endl;
      
  }
  else if (conf.mode == "create_model")
  {
    //Create model of Alignment (choose which point should create model)
    std::cerr<<"Create model 2D"<<endl;
    cv::Mat image = cv::imread(conf.nameScene);
    FaceAttribute faceAtt(conf);
    FacePoints face_points;
    faceAtt.detectFacePoint(image, face_points);
    std::vector<int> values_point;
    std::vector<int> left_eye;
    std::vector<int> right_eye;
    //nose
    // values_point.push_back(30);
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
    // //left mouth
    // values_point.push_back(48);
    // //right mouth
    // values_point.push_back(54);
    // //middle mouth
    values_point.push_back(62);

    std::vector<cv::Point2f> point_model_6,point_model_68;

    for(auto& elem : values_point)
    {
      point_model_6.push_back(face_points[elem]);
      // cv::Mat cct = image.clone();
      // cv::circle(cct,face_points[0][elem],3,cv::Scalar::all(255),-1);
      // cv::imwrite(std::to_string(elem) + "facepoint.jpg",cct);
      // cerr<<"6: "<<face_points[0][elem]<<endl;

    }
    std::vector<cv::Point2f> left_eye_model;
    for(auto& elem : left_eye)
    {
      left_eye_model.push_back(face_points[elem]);
    }
    std::vector<cv::Point2f> right_eye_model;
    for(auto& elem : right_eye)
    {
      right_eye_model.push_back(face_points[elem]);
    }
    //collect all 68 points
    for(auto& elem : face_points)
    {
      point_model_68.push_back(elem);
    }
    //calculate mean_point using eye_model
    cv::Point2f center_left_eye, center_right_eye;
    calculateMeanPoint(left_eye_model,center_left_eye);
    calculateMeanPoint(right_eye_model,center_right_eye);
    point_model_6.push_back(center_left_eye);
    point_model_6.push_back(center_right_eye);

    savePoints(conf.model2D_6,point_model_6);
    savePoints(conf.model2D_68,point_model_68);

  }
  return 0;
}
