#ifndef FACEDATABASE_HPP
#define FACEDATABASE_HPP
//store information about learned People. 
//For given representation it find closest people or say "Not Found"
#include "Utils/Utils.hpp"
#include "Verificator.hpp"

class FaceDataBase
{
public:
  FaceDataBase(){};
  FaceDataBase(struct Configuration& config);
  //return only ID of person
  int         returnClosestID(cv::Mat& feature);
  //return only name of Person
  std::string returnClosestIDName(cv::Mat& feature);
  //return all information about closest match in DataBase
  void returnClosestIDNameScore(cv::Mat& feature, int& id, std::string& name, float& score);
  //add new person to database, image is needed for future update
  void addNewPerson(std::string& name, cv::Mat& feature, cv::Mat& image);
  //check if name exist in current database, if yes, return true.
  bool checkName(std::string& name);
  ~FaceDataBase();

private:
  //compare if two features descrive same person or not
  float compare(cv::Mat& featureOne, cv::Mat& featureTwo);
  //save database to the disk, need to be done after changes like adding new person
  void  backupData();
  //save image in case of uploding new Models
  void saveNewPersonImage(cv::Mat& image);
  struct Features*         _dataFeatures = NULL;
  std::vector<std::string> _labelsNames;
  std::string              _unknown = "Unknown";
  //configuration
  struct Configuration    _config;
  //learning algorithm
  Verificator*            _verificator;
  float                   _threshold;
  //extension for saved images
  std::string             _extension = ".jpg";



};

#endif