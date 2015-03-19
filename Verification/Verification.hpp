#ifndef VERIFICATION_HPP
#define VERIFICATION_HPP
#include "Utils/Utils.hpp"

//verification Process

//declaration of struct, defined in serialization
struct Features;

class Verificator
{
  public:
    Verificator(struct Configuration& config);
    //get id of closest feature from train data
    int verify(cv::Mat& features);
    //verify validation data
    void verifyVal();
    //verify if same person or not
    void verifyValPerson();
    ~Verificator();

  private:
    std::string      _metric;
    std::string      _pathTrainFeatures;
    std::string      _pathValFeatures;
    struct Features* _trainFeatures;
   
  
};

#endif 
