#ifndef SERVERTCP_HPP
#define SERVERTCP_HPP

#include <cstdlib>
#include <iostream>
#include <memory>
#include <utility>
#include <boost/asio.hpp>
#include <memory>
#include "Utils.hpp"
#include "Frontalization/FaceExtractor.hpp"
#include "Net/FetureExtractor.hpp"
#include "Verification/FaceDataBase.hpp"


/*
* Communication Server between client and Face Verification server. 
* Protocol:
*  - status   -> return current status, like port and IP adress 
*  - stop     -> stop the server
*  - class <link> -> run  Fave Verification on image, which is download from net
*/ 

class ServerTCP_Face
{
public:
  ServerTCP_Face(Configuration& config);
  void run();
  void stopServer();
  //Run Face Verification on given Image
  std::string runFaceVerification(cv::Mat& image);
  //Return current status of Server
  std::string returnStatus();
  //add New Person to DataBase
  std::string addNewPerson(std::string& name, cv::Mat& image);
  //type of message which are accepted by server
  std::string _classifyProtocol  = "classify";
  std::string _statusProtocol    = "status";
  std::string _stopProtocol      = "stop";
  std::string _echoProtocol      = "echo";
  std::string _addPersonProtocol = "add";
  //Answers to client, when some error occur
  std::string _notFound          = "Command Not Found, available command: classify <link>, status, stop, echo, add <name> <link>";
  std::string _stopMessage       = "Server will be stopped";
  std::string _errorDownload     = "Error when download link: ";
  std::string _argumentAddPerson = "You should provide 2 arguments to add new person, name and link (pointing to image in Web)";
  ~ServerTCP_Face();
  
private:
  //Initilizing logging 
  void initLoggig();
  //Configuration file
  Configuration   _config;
  std::string     _pathLog;
  int             _portNumber;
  std::string     _ipServer;
  bool            _stop = false;
  boost::asio::io_service _io_service;
  //models
  std::shared_ptr<FaceExtractor>   _faceExt;
  std::shared_ptr<FetureExtractor> _netExt;
  std::shared_ptr<FaceDataBase>    _faceData;

  std::string     _versionFV = "0.1";

  //message: adding new person
  std::string     _noFaceDetected = "No Face Detected in Image, try other image";
  std::string     _addedPerson    = "Added New Person";
  std::string     _idExist        = " exist in current DataBase, duplicats are not allowed";
  
 
}; 
#endif //SERVERTCP_HPP