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
  std::string runFaceVerification(cv::Mat& image);
  std::string returnStatus();
   //type of message
  std::string _classifyProtocol  = "classify";
  std::string _statusProtocol    = "status";
  std::string _stopProtocol      = "stop";
  std::string _echoProtocol      = "echo";
  std::string _notFound          = "Command Not Found, available command: classify <link>, status, stop";
  std::string _stopMessage       = "Server will be stopped";
  std::string _errorDownload     = "Error when download link: ";
  ~ServerTCP_Face();
  
private:
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
 
 
}; 
#endif //SERVERTCP_HPP