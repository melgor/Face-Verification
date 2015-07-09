#include "ServerTCP.hpp"
#include <sstream>
#include <thread>
#include <boost/algorithm/string.hpp>
#include <boost/array.hpp>

using boost::asio::ip::tcp;
const int max_length = 1024;
using namespace std;

void 
session(  
          tcp::socket& sock
        , ServerTCP_Face* server
       )
{
  try
  {
    for (;;)
    {
      // char data[max_length];
      boost::array<char,max_length> buf;
      boost::system::error_code error;
      size_t length = sock.read_some(boost::asio::buffer(buf), error);
      if (error == boost::asio::error::eof)
        break; // Connection closed cleanly by peer.
      else if (error)
        throw boost::system::system_error(error); // Some other error.
      //Choose method, which should be proccessed
      string message = std::string(buf.data(), length);
      vector<string> splitted_message;
      boost::split(splitted_message, message, boost::is_any_of(" "));
      if(splitted_message[0] == server->_classifyProtocol) //run Face-Verification at image
      {
        //Download image
        cv::Mat image;
        bool error_down = loadImage(splitted_message[1], image); 
        if (error_down)
        {
          std::string result = server->runFaceVerification(image);
          std::ostringstream oss;
          // oss << image.size();
          oss << result;
          std::string s = oss.str();
          boost::asio::write(sock, boost::asio::buffer( s.c_str(), s.length()));
        }
        else
        {
          std::string error_link = server->_errorDownload + splitted_message[1];
          boost::asio::write(sock, boost::asio::buffer( error_link.c_str(), error_link.length()));
        }
      }
      else if(splitted_message[0] == server->_statusProtocol) //return status of program
      {
        std::string s = server->returnStatus();
        boost::asio::write(sock, boost::asio::buffer( s.c_str(), s.length()));
      }
      else if(splitted_message[0] == server->_stopProtocol) //stop server
      {
        boost::asio::write(sock, boost::asio::buffer(server->_stopMessage.c_str(), server->_stopMessage.length()));
        server->stopServer();
        sock.close();
        break;
      }
      else if(splitted_message[0] == server->_echoProtocol)
      {  
          //return same message afer echo
          boost::asio::write(sock, boost::asio::buffer(splitted_message[1].c_str(), splitted_message[1].length()));
      }
      else //command not found
      {
        boost::asio::write(sock, boost::asio::buffer(server->_notFound.c_str(), server->_notFound.length()));
      }
      
    }
  }
  catch (std::exception& e)
  {
     LOG(ERROR) << "Exception in thread: " << e.what();
  }
}


ServerTCP_Face::ServerTCP_Face(Configuration& config)
{
  _pathLog    = config.pathLog;
  _portNumber = config.portNumber;
  _ipServer   = config.ipServer;
  _config     = config;

  _config.print();
  initLoggig();
  try {
    LOG(WARNING)<<"Create Models";
    _faceExt   = std::make_shared<FaceExtractor>(_config);
    _netExt    = std::make_shared<FetureExtractor>(_config);
    _faceData  = std::make_shared<FaceDataBase>(_config);
    LOG(WARNING)<<"Models Created";
  } catch (std::exception &e) {
      LOG(ERROR) << "Exception occured: " << e.what();
  } catch (...) {
      LOG(ERROR) << "unknown exception occured";
  } 
}

void
ServerTCP_Face::run()
{ 
  
  //tcp::v4()
  try
  {
    tcp::acceptor a(_io_service, tcp::endpoint(boost::asio::ip::address::from_string(_ipServer), _portNumber));
    std::cerr << "Server Started: "<< a.local_endpoint().address().to_string() << " port " << _portNumber << std::endl;
    LOG(WARNING) << "Server Started: "<< a.local_endpoint().address().to_string() << " port " << _portNumber;
    while(!_stop)
    {
      tcp::socket sock(_io_service);
      a.accept(sock);
      // std::thread(session, std::move(sock), this).detach(); //this is used for asynchronus connection, problem with stoping all sockets
      session(sock,this);
    }
  }
  catch (std::exception& e)
  {
     LOG(ERROR) << "Exception while creating server: " << e.what();
     return;
  }

 
}
void 
ServerTCP_Face::stopServer()
{
  _stop = true;  
  // _io_service.stop();
  LOG(WARNING) << "Server Stoped";
}

void 
ServerTCP_Face::initLoggig()
{
  std::cerr<<"Log file: "<< (_pathLog + "info.log").c_str() << std::endl;
  google::SetLogDestination(google::WARNING, (_pathLog + "info.log").c_str());     
  google::InitGoogleLogging("log_test");
  LOG(WARNING) << "Start Logging";
}

std::string 
ServerTCP_Face::runFaceVerification(cv::Mat& image)
{
  LOG(WARNING) <<" Size: "<< image.size();
  //get frontal face
  std::vector<cv::Mat> outFrontals;
  _faceExt->getFrontalFace(image, outFrontals);
  
  std::string name_label;
  float       score_label;
  int         id;
  std::ostringstream osstream_result;
  int num_face = 0;
  for(auto& face : outFrontals)
  {
    //extract feature
    cv::Mat features;
    _netExt->extractFeature(face,features);          
    //classify image
    _faceData->returnClosestIDNameScore(features, id, name_label, score_label);
    cv::Rect pt = _faceExt->_faceRect[num_face];
    osstream_result << "Label: "<< name_label <<" score: " << score_label << " x: " << pt.tl().x << " y: "<< pt.tl().y<< " w: " << pt.width<< " h: " << pt.height<< std::endl;
    num_face++;
  }
  if(num_face == 0)
    osstream_result << "No Face Detected" << std::endl;

  return osstream_result.str();
}

std::string 
ServerTCP_Face::returnStatus()
{
  std::ostringstream oss;

  oss <<"Version:" << _versionFV << " IP: "<< _ipServer << " Port: "<< _portNumber << " Log Path: "<< _pathLog <<" PID: "<< ::getpid() << std::endl;
  std::string s = oss.str();
  return s;
}

ServerTCP_Face::~ServerTCP_Face()
{
  
}
