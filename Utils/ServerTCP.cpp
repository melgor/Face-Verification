#include "ServerTCP.hpp"
#include <sstream>
#include <thread>
#include <boost/algorithm/string.hpp>

using boost::asio::ip::tcp;
const int max_length = 1024;
using namespace std;

void 
session(  
          tcp::socket sock
        , ServerTCP_Face* server
       )
{
  try
  {
    for (;;)
    {
      char data[max_length];

      boost::system::error_code error;
      size_t length = sock.read_some(boost::asio::buffer(data), error);
      if (error == boost::asio::error::eof)
        break; // Connection closed cleanly by peer.
      else if (error)
        throw boost::system::system_error(error); // Some other error.
      //Choose method, which should be proccessed
      string message = std::string(data);
      vector<string> splitted_message;
      boost::split(splitted_message, message, boost::is_any_of(" "));
      if(splitted_message[0] == server->_classifyProtocol) 
      {
        //Download image
        cv::Mat image;
        bool error_down = loadImage(std::string(data), image); 
        if (error_down)
        {
          server->runFaceVerification(image);
          std::ostringstream oss;
          oss << image.size();
          std::string s = oss.str();
          boost::asio::write(sock, boost::asio::buffer( s.c_str(), length));
        }
        else
        {
          std::string error_link = server->_errorDownload + std::string(data);
          boost::asio::write(sock, boost::asio::buffer( error_link.c_str(), length));
        }
      }
      else if(splitted_message[0] == server->_statusProtocol) 
      {
        std::string s = server->returnStatus();
        boost::asio::write(sock, boost::asio::buffer( s.c_str(), length));
      }
      else if(splitted_message[0] == server->_stopProtocol) 
      {
        boost::asio::write(sock, boost::asio::buffer(server->_stopMessage.c_str(), length));
        server->stopServer();
      }
      else //command not found
      {
        boost::asio::write(sock, boost::asio::buffer(server->_notFound.c_str(), length));
      }
      
    }
  }
  catch (std::exception& e)
  {
    std::cerr << "Exception in thread: " << e.what() << "\n";
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
  LOG(WARNING)<<"Resize: "<< _config.resizeImageRatio <<" Port Number: "<< _portNumber;
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
  boost::asio::io_service io_service;
  //tcp::v4()
  tcp::acceptor a(io_service, tcp::endpoint(boost::asio::ip::address::from_string(_ipServer), _portNumber));
  cerr << a.local_endpoint().address().to_string() << " port " << _portNumber << endl;
  while(!_stop)
  {
    tcp::socket sock(io_service);
    a.accept(sock);
    std::thread(session, std::move(sock), this).detach();
  }
}
void 
ServerTCP_Face::stopServer()
{
  _stop = true;  
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
  _faceExt->getFrontalFace(image,outFrontals);
  int num_face = 0;
  std::string name_label;
  float       score_label;
  std::ostringstream osstream_result;
  for(auto& face : outFrontals)
  {
    //extract feature
    cv::Mat features;
    _netExt->extractFeature(face,features);          
    //classify image
    _faceData->returnClosestIDNameScore(features, name_label, score_label);
    LOG(WARNING)<< "Face: "<< score_label <<" Label "<< name_label;
    cv::Rect pt = _faceExt->_faceRect[num_face];
    osstream_result << "Label: "<< name_label <<" score: " << score_label << " x: " << pt.tl().x << " y: "<< pt.tl().y<< " w: " << pt.width<< " h: " << pt.height<< std::endl;
  
    //save result to file
    num_face++;
  }
  return osstream_result.str();
}

std::string 
ServerTCP_Face::returnStatus()
{
  std::ostringstream oss;
  oss << "IP: "<< _ipServer << " Port: "<< _portNumber << std::endl;
  std::string s = oss.str();
  return s;
}
