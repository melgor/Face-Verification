/* 
* @Author: melgor
* @Date:   2015-04-24 17:49:31
* @Last Modified 2015-05-04
* @Last Modified time: 2015-05-04 10:14:18
*/
#include <exception>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <syslog.h>
#include <glog/logging.h>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include "inotify-cxx/inotify-cxx.h"
#include "Daemon.hpp"


using namespace std;

Daemon::Daemon(Configuration& config)
{
  _pathWatch = config.watchFolder;
  _pathLog   = config.pathLog;
  _config    = config;
}

void
Daemon::run()
{
  initWatcher();
  //create model for extraction
  _config.print();
  LOG(WARNING)<<"Resize: "<< _config.resizeImageRatio;
  try {
    LOG(WARNING)<<"Create Models";
    _faceExt   = std::make_shared<FaceExtractor>(_config);
    _netExt    = std::make_shared<FetureExtractor>(_config);
    _faceData  = std::make_shared<FaceDataBase>(_config);
    LOG(WARNING)<<"Models Created";
  } catch (exception &e) {
      LOG(ERROR) << "Exception occured: " << e.what();
  } catch (...) {
      LOG(ERROR) << "unknown exception occured";
  } 

  runWatcher();

}

void 
Daemon::initWatcher()
{
  /* Our process ID and Session ID */
  pid_t pid, sid;
  
  /* Fork off the parent process */
  pid = fork();
  if (pid < 0) {
    exit(EXIT_FAILURE);
  }
  /* If we got a good PID, then
      we can exit the parent process. */
  if (pid > 0) {
    exit(EXIT_SUCCESS);
  }

  /* Change the file mode mask */
  umask(0);
  initLoggig();
    /* Create a new SID for the child process */
  sid = setsid();
  LOG(WARNING) << "PID " << sid;
  if (sid < 0) {
    /* Log the failure */
    LOG(ERROR) << "Could not create child process";
    exit(EXIT_FAILURE);
  }

  //   /* Change the current working directory */
  if ((chdir("/")) < 0) {
    /* Log the failure */
    LOG(ERROR) << "Could not change working directory";
    exit(EXIT_FAILURE);
  }
  /* Close out the standard file descriptors */
  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  close(STDERR_FILENO);



}

void 
Daemon::initLoggig()
{
  std::cerr<<"Log file: "<< (_pathLog + "info.log").c_str() << std::endl;
  google::SetLogDestination(google::WARNING, (_pathLog + "info.log").c_str());     
  google::InitGoogleLogging("log_test");
  LOG(WARNING) << "Start Logging";
  LOG(WARNING) << "Watch Director: "<< _pathWatch;
}

void
Daemon::runWatcher()
{
  try {
    Inotify notify;

    InotifyWatch watch(_pathWatch, IN_CREATE);
    notify.Add(watch);
    //infinity loop for detecting new file in folder
    for (;;) 
    {
      notify.WaitForEvents();

      size_t count = notify.GetEventCount();
      while (count > 0) {
        InotifyEvent event;
        bool got_event = notify.GetEvent(&event);

        if (got_event) {
          string mask_str;
          event.DumpTypes(mask_str);

          string filename = event.GetName();
          LOG(WARNING) << "[watch " << _pathWatch << "] "<< "event mask: \"" << mask_str << "\", "<< "filename: \"" << filename << "\"";  
          std::vector<std::string> splitteds;
          boost::split(splitteds, filename, boost::is_any_of("."));
          if(splitteds[1] == "jpeg" || splitteds[1] == "png" || splitteds[1] == "jpg")
            runFaceVerification(filename);
        }

        count--;
      }
    }
  } catch (InotifyException &e) {
      int errsv = errno;
      LOG(ERROR) << "Inotify exception occured: " << e.GetMessage() << "  "<< errsv;
  } catch (exception &e) {
      LOG(ERROR) << "Exception occured: " << e.what();
  } catch (...) {
      LOG(ERROR) << "unknown exception occured";
  }
}

bool 
replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

void 
Daemon::runFaceVerification(string path)
{
  //get frontal face
  std::string path_image = _pathWatch +  path;
  cv::Mat image = cv::imread(path_image);
  std::vector<std::string> splitteds;
  boost::split(splitteds, path_image, boost::is_any_of("."));
  std::string save_result = splitteds[0] + std::string(".txt");
  std::ofstream file_result;
  file_result.open (save_result);
  
  LOG(WARNING) <<"Image: "<< path_image << " Size: "<< image.size() <<" Result at: "<< save_result;
  std::vector<cv::Mat> outFrontal;
  _faceExt->getFrontalFace(image,outFrontal);
  int num_face = 0;
  std::string name_label;
  float       score_label;

  for(auto& face : outFrontal)
  {
    //extract feature
    cv::Mat features;
    _netExt->extractFeature(face,features);          
    //classify image
    _faceData->returnClosestIDNameScore(features,name_label,score_label);
    LOG(WARNING)<< "Face: "<< score_label <<" Label "<< name_label;
    cv::Rect pt = _faceExt->_faceRect[num_face];
    file_result << "Label: "<< name_label <<" score: " << score_label << " x: " << pt.tl().x << " y: "<< pt.tl().y<< " w: " << pt.width<< " h: " << pt.height<< std::endl;
  
    //save result to file
    num_face++;
  }
  file_result.close();

}


Daemon::~Daemon()
{

}
