#include <memory>
#include "Utils.hpp"
#include "Frontalization/FaceExtractor.hpp"
#include "Net/FetureExtractor.hpp"
#include "Verification/FaceDataBase.hpp"


//implement daemon which watch changes in directory
// for detected changes, run Face-Verfication task
class Daemon
{
public:
  Daemon(Configuration& config);
  void run();
  ~Daemon();

private:
  void initLoggig();
  void initWatcher();
  void runWatcher();
  void runFaceVerification(std::string path);
  std::string _pathWatch;
  std::string _pathLog;

  Configuration   _config;
  //models
  std::shared_ptr<FaceExtractor>   _faceExt;
  std::shared_ptr<FetureExtractor> _netExt;
  std::shared_ptr<FaceDataBase>    _faceData;
};