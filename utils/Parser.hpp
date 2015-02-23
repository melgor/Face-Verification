#ifndef PARSER_HPP
#define PARSER_HPP

#include <boost/program_options.hpp>
#include <string>

class Parser
{
public:
	Parser();
	void read(int argc, char** argv);
	void printHelp();
	~Parser();

	boost::program_options::options_description desc;
	std::string scene;
  std::string posemodel;
  std::string facemodel;
  std::string folderpath;
	bool reset;
	bool help;
};

#endif