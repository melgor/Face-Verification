#include <opencv2/core/core.hpp>
#include <vector>

//Function implement Sub2Ind from MatLab
template<typename T>
std::vector<int> Sub2Ind(int width, int height, cv::Mat X, cv::Mat Y);

//Calculate unique for input vector 
//and return ic which enable to reconstrut input from ic (based on MatLab function)
template<typename T>
void unique( const std::vector<T>& input , std::vector<T>& out, std::vector<T>& ic);

//Fill the dst Matrix with value of src Matrix only when current idx of element is in ind_frontal list
//Implement MatLab style: "dst[ind_frontal] = src", where ind_frontal is sparse
template<typename T>
void modify( cv::Mat& src , cv::Mat& dst, std::vector<int>& ind_frontal);

//Fill the dst Matrix with value of src vector only when current idx of element is in ind_frontal list
//Implement MatLab style: "dst[ind_frontal] = src", where ind_frontal is sparse
template<typename T>
void modify( std::vector<int>& src, cv::Mat& dst , std::vector<int>& ind_frontal);
