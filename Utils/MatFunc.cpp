#include <algorithm>
#include "MatFunc.hpp"

using namespace std;
using namespace cv;

template<typename T>
vector<int>
Sub2Ind(
          int width
        , int height
        , Mat X
        , Mat Y
        )
{
  /*sub2ind(size(a), rowsub, colsub)
  sub2ind(size(a), 2     , 3 ) = 6
  a = 1 2 3 ;
  4 5 6
  rowsub + colsub-1 * numberof rows in matrix*/

  vector<int> index;
  transpose(Y,Y);
  MatConstIterator_<T> iterX = X.begin<T>();//, it_endX = X.end<T>();
  MatConstIterator_<T> iterY = Y.begin<T>();//, it_endY = Y.end<T>();
  for (int j = 0; j < X.rows; ++j,++iterX)
  {
      //running on each col of y matrix
      for (int i =0 ;i < Y.rows; ++i,++iterY )
      {
          T colsub = *iterY;
          T rowsub = *iterX;

          int res = int(rowsub + ((colsub-1)*height));
          index.push_back(res);
      }
  }
  return index;
}

//Calculate unique for input vector 
//and return ic which enable to reconstrut input from ic (based on MatLab function)
template<typename T>
void
unique(
        const vector<T>& input
      , vector<T>& out
      , vector<T>& ic
      )
{
  //get unieque value
  out = input;
  auto it = std::unique (out.begin(), out.end());
  out.resize( std::distance(out.begin(),it) );

  //ic is vector of index, which enable to transform out to Mat
  ic.reserve( input.size() );
  std::transform( input.begin(), input.end(), std::back_inserter( ic ),
                 [&]( T x )
                 {
                    return ( std::distance( out.begin(),
                             std::lower_bound( out.begin(), out.end(), x ) ) );
                 } );

}

//Fill the dst Matrix with value of src Matrix only when current idx of element is in ind_frontal list
//Implement MatLab style: "dst[ind_frontal] = src", where ind_frontal is sparse
template<typename T>
void
modify(
        Mat& src
      , Mat& dst
      , vector<int>& ind_frontal
      )
{
  MatIterator_<T> it_dst = dst.begin<T>(), it_end_dst = dst.end<T>();
  MatConstIterator_<T> it_src = src.begin<T>();//, it_end_src = src.end<T>();
  auto iter_ind_frontal = ind_frontal.begin();

  int ii = 0;
  for(MatIterator_<T> j = it_dst; j != it_end_dst ;++j)
  {
    if (ii == (*iter_ind_frontal))
    {
      *j = *it_src;
      it_src++;
      iter_ind_frontal++;
    }
    ii++;
  }
}

//Fill the dst Matrix with value of src vector only when current idx of element is in ind_frontal list
//Implement MatLab style: "dst[ind_frontal] = src", where ind_frontal is sparse
template<typename T>
void
modify(
        vector<int>& src
      , Mat& dst
      , vector<int>& ind_frontal
      )
{
  MatIterator_<T> it_dst = dst.begin<T>(), it_end_dst = dst.end<T>();
  auto it_src = src.begin(), it_src_end = src.end();
  auto iter_ind_frontal = ind_frontal.begin(), iter_ind_frontal_end = ind_frontal.end();

  int ii = 0;
  for(MatIterator_<T> j = it_dst; j != it_end_dst ;++j)
  {
    if (ii == *iter_ind_frontal)
    {
      *j = T(*it_src);
      it_src++;
      iter_ind_frontal++;
    }
    ii++;
    if(iter_ind_frontal == iter_ind_frontal_end || it_src == it_src_end)
    {
      break;
    }
  }
} 
