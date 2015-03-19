#ifndef SERIALIZATION_HPP
#define SERIALIZATION_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/serialization/vector.hpp>
#include <fstream>
#include <memory>

struct Feature
{
  cv::Mat data;
  int label;
};

struct Features
{
  cv::Mat            data;
  std::vector<int> labels;
};

BOOST_SERIALIZATION_SPLIT_FREE(::Feature)
BOOST_SERIALIZATION_SPLIT_FREE(::Features)
namespace boost {
  namespace serialization {
 
    /** Serialization support for Feature */
    template<class Archive>
    void save(Archive & ar, const ::Feature& m, const unsigned int version)
    {
      size_t elem_size = m.data.elemSize();
      size_t elem_type = m.data.type();
 
      ar & m.data.cols;
      ar & m.data.rows;
      ar & elem_size;
      ar & elem_type;
      ar & m.label;
 
      const size_t data_size = m.data.cols * m.data.rows * elem_size;
      ar & boost::serialization::make_array(m.data.ptr(), data_size);
    }
    
    /** Serialization support for Feature */
    template<class Archive>
    void load(Archive & ar, ::Feature& m, const unsigned int version)
    {
      int cols, rows,label;
      size_t elem_size, elem_type;
 
      ar & cols;
      ar & rows;
      ar & elem_size;
      ar & elem_type;
      ar & label;
 
      m.data.create(rows, cols, elem_type);
      m.label = label;
      size_t data_size = m.data.cols * m.data.rows * elem_size;
      ar & boost::serialization::make_array(m.data.ptr(), data_size);
    }
    
    /** Serialization support for Features */
    template<class Archive>
    void save(Archive & ar, const ::Features& m, const unsigned int version)
    {
      size_t elem_size = m.data.elemSize();
      size_t elem_type = m.data.type();
 
      ar & m.data.cols;
      ar & m.data.rows;
      ar & elem_size;
      ar & elem_type;
      ar & m.labels;
 
      const size_t data_size = m.data.cols * m.data.rows * elem_size;
      ar & boost::serialization::make_array(m.data.ptr(), data_size);
    }
    
    /** Serialization support for Features */
    template<class Archive>
    void load(Archive & ar, ::Features& m, const unsigned int version)
    {
      int cols, rows;
      std::vector<int> labels;
      size_t elem_size, elem_type;
 
      ar & cols;
      ar & rows;
      ar & elem_size;
      ar & elem_type;
      ar & labels;
 
      m.data.create(rows, cols, elem_type);
      m.labels = labels;
      size_t data_size = m.data.cols * m.data.rows * elem_size;
      ar & boost::serialization::make_array(m.data.ptr(), data_size);
    }
    
//   Try read next object from archive
      template<class Archive, class Stream, class Obj>
      bool try_stream_next(Archive &ar, const Stream &s, Obj &o)
      {
        bool success = false;
      
        try {
          ar >> o;
          success = true;
        } catch (const boost::archive::archive_exception &e) {
          if (e.code != boost::archive::archive_exception::input_stream_error) {
            throw;
          }
        }
      
        return success;
      }
  }  
}

template<class T>
void compress(T& obj, std::string path)
{
  namespace io = boost::iostreams;
 
 
  std::ofstream ofs(path, std::ios::out | std::ios::binary);
 
  { // use scope to ensure archive and filtering stream buffer go out of scope before stream
    io::filtering_streambuf<io::output> out;
    out.push(io::zlib_compressor(io::zlib::best_speed));
    out.push(ofs);
    boost::archive::binary_oarchive oa(out);
    oa << obj;
    
  }
 
  ofs.close();
};

template<class T>
void load(T& obj, std::string path)
{
  namespace io = boost::iostreams;
 
 
  std::ifstream ifs(path, std::ios::in | std::ios::binary);
 
  {
    io::filtering_streambuf<io::input> in;
    in.push(io::zlib_decompressor());
    in.push(ifs);
 
    boost::archive::binary_iarchive ia(in);
 
    bool cont = true;
    while (cont)
    {
      cont = boost::serialization::try_stream_next(ia, ifs, obj);
    }
  }
 
  ifs.close();
}

#endif